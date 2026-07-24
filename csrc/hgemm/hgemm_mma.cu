#include <cstdint>
#include <cuda_runtime.h>
#include "common/common.h"
#include "common/tester.h"
#include "common/util.h"
#include "ptx.cuh"
#include "swizzle.cuh"

using namespace nvcuda;
__device__ __forceinline__ void ld_st_128bit(void *dst, void *src) {
    *reinterpret_cast<float4 *>(dst) = *reinterpret_cast<float4 *>(src);
} 
__device__ __forceinline__ void ld_st_32bit(void *dst, void *src) {
    *reinterpret_cast<half2 *>(dst) = *reinterpret_cast<half2 *>(src);
}

/*
    @ C = A * B
        * tileB[K][N]
        * ldmatrix_trans_sync
    @ C = A * B^T
        * 一般可以先进入kernel之前就把B转置成B^T,kernel本身不变
    @ldmatrix+自定义uint32_t寄存器
*/
template<const int MMA_M = 16,
         const int MMA_N = 8,
         const int MMA_K = 16>
__global__ void hgemm_mma_m16n8k16_ldmatrix_kernel(half *A, half *B, half *C, unsigned int M, unsigned int N, unsigned int K) {

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, MMA_K);
    constexpr int BM = MMA_M;
    constexpr int BN = MMA_N;
    constexpr int BK = MMA_K;

    __shared__ half tileA[MMA_M][MMA_K];
    __shared__ half tileB[MMA_K][MMA_N];
    __shared__ half tileC[MMA_M][MMA_N];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int lane_id = tid % 32;

    const int load_smem_a_m = tid / 2;
    const int load_smem_a_k = (tid % 2) * 8;
    const int load_smem_b_k = tid;
    const int load_smem_b_n = 0;
    const int load_gmem_a_m = by * BM + load_smem_a_m;
    const int load_gmem_b_n = bx * BN + load_smem_b_n;
    if(load_gmem_a_m >= M && load_gmem_b_n >= N) return;
   
    uint32_t RC[2] = {0, 0};
    for(int k = 0; k < NUM_K_TILES; k ++) {
        int load_gmem_a_k = k * BK + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        ptx::cp_async_cg<16>(&tileA[load_smem_a_m][load_smem_a_k], &A[load_gmem_a_addr]);
        
        if(lane_id < MMA_K) {
            int load_gmem_b_k = k * MMA_K + load_smem_b_k;
            int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
            ptx::cp_async_cg<16>(&tileB[load_smem_b_k][load_smem_b_n], &B[load_gmem_b_addr]);
        }
        ptx::cp_async_commit_group();
        ptx::cp_async_wait_group<0>();
        __syncthreads();

        uint32_t RA[4];
        uint32_t RB[2];

        ptx::ldmatrix_sync(RA, &tileA[lane_id % 16][(lane_id/16)*8]);
        ptx::ldmatrix_trans_sync(RB, &tileB[lane_id % 16][0]);
        ptx::mma_sync_m16n8k16(RC, RA, RB);
        __syncthreads();
    }

    ld_st_32bit(&tileC[lane_id / 4][(lane_id % 4) * 2], &RC[0]);
    ld_st_32bit(&tileC[lane_id / 4 + 8][((lane_id % 4) * 2)], &RC[1]);

    __syncthreads();
    if(lane_id < MMA_M) {
        int store_gmem_c_m = by * BM + lane_id;
        int store_gmem_c_n = bx * BN;
        int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
        ld_st_128bit(&C[store_gmem_c_addr], &tileC[lane_id][0]);
    }
    return;
}
void hgemm_mma_m16n8k16_ldmatrix(half *A, half *B, half *C, unsigned int M, unsigned int N, unsigned K) {

    constexpr int MMA_M = 16; 
    constexpr int MMA_K = 16; 
    constexpr int MMA_N = 8; 

    dim3 block(32);
    dim3 grid(div_ceil(N, MMA_N), div_ceil(M, MMA_M));
    hgemm_mma_m16n8k16_ldmatrix_kernel<MMA_M, MMA_N, MMA_K><<<grid, block>>>(A, B, C, M, N, K);
    return;
}

/*
    @ C = A * B^T
        * wmma::col_major
        * ldmatrix_sync
        * swap R1 and R2
    @bank conflict solved by swizzle
    @16x16 half
    @wmma寄存器
*/
__global__ void hgemm_mma16x16_swizzle_kernel(half *A, half *B, half *C) {

    __shared__ half tileA[16*16];
    __shared__ half tileB[16*16];
    __shared__ half tileC[16*16];

    int tx = threadIdx.x;
    uint32_t gAddr = tx * 8;
    auto g2sAddr = swizzle<3, 1, 3>(gAddr);
    ld_st_128bit(tileA + g2sAddr, A+gAddr);
    ld_st_128bit(tileB + g2sAddr, B+gAddr);
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    uint32_t rAddr = (tx % 16) * 16 + (tx / 16) * 8;
    auto r2sAddr = swizzle<3, 1, 3>(rAddr);

    ptx::ldmatrix_sync(a_frag.x, tileA + r2sAddr);
    ptx::ldmatrix_sync(b_frag.x, tileB +r2sAddr);

    half2 tmp = HALF2(b_frag.x[2]);
    HALF2(b_frag.x[2]) = HALF2(b_frag.x[4]);
    HALF2(b_frag.x[4]) = tmp;

    // mma_sync(c_frag, a_frag, b_frag, c_frag);
    ptx::mma_sync_m16n8k16(c_frag.x, a_frag.x, b_frag.x);
    ptx::mma_sync_m16n8k16(c_frag.x + 4, a_frag.x, b_frag.x + 4);
    ptx::stmatrix_sync(tileC + r2sAddr, c_frag.x);

    ld_st_128bit(C + gAddr, tileC + g2sAddr);
}
void hgemm_mma16x16_swizzle(half *A, half *B, half*C, int M, int N, int K) {
    dim3 block(32);
    dim3 grid(1);
    hgemm_mma16x16_swizzle_kernel<<<grid, block>>>(A, B, C);
    return;
}

/*
    @ C = A * B^T
        * wmma::col_major
        * ldmatrix_sync
        * swap R1 and R2
    @bank conflict solved by swizzle
    @16x64 half
    @wmma寄存器
 */
__global__ void hgemm_mma16x64_swizzle_kernel(half *A, half *B, half *C) {
    __shared__ half smem_a[16 * 64];
    __shared__ half smem_b[16 * 64];
    __shared__ half smem_c[16 * 64];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = tx + ty * blockDim.x;
    // swizzle load A and B
    constexpr int stride = 64;
    int gRow = tid * 8 / stride;
    int gCol = tid * 8 % stride;

    int g2sRow = gRow;
    // [xxxx] [xxx] [xxx]
    // [16row] [8col] [8fp16]
    int g2sCol = gCol ^ ((gRow & 0x7) << 3);

    ld_st_128bit(smem_a + g2sRow * stride + g2sCol, A + tid * 8);
    ld_st_128bit(smem_b + g2sRow * stride + g2sCol, B + tid * 8);
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);
    // swizzle load frag a and b
    int rRow = tx % 16;
    int rCol = (ty * 2 + tx / 16) * 8;
    int r2sRow = rRow;
    int r2sCol = rCol ^ ((rRow & 0x7) << 3);
    ptx::ldmatrix_sync(a_frag.x, smem_a + r2sRow * stride + r2sCol);
    ptx::ldmatrix_sync(b_frag.x, smem_b + r2sRow * stride + r2sCol);
    // swap R1 and R2 of B, this is required by B's layout, more info see PTX
    half2 tmp = HALF2(b_frag.x[2]);
    HALF2(b_frag.x[2]) = HALF2(b_frag.x[4]);
    HALF2(b_frag.x[4]) = tmp;
    // calc and store
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    // store_matrix_sync(smem_c + 16 * ty, c_frag, 16 * 4, mem_row_major);
    ptx::stmatrix_sync(smem_c + 16 * ty + (tx % 16) * 64 + (tx / 16) * 8,
                       c_frag.x);
    __syncthreads();
    ld_st_128bit(C + 8 * tid, smem_c + 8 * tid);
}

/**
 * \brief 2 patterns are resolved in this kernel, one is a row 1x256 regarded as
 * 16x16, the other is block 16x16
 *
 * \note this kernel has serious LDSM bank
 * conflicts calculated as follows
 *
 * Pattern 1: 1x256 regarded as 16x16, each 1x256 has 4 bank conflicts, result
 * in 4x16(rows)x2(matrix A/B) = 128 bank conflicts
 *
 * Pattern 2: 16x16 block, each 16x16 has 7x4 bank conflicts, result in
 * 7x4x16(blocks)x2(matrix A/B) = 896 bank conflicts
 *
 * Total bank conflicts = 128 + 896 = 1024
 */
__global__ void mma_multi_pattern_simple(half *A, half *B, half *C) {
    __shared__ half smem_a[16 * 256];
    __shared__ half smem_b[16 * 256];
    __shared__ half smem_c[16 * 256];

    int tx = threadIdx.x; // 0-31
    int ty = threadIdx.y; // 0-15

    int tid = tx + ty * blockDim.x;
    // TODO: swizzle load A and B
    ld_st_128bit(smem_a + tid * 8, A + tid * 8);
    ld_st_128bit(smem_b + tid * 8, B + tid * 8);

    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // 1x256 regarded as 16x16, compute C = A * B^T
    load_matrix_sync(a_frag, smem_a + 256 * ty, 16);
    load_matrix_sync(b_frag, smem_b + 256 * ty, 16);

    mma_sync(c_frag, a_frag, b_frag, c_frag);

    store_matrix_sync(smem_c + 256 * ty, c_frag, 16, wmma::mem_row_major);

    __syncthreads();

    // 16x16 block, compute C = A * B^T
    load_matrix_sync(a_frag, smem_a + 16 * ty, 256);
    load_matrix_sync(b_frag, smem_b + 16 * ty, 256);

    fill_fragment(c_frag, 0.0f);
    mma_sync(c_frag, a_frag, b_frag, c_frag);

    store_matrix_sync(smem_c + 16 * ty, c_frag, 256, wmma::mem_row_major);

    __syncthreads();

    ld_st_128bit(C + tid * 8, smem_c + tid * 8);
}

__global__ void mma_multi_pattern_swizzle(half *A, half *B, half *C) {
    __shared__ half smem_a[16 * 256];
    __shared__ half smem_b[16 * 256];
    __shared__ half smem_c[16 * 256];

    int tx = threadIdx.x; // 0-31
    int ty = threadIdx.y; // 0-15
    int tid = tx + ty * blockDim.x;

    // swizzle load A and B
    // [xxxx]    [xxxxx]    [xxx]
    // [16rows]  [32cols]    [8fp16]
    // split cols into 4 groups:
    // [xxxx]    [xx] [xxx]     [xxx]

    uint32_t gAddr = tid * 8;
    auto g2sAddr = swizzle<3, 2, 3>(swizzle<5, 3, 3>(gAddr));
    ld_st_128bit(smem_a + g2sAddr, A + gAddr);
    ld_st_128bit(smem_b + g2sAddr, B + gAddr);

    __syncthreads();

    using namespace nvcuda::wmma;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;
    // 1x256 regarded as 16x16
    uint32_t rAddr = ty * 256 + (tx % 16) * 16 + (tx / 16 * 8);
    auto r2sAddr = swizzle<3, 2, 3>(swizzle<5, 3, 3>(rAddr));

    ptx::ldmatrix_sync(a_frag.x, smem_a + r2sAddr);
    ptx::ldmatrix_sync(b_frag.x, smem_b + r2sAddr);

    half2 tmp = HALF2(b_frag.x[2]);
    HALF2(b_frag.x[2]) = HALF2(b_frag.x[4]);
    HALF2(b_frag.x[4]) = tmp;

    wmma::fill_fragment(c_frag, 0.0f);
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    store_matrix_sync(smem_c + 256 * ty, c_frag, 16, mem_row_major);
    __syncthreads();
    // 16x16 blocks
    rAddr = ty * 16 + (tx % 16) * 256 + (tx / 16 * 8);
    r2sAddr = swizzle<3, 2, 3>(swizzle<5, 3, 3>(rAddr));

    ptx::ldmatrix_sync(a_frag.x, smem_a + r2sAddr);
    ptx::ldmatrix_sync(b_frag.x, smem_b + r2sAddr);

    tmp = HALF2(b_frag.x[2]);
    HALF2(b_frag.x[2]) = HALF2(b_frag.x[4]);
    HALF2(b_frag.x[4]) = tmp;

    wmma::fill_fragment(c_frag, 0.0f);
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    store_matrix_sync(smem_c + 16 * ty, c_frag, 256, mem_row_major);
    __syncthreads();
    ld_st_128bit(C + tid * 8, smem_c + tid * 8);
}

int main() {
    Tester tester(512, 2048, 1024, 1, 10, 100, true);
    const int opt = 1;
    if(opt == 1) {
        tester.evaluate(hgemm_mma_m16n8k16_ldmatrix, "hgemm_mma_m16n16k16_ldmatrix_kernel");
    }
    return 0;
}