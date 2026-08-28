#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>


namespace spec{
    using namespace cute;

    template<typename OutType_,
             typename ComputeTypeA_,
             typename ComputeTypeB_,
             typename ComputeTypeC_,
             int kBlockM_ = 128,
             int kBlockN_ = 128,
             int kBlockK_ = 64>
    struct KernelSpec{
        using OutType = OutType_;
        using ComputeTypeA = ComputeTypeA_;
        using ComputeTypeB = ComputeTypeB_;
        using ComputeTypeC = ComputeTypeC_;

        static constexpr int kBlockM = kBlockM_;
        static constexpr int kBlockN = kBlockN_;
        static constexpr int kBlockK = kBlockK_;

        using MMA_op = SM80_16x8x16_F32BF16BF16F32_TN;
        using MMA_traits = MMA_Traits<MMA_op>;
        using MMA_atom = MMA_Atom<MMA_traits>;
        using MMA_shape = MMA_traits::Shape_MNK;

        static constexpr int kMmaThrExpandM = 2;
        static constexpr int kMmaThrExpandN = 4;
        static constexpr int kMmaThrExpandK = 1;

        static constexpr int kMmaValExpandM = 1;
        static constexpr int kMmaValExpandN = 1;
        static constexpr int kMmaValExpandK = 2;

        static constexpr int kMmaTileM = kMmaThrExpandM * kMmaValExpandM * get<0>(MMA_shape{});
        static constexpr int kMmaTileN = kMmaThrExpandN * kMmaValExpandN * get<1>(MMA_shape{});
        static constexpr int kMmaTileK = kMmaThrExpandK * kMmaValExpandK * get<2>(MMA_shape{});
        
        // (M, N, K)->warp_idx(MMA Atom)
        using MMAThrLayout = decltype(make_layout(make_shape(Int<kMmaThrExpandM>{}, Int<kMmaThrExpandN>{}, Int<kMmaThrExpandK>{})));
        // Permutation Layout
        using MMATileLayout = Tile<Int<kMmaTileM>, Int<kMmaTileN>, Int<kMmaTileK>>;
        using TiledMMA = decltype(make_tiled_mma(MMA_op{}, MMAThrLayout{}, MMATileLayout{}));

        using Copy_G2S_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
        using Copy_S2R_op = AutoVectorizingCopy;

        using CopyA_G2S_atom = Copy_Atom<Copy_G2S_op, ComputeTypeA>;
        using CopyB_G2S_atom = Copy_Atom<Copy_G2S_op, ComputeTypeB>;
        using CopyC_G2S_atom = Copy_Atom<Copy_G2S_op, ComputeTypeC>;

        using CopyA_S2R_atom = Copy_Atom<Copy_S2R_op, ComputeTypeA>;
        using CopyB_S2R_atom = Copy_Atom<Copy_S2R_op, ComputeTypeB>;
        using CopyC_S2R_atom = Copy_Atom<Copy_S2R_op, ComputeTypeC>;

        using Copy_R2S_op = AutoVectorizingCopy;
        using Copy_S2G_op = AutoVectorizingCopy;

        using CopyC_R2S_atom = Copy_Atom<Copy_R2S_op, ComputeTypeC>;
        using CopyO_R2S_atom = Copy_Atom<Copy_R2S_op, OutType>;

        using CopyC_S2G_atom = Copy_Atom<Copy_S2G_op, ComputeTypeC>;
        using CopyO_S2G_atom = Copy_Atom<Copy_S2G_op, OutType>;


        static constexpr int kThreadNum = size(TiledMMA{});
        static constexpr int kBlockK_Copy = cute::min(64, kBlockK) / 8;
        static constexpr int kBlockN_Copy = cute::min(64, kBlockN) / 8;

        using TiledCopyA_G2S = decltype(make_tiled_copy(CopyA_G2S_atom{}, 
                    make_layout(make_shape(Int<kThreadNum / kBlockK_Copy>{}, Int<kBlockK_Copy>{}), make_stride(Int<kBlockK_Copy>{}, Int<1>{})), 
                make_layout(make_shape(Int<1>{}, Int<8>{}))));

        using TiledCopyB_G2S = decltype(make_tiled_copy(CopyB_G2S_atom{}, 
                    make_layout(make_shape(Int<kThreadNum / kBlockK_Copy>{}, Int<kBlockK_Copy>{}), make_stride(Int<kBlockK_Copy>{}, Int<1>{})), 
                make_layout(make_shape(Int<1>{}, Int<8>{}))));
        using TiledCopyC_G2S = decltype(make_tiled_copy(CopyC_G2S_atom{}, 
                    make_layout(make_shape(Int<kThreadNum / kBlockN_Copy>{}, Int<kBlockN_Copy>{}), make_stride(Int<kBlockN_Copy>{}, Int<1>{})),
                make_layout(make_shape(Int<1>{}, Int<8>{}))));

        using TiledCopyA_S2R = decltype(make_tiled_copy_A(CopyA_S2R_atom{}, TiledMMA{}));
        using TiledCopyB_S2R = decltype(make_tiled_copy_B(CopyB_S2R_atom{}, TiledMMA{}));
        using TiledCopyC_S2R = decltype(make_tiled_copy_C(CopyC_S2R_atom{}, TiledMMA{}));

        using TiledCopyC_R2S = decltype(make_tiled_copy_C(CopyC_R2S_atom{}, TiledMMA{}));
        using TiledCopyO_R2S = decltype(make_tiled_copy_C(CopyO_R2S_atom{}, TiledMMA{}));

        using TiledCopyC_S2G = decltype(make_tiled_copy(CopyC_S2G_atom{}, 
                    make_layout(make_shape(Int<kThreadNum / kBlockK_Copy>{}, Int<kBlockK_Copy>{}), make_stride(Int<kBlockK_Copy>{}, Int<1>{})), 
                make_layout(make_shape(Int<1>{}, Int<8>{}))));
        using TiledCopyO_S2G = decltype(make_tiled_copy(CopyO_S2G_atom{}, 
                    make_layout(make_shape(Int<kThreadNum / kBlockN_Copy>{}, Int<kBlockN_Copy>{}), make_stride(Int<kBlockN_Copy>{}, Int<1>{})),
                make_layout(make_shape(Int<1>{}, Int<8>{})))); 

        
        using SmemLayoutA = decltype(make_layout(make_shape(Int<kBlockM>{}, Int<kBlockK>{}), make_stride(Int<kBlockK>{}, Int<1>{})));
        using SmemLayoutB = decltype(make_layout(make_shape(Int<kBlockN>{}, Int<kBlockK>{}), make_stride(Int<kBlockK>{}, Int<1>{})));
        using SmemLayoutC = decltype(make_layout(make_shape(Int<kBlockM>{}, Int<kBlockN>{}), make_stride(Int<kBlockN>{}, Int<1>{})));
        using SmemLayoutO = decltype(make_layout(make_shape(Int<kBlockM>{}, Int<kBlockN>{}), make_stride(Int<kBlockN>{}, Int<1>{})));

        static constexpr int kShmSizeA = cosize(SmemLayoutA{}) * sizeof(ComputeTypeA);
        static constexpr int kShmSizeB = cosize(SmemLayoutB{}) * sizeof(ComputeTypeB);
        static constexpr int kShmSizeC = cosize(SmemLayoutC{}) * sizeof(ComputeTypeC);
        static constexpr int kShmSizeO = cosize(SmemLayoutO{}) * sizeof(OutType);

        static constexpr int kShmSize = cute::max( kShmSizeA + kShmSizeB + kShmSizeC, kShmSizeO);
    };
}

template<typename Spec, bool IsGemm, bool IsCvtPrecision>
__global__ void hgemm_cute(void *__restrict__ Cptr, const void *__restrict__ Aptr, const void *__restrict__ Bptr, int m, int n, int k, void *__restrict__ Outptr) {

    using namespace cute;

    using X = Underscore;
    using MMA_shape = typename Spec::MMA_shape;
    using OutType = typename Spec::OutType;
    using ComputeTypeA = typename Spec::ComputeTypeA;
    using ComputeTypeB = typename Spec::ComputeTypeB;
    using ComputeTypeC = typename Spec::ComputeTypeC;
    using SmemLayoutA = typename Spec::SmemLayoutA;
    using SmemLayoutB = typename Spec::SmemLayoutB;
    using SmemLayoutC = typename Spec::SmemLayoutC;
    using SmemLayoutO = typename Spec::SmemLayoutO;

    constexpr int kBlockM = Spec::kBlockM;
    constexpr int kBlockN = Spec::kBlockN;
    constexpr int kBlockK = Spec::kBlockK;
    constexpr int kShmSizeA = Spec::kShmSizeA;
    constexpr int kShmSizeB = Spec::kShmSizeB;

    extern __shared__ __align__(1024) uint8_t smem[];

    uint8_t *Aptr_smem = smem;
    uint8_t *Bptr_smem = smem + kShmSizeA;
    uint8_t *Cptr_smem = smem + kShmSizeA + kShmSizeB;
    uint8_t *Optr_smem = smem;

    int tid = threadIdx.x;

    Tensor mA = make_tensor(make_gmem_ptr((ComputeTypeA*)Aptr), make_shape(m, k), make_stride(k, Int<1>{}));        
    Tensor mB = make_tensor(make_gmem_ptr((ComputeTypeB*)Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor mC = make_tensor(make_gmem_ptr((ComputeTypeC*)Cptr), make_shape(m, n), make_stride(n, Int<1>{}));
    Tensor m0 = make_tensor(make_gmem_ptr((ComputeTypeC*)Outptr), make_shape(m, n), make_stride(n, Int<1>{}));

    auto tiler = make_tile(Int<kBlockM>{}, Int<kBlockN>{}, Int<kBlockK>{});
    auto coord = make_coord(0, 0, 0);  

    Tensor gA = local_tile(mA, tiler, coord, Step<_1, X, _1>{});
    Tensor gB = local_tile(mB, tiler, coord, Step<X, _1, _1>{}); 
    Tensor gC = local_tile(mC, tiler, coord, Step<_1, _1, X>{});
    Tensor g0 = local_tile(m0, tiler, coord, Step<_1, _1, X>{});

    Tensor sA = make_tensor(make_smem_ptr((ComputeTypeA*)Aptr_smem), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr((ComputeTypeB*)Bptr_smem), SmemLayoutB{});
    Tensor sC = make_tensor(make_smem_ptr((ComputeTypeC*)Cptr_smem), SmemLayoutC{});
    Tensor s0 = make_tensor(make_smem_ptr((OutType*)Optr_smem), SmemLayoutO{});

    typename Spec::TiledMMA tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_slice(tid);

    Tensor tCgA = thr_mma.partition_A(gA);  // (MMA, MMA_M, MMA_K)
    Tensor tCgB = thr_mma.partition_B(gB);
    Tensor tCgC = thr_mma.partition_C(gC);

    Tensor tCrA = thr_mma.partition_fragment_A(gA); // (MMA, MMA_M, MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(gB);
    Tensor tCrC = thr_mma.partition_fragment_C(gC);

    //--- Copy all global matrix Tile A/B/C to SMEM
    typename Spec::TiledCopyA_G2S g2s_tiled_copy_a;
    ThrCopy g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(tid);
    Tensor tAgA_g2s = g2s_thr_copy_a.partition_S(gA); // (CPY, CPY_M, CPY_K)
    Tensor tAsA_g2s = g2s_thr_copy_a.partition_D(sA);

    typename Spec::TiledCopyB_G2S g2s_tiled_copy_b;
    ThrCopy g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(tid);
    Tensor tBgB_g2s = g2s_thr_copy_b.partition_S(gB); // (CPY, CPY_N, CPY_K)
    Tensor tBsB_g2s = g2s_thr_copy_b.partition_D(sB);


    typename Spec::TiledCopyC_G2S g2s_tiled_copy_c;
    ThrCopy g2s_thr_copy_c = g2s_tiled_copy_c.get_slice(tid);
    Tensor tCgC_g2s = g2s_thr_copy_c.partition_S(gC); // (CPY, CPY_M, CPY_N)
    Tensor tCsC_g2s = g2s_thr_copy_c.partition_D(sC);

    copy(g2s_tiled_copy_a, tAgA_g2s, tAsA_g2s);
    copy(g2s_tiled_copy_b, tBgB_g2s, tBsB_g2s);
    if constexpr (!IsGemm) {
        copy(g2s_tiled_copy_c, tCgC_g2s, tCsC_g2s);
    }

    #if defined(CP_ASYNC_ENABLED)
        cp_async_fence();
        cp_async_wait<0>();
    #endif
    __syncthreads();
    //--- Complete copy from GMEM to SMEM

    typename Spec::TiledCopyA_S2R s2r_tiled_copy_a;
    ThrCopy s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(tid);
    Tensor tAsA_s2r = s2r_thr_copy_a.partition_S(sA);
    Tensor tArA_s2r = s2r_thr_copy_a.partition_D(tCrA);

    typename Spec::TiledCopyB_S2R s2r_tiled_copy_b;
    ThrCopy s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(tid);
    Tensor tBsB_s2r = s2r_thr_copy_b.partition_S(sB);
    Tensor tBrB_s2r = s2r_thr_copy_b.partition_D(tCrB);


    typename Spec::TiledCopyC_S2R s2r_tiled_copy_c;
    ThrCopy s2r_thr_copy_c = s2r_tiled_copy_c.get_slice(tid);
    Tensor tCsC_s2r = s2r_thr_copy_c.partition_S(sC);
    Tensor tCrC_s2r = s2r_thr_copy_c.partition_D(tCrC);

    if constexpr (!IsGemm) {
        clear(tCrC);
    }else {
        copy(s2r_tiled_copy_c, tCsC_s2r, tCrC_s2r);
    }

    #if 1
        copy(s2r_tiled_copy_a, tAsA_s2r, tArA_s2r);
        copy(s2r_tiled_copy_b, tBsB_s2r, tBrB_s2r);

        gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);
    #else
        constexpr int kMmaValExpandM = Spec::kMmaValExpandM;
        constexpr int kMmaValExpandN = Spec::kMmaValExpandN;
        constexpr int kMmaValExpandK = Spec::kMmaValExpandK;

        constexpr int kMmaTileM = Spec::kMmaTileM;
        constexpr int kMmaTileN = Spec::kMmaTileN;
        constexpr int kMmaTileK = Spec::kMmaTileK;

        constexpr int NTilesM = kBlockM / kMmaTileM; // 4
        constexpr int NTilesN = kBlockN / kMmaTileN; // 4
        constexpr int NTilesK = kBlockK / kMmaTileK; // 2

    #pragma unroll
        for(int m_tile = 0; m_tile < NTilesM; ++m_tile) {
    #pragma unroll
            for(int n_tile = 0; n_tile < NTilesN; ++n_tile) {
    #pragma unroll
                for(int k_tile = 0; k_tile < NTilesK; ++k_tile) {
    #pragma unroll
                copy(s2r_tiled_copy_a, tAsA_s2r(_, m_tile, k_tile), tArA_s2r(_, m_tile, k_tile));
                copy(s2r_tiled_copy_b, tBsB_s2r(_, n_tile, k_tile), tBrB_s2r(_, n_tile, k_tile));
    #pragma unroll
                    for(int im = m_tile * kMmaValExpandM; im < (m_tile + 1) * kMmaValExpandM; ++im) {
    #pragma unroll
                        for(int in = n_tile * kMmaValExpandN; in < (n_tile + 1) * kMmaValExpandN; ++in) {
    #pragma unroll 
                            for(int ik = k_tile * kMmaValExpandK; ik < (k_tile + 1) * kMmaValExpandK; ++ik) {
                                gemm(tiled_mma, tCrC(_, im, in), tCrA(_, im, ik), tCrB(_, in, ik), tCrC(_, im, in));
                            }
                        }
                    }
                }
            }
        }

    #endif
    __syncthreads();

    if constexpr (!IsCvtPrecision) {
        typename Spec::TiledCopyC_R2S r2s_tiled_copy_c;
        ThrCopy r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(tid);
        Tensor tCrC_r2s = r2s_thr_copy_c.retile_S(tCrC);
        Tensor tCsC_r2s = r2s_thr_copy_c.partition_D(sC);
        copy(r2s_tiled_copy_c, tCrC_r2s, tCsC_r2s);

        __syncthreads();

        typename Spec::TiledCopyC_S2G s2g_tiled_copy_c;
        Tensor s2g_thr_copy_c = s2g_tiled_copy_c.get_slice(tid);
        Tensor tCsC_s2g = s2g_thr_copy_c.partition_S(sC);
        Tensor tCgC_s2g = s2g_thr_copy_c.partition_D(gC);
        copy(s2g_tiled_copy_c, tCsC_s2g, tCgC_s2g);
    }else {

        auto t = make_tensor_like<OutType>(tCrC);
        copy(tCrC, t); // Convert precision

        typename Spec::TiledCopyO_R2S r2s_tiled_copy_o;
        ThrCopy r2s_thr_copy_o = r2s_tiled_copy_o.get_slice(tid);
        Tensor tOrC_r2s = r2s_thr_copy_o.retile_S(t);     // (CPY, CPY_M, CPY_N)
        Tensor tOsO_r2s = r2s_thr_copy_o.partition_D(s0); // (CPY, CPY_M, CPY_N)
        copy(r2s_tiled_copy_o, tOrC_r2s, tOsO_r2s);

        __syncthreads();

        typename Spec::TiledCopyO_S2G s2g_tiled_copy_o;
        ThrCopy s2g_thr_copy_o = s2g_tiled_copy_o.get_slice(tid);
        Tensor tOsO_s2g = s2g_thr_copy_o.partition_S(s0); // (CPY, CPY_M, CPY_N)
        Tensor tOgO_s2g = s2g_thr_copy_o.partition_D(g0); // (CPY, CPY_M, CPY_N)
        copy(s2g_tiled_copy_o, tOsO_s2g, tOgO_s2g);
    }
}