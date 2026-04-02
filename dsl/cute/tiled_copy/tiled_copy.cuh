#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>


namespace spec{

using namespace cute;

template <typename OutType_,
          typename ComputeTypeA_,
          typename ComputeTypeB_,
          typename ComputeTypeC_,
          int kTiledM_ = 32,
          int kTiledN_ = 32,
          int kTiledK_ = 16>
struct KernelSpec{

    using OutType = OutType_;
    using ComputeTypeA = ComputeTypeA_;
    using ComputeTypeB = ComputeTypeB_;
    using ComputeTypeC = ComputeTypeC_;

    static constexpr int kTiledM = kTiledM_;
    static constexpr int kTiledN = kTiledN_;
    static constexpr int kTiledK = kTiledK_;

    using MMA_op = SM80_16x8x8_F32BF16BF16F32_TN;
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

    using MMAThrLayout = decltype(make_layout(make_shape(Int<kMmaThrExpandM>{}, Int<kMmaThrExpandN>{}, Int<kMmaThrExpandK>{})));
    using MMATileLayout = Tile<Int<kMmaTileM>, Int<kMmaTileN>, Int<kMmaTileK>>;
    using TiledMMA = decltype(make_tiled_mma(MMA_op{}, MMAThrLayout{}, MMATileLayout{}));

    using Copy_op = AutoVectorizingCopy;

    using CopyA_atom = Copy_Atom<Copy_op, ComputeTypeA>;
    using CopyB_atom = Copy_Atom<Copy_op, ComputeTypeB>; 
    using CopyC_atom = Copy_Atom<Copy_op, ComputeTypeC>;
    using Copy0_atom = Copy_Atom<Copy_op, OutType>;
    
    using TileCopyA = decltype(make_tiled_copy_A(CopyA_atom{}, TiledMMA{}));
    using TileCopyB = decltype(make_tiled_copy_B(CopyB_atom{}, TiledMMA{}));
    using TileCopyC = decltype(make_tiled_copy_C(CopyC_atom{}, TiledMMA{}));
    using TileCopy0 = decltype(make_tiled_copy_C(Copy0_atom{}, TiledMMA{}));

    static constexpr int kThreadNum = size(TiledMMA{});
    static constexpr int kShmSize = 0;
};
}

template<typename Spec, bool IsGemm, bool IsCvtPrecision>
__global__ void tiled_copy(void *Cptr, const void *Aptr, const void *Bptr, int m, int n, int k, void *Outptr) {

    using namespace cute;

    using X = Underscore;
    using OutType = typename Spec::OutType;
    using ComputeTypeA = typename Spec::ComputeTypeA;
    using ComputeTypeB = typename Spec::ComputeTypeB;
    using ComputeTypeC = typename Spec::ComputeTypeC;
    using TiledMMA = typename Spec::TiledMMA;
    using TiledCopyA = typename Spec::TiledCopyA;
    using TiledCopyB = typename Spec::TiledCopyB;
    using TiledCopyC = typename Spec::TiledCopyC;
    using TiledCopy0 = typename Spec::TiledCopy0; 

    constexpr int kTileM = Spec::kTileM;
    constexpr int kTileN = Spec::kTileN;
    constexpr int kTileK = Spec::kTileK;

    int tid = threadIdx.x;

    Tensor mA = make_tensor(make_gmem_ptr((ComputeTypeA*)Aptr), make_shape(m, k), make_stride(k, Int<1>{}));  // (M, K)
    Tensor mB = make_tensor(make_gmem_ptr((ComputeTypeB*)Bptr), make_shape(n, k), make_stride(k, Int<1>{}));  // (N, K)
    Tensor mC = make_tensor(make_gmem_ptr((ComputeTypeC*)Cptr), make_shape(m, n), make_stride(n, Int<1>{})); // (M, N)
    Tensor m0 = make_tensor(make_gmem_ptr((OutType *)Outptr), make_shape(m, n), make_stride(n, Int<1>{})); // (M, N)

    auto tiler = make_tile(Int<kTileM>{}, Int<kTileN>{}, Int<kTileK>{});
    auto coord = make_coord(0, 0, 0);

    // Define the block global tensors (static)
    Tensor gA = local_tile(mA, tiler, coord, Step<_1, X, _1>{});  // (kTileM, kTileK)
    Tensor gB = local_tile(mB, tiler, coord, Step<X, _1, _1>{});  // (kTileN, kTileK)
    Tensor gC = local_tile(mC, tiler, coord, Step<_1, _1, X>{});  // (kTileM, kTileN)
    Tensor g0 = local_tile(m0, tiler, coord, Step<_1, _1, X>{});  // (kTileM, kTileN)

    TiledMMA tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_slice(tid);

    Tensor tCgA = thr_mma.partition_A(gA); // (MMA, MMA_M, MMA_K)
    Tensor tCgB = thr_mma.partition_B(gB);
    Tensor tCgC = thr_mma.partition_C(gC);

    Tensor tCrA = thr_mma.partition_fragment_A(gA); // (MMA, MMA_M, MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(gB);
    Tensor tCrC = thr_mma.partition_fragment_C(gC);


    /* A*/
    TiledCopyA g2r_tiled_copy_a;
    ThrCopy g2r_thr_cpy_a = g2r_tiled_copy_a.get_slice(tid);
    Tensor tAgA = g2r_thr_cpy_a.retile_S(tCgA);  // (CPY, CPY_M, CPY_K)
    // Tensor tAgA = g2r_thr_cpy_a.partition_S(gA);
    Tensor tArA = g2r_thr_cpy_a.retile_D(tCrA);

    /* B*/
    TiledCopyB g2r_tiled_copy_b;
    ThrCopy g2r_thr_cpy_b = g2r_tiled_copy_b.get_slice(tid);
    Tensor tBgB = g2r_thr_cpy_b.retile_S(tCgB);  // (CPY, CPY_N, CPY_K)
    // Tensor tBgB = g2r_thr_cpy_b.partition_S(gB);
    Tensor tBrB = g2r_thr_cpy_b.retile_D(tCrB);

   
    copy(g2r_tiled_copy_a, tAgA, tArA);
    copy(g2r_tiled_copy_b, tBgB, tBrB);

    if constexpr (IsGemm) {
        clear(tCrC);        
    }else {
        TiledCopyC g2r_tiled_copy_c;
        ThrCopy g2r_thr_cpy_c = g2r_tiled_copy_c.get_slice(tid);
        Tensor tCgC_g2r = g2r_thr_cpy_c.retile_S(tCgC);
        Tensor tCrC_g2r = g2r_thr_cpy_c.retile_D(tCrC); 
        copy(g2r_tiled_copy_c, tCgC_g2r, tCrC_g2r);
    }

    gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);

    TiledCopy0 r2g_tiled_copy_o;
    if constexpr (!IsCvtPrecision) {
        ThrCopy r2g_thr_cpy_o = r2g_tiled_copy_o.get_slice(tid);
        Tensor tCrC_r2g = r2g_thr_cpy_o.retile_S(tCrC);
        Tensor tCgC_r2g = r2g_thr_cpy_o.retile_D(tCgC);
        copy(r2g_tiled_copy_o, tCrC_r2g, tCgC_r2g);
    } else {
        Tensor tCg0 = thr_mma.partition_C(g0);
        auto t = make_tensor_like<OutType>(tCrC);
        copy(tCrC, t);

        ThrCopy r2g_thr_cpy_o = r2g_tiled_copy_o.get_slice(tid);
        Tensor tCrC_r2g = r2g_thr_cpy_o.retile_S(t);
        Tensor tCg0_r2g = r2g_thr_cpy_o.retile_D(tCg0);
        copy(r2g_tiled_copy_o, tCrC_r2g, tCg0_r2g);
    }

}

using namespace cute;
template<int kTileM, int kTileN>
__global__ void g2s_tiled_copy(const float *input) {

    int tid = threadIdx.x;
    __shared__ float shm[kTileM * kTileN];

    Tensor g_input_tile = make_tensor(cute::make_gmem_ptr((float*)input), make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));
    Tensor s_tensor = make_tensor(cute::make_gmem_ptr((float*)shm), make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));

    using g2s_copy_op = UniversalCopy<cute::uint64_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, float>;

    Layout thr_layout = make_layout(make_shape(Int<2>{}, Int<2>{}), make_stride(Int<2>{}, Int<1>{}));
    Layout val_layout = make_layout(make_shape(Int<1>{}, Int<2>{}));

    auto tiled_copy_g2s = make_tiled_copy(g2s_copy_atom{}, thr_layout, val_layout);

    /*
    我们构建出来的 TV-layout 其实只是一次 atom copy 的 TV-layout。为了完成整个 tile 的 partition，在 CuTe 的实现中，
    首先是将一个 tile 平均拆分成（CPY_M, CPY_N）个 atom 大小，然后构建出对应的 atom-TV-layout，进而拿到整个 tile 的 TV-layout。
    */
    auto thr_copy_g2s = tiled_copy_g2s.get_slice(tid);
    auto tgA_g2s = thr_copy_g2s.partition_S(g_input_tile); // (CPY_ATOM, CPY_M, CPY_N)
    auto tsA_g2s = thr_copy_g2s.partition_D(s_tensor);

    copy(tiled_copy_g2s, tgA_g2s, tsA_g2s);

    __syncthreads();
}