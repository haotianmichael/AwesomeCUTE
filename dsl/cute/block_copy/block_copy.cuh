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

