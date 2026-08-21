# CUTLASS

* [CUTLASS源码导读-霸王手枪腿](https://zhuanlan.zhihu.com/p/588953452)
* [CUTLASS笔记-杨远航](https://zhuanlan.zhihu.com/p/1937220431728845963)
* [深入分析CUTLASS系列-JoeNomad](https://zhuanlan.zhihu.com/p/677616101)
* [NV-CUTLASS](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)
* [reed-cute](https://zhuanlan.zhihu.com/p/661182311)
* [CUTE-Declk](https://declk.github.io/blog/CUDA/index.html)

# Compile-Debug
> rm -rf build
> cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCUTLASS_EXAMPLES=ON \
  -DCUTLASS_NVCC_ARCHS=70 \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_UNITY_BUILD=ON \
  -DCMAKE_CUDA_FLAGS_DEBUG="-g -G -O0 -lineinfo -maxrregcount=128 -Xptxas -v -Xcompiler -fno-inline -Xcompiler -fno-omit-frame-pointer -Xcompiler -ggdb" \
  -DCMAKE_CXX_FLAGS_DEBUG="-O0 -ggdb -fno-inline -fno-omit-frame-pointer"