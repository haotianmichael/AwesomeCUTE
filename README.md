# AwesomeCUTE
`CUTLASS`是NVIDIA开源的一套底层高性能计算库，广泛用于加速矩阵乘法、卷积等深度学习与线性代数运算。在3.0版本中，`CUTE`正式作为其核心抽象层引入，显著降低了算子开发难度，也为编译器DSL设计提供了SOTA级别的参考。2026年初，`CUTE`作者之一Cris Cecka 在arXiv上发布了`CUTE`白皮书[CuTe Layout Representation and Algebra
](https://arxiv.org/abs/2603.02298)，笔者将其翻译成中文作为参考。为保证精确性，文中保留了大量必要的专业术语，翻译过程加上了笔者自己的理解，如发现有问题的还望指出。pdf和latex版本见[Github](https://github.com/haotianmichael/AwesomeCUTE).

* [CUTE Whitepaper中文版](latex/)
* [Sgemm优化代码](csrc/sgemm/)
* [Hgemm优化代码](csrc/hgemm/)
* [CuTe优化代码](dsl/cute/)
