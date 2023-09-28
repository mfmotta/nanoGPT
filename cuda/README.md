# Notes 

</br>

# Relevant concepts:

</br>

## Constant Memory

``__constant__`` qualifier  used to define variables in the constant memory, advantageous when they are frequently used.

Benefits:

- Faster Access: Constant memory allows for faster memory access compared to global memory due to its optimized hardware architecture.

- Cache Utilization: Data stored in constant memory is cached, which further reduces memory access latency.

- Broadcasting: When a thread in a warp accesses a constant memory location, that value is broadcasted to all threads in the warp, which can lead to efficient data sharing.

Caveats:

- Limited size, usually ranging from 64KB to 128KB, depending on the GPU architecture.
 
- Read-only:  variables declared as __constant__ can only be read from kernels, not modified.


</br>

# Notes on CUTLASS 3.0

</br>

# [CuTe MMA Atoms](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/0t_mma_atom.md)

CuTe exposes each MMA to generic CUDA C++ code as a pair of structs: an "Operation" struct, and an MMA_Traits struct templated on the Operation struct type.

An "Operation" struct exposes the PTX instruction for that specific operation. It defines the arguments and interface it expects.
MMA_Traits struct defines meta-information about the Operation, such as the compute types, the logical shape of the operation, and the Layouts of threads and values within the operation. The MMA_Traits struct takes the Operation as a template parameter. 

CuTe specializes MMA_Traits for each Operation type that it supports.

Together, these two types comprise an "Atom" that decouples the complexity of thread and data layouts from the call site of of the PTX instruction. The Atom's Traits struct exposes information that is relevant to a single MMA operation, no matter the granularity at which it operates.

CuTe MMA atoms expose the semantics of a single MMA operation. This is true regardless of the hardware level at which the MMA operates. CuTe supports MMA atoms that operate at a variety of hardware levels, including

- a single thread (e.g., fused multiply-add (FMA) instruction) = multiply-accumulate with a single rounding (https://en.wikipedia.org/wiki/Multiply-accumulate_operation)

- a quadpair (Volta);

- a single warp (Ampere); and

- a warpgroup (Hopper).

An **Operation** struct contains:

- Type aliases, e.g. ``using DRegisters = float[8];`` D's type is F32
- FMA static member device function

Example:

```hpp
struct SM70_8x8x4_F16F16F16F16_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  // Register asm fma
  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM70_ENABLED)
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16"
                 "{%0, %1,  %2,  %3},"
                 "{%4, %5},"
                 "{%6, %7},"
                 "{%8, %9, %10, %11};\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        :  "r"(a0),  "r"(a1),
           "r"(b0),  "r"(b1),
           "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM70_8x8x4_F16F16F16F16_TN without CUTE_ARCH_MMA_SM70_ENABLED");
#endif
  }
};
```
where:
- asm: inline insert an [Assembler Statement (PTX)](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#assembler-asm-statements) at this point
- volatile: signal the compiler to not do any optimizations -- read and write registers as-is.

</br>

**Traits** contain:

- public type aliases, e.g. ``ElementDVal``: Compute type of the D matrix

Example:

```hpp
template <>
struct MMA_Traits<SM70_8x8x4_F32F16F16F32_NT>
{
  using ElementDVal = float;
  using ElementAVal = half_t;
  using ElementBVal = half_t;
  using ElementCVal = float;

  using Shape_MNK = Shape<_8,_8,_4>;
  using ThrID   = SM70_QuadPair; // alias for Layout<Shape <_4, _2>, Stride<_1,_16>>
  using ALayout = SM70_8x4_Col;
  using BLayout = SM70_8x4_Col;
  using CLayout = SM70_8x8_32b;
};
```
</br>

## MMA Atom in the Volta architecture

(NT = not transposed = M-major )

``Shape <_8,_8,_4> : (M, N, K) = (8, 8, 4) `` = dimensions of the MMA operation

Matrix product between a block A_block (8 rows, 4 cols) and a block B (4 rows, 8 cols)

The 32 threads in a warp are logically indexed by [0 ... 31]. The 0th quadpair is the upper-left block of threads with indices ``[0,1,2,3]U[16,17,18,19]``

32 threads within warp collectively hold A, B, C, and D operands (page 13 of https://developer.nvidia.com/gtc/2020/video/s21745)
  
</br>
---

# ATTENTION:

Strides:
strides tell you the mapping from a multidimensional index into a one-dimensional offset. Here, we're describing the shapes and strides of the "global" matrices.

CuTe interprets A as M x K, B as N x K, and C as M x N. Instead of row-major or column-major (or Transposed and Not-Transposed like above), we like to be more specific with M-major, N-major, or K-major. The reduction mode is outermost, i.e. K

CuTe's considers arrays as logically column major.

The vector mode (innermost mode of matrix) is the mode that we want in the innermost loop (in the nesting of loops that implements GEMM) (outer loop runs over k! --more efficient )

The vector mode (innermost mode of matrix) contains all of the information needed to execute the smallest possible computation or communication operations on hardware, that is, what CuTe calls the "atoms." ? i.e. a fragment of A and a fragment of B

Innermost mode of A = M
Innermost mode of B = N


ldA = the "leading dimension of A," a run-time value

Ex:
// Define strides (mixed)
auto dA = make_stride(Int<1>{}, ldA); // (dM,dK) 
the offset of A(index_m, index_k) is index_m * 1 + index_k * ldA

A matrix is "M-major" if it is stride 1 in the M-mode, "N-major" if it is stride 1 in the N-mode, or "K-major" if it is stride 1 in the K-mode
The stride between A(m, k) and A(m+1, k) is Int<1>, a compile-time value 1. The stride between A(m, k) and A(m, k+1), however, is ldA

The "leading dimension" of a matrix refers to the stride between consecutive columns of a column-major matrix (where the stride between consecutive rows is 1), or the stride between consecutive rows of a row-major matrix.

tAgA(_,_,k) means "create a Tensor that views (i, j, k) for all valid i, all valid j, and a specific value of k." 
CuTe uses the same notation for slices as for tensor indexing. The implementation can distinguish the two cases by checking whether any of the arguments is an underscore.

!If you have looked at other GEMM examples in CuTe, you might be wondering about hardware matrix-matrix multiply instructions. Those instructions tend to require certain values for shapes and strides, that may be a function of the matrix's element type. CuTe knows about these instructions and their required shapes and strides. We will go into more detail about that elsewhere.

// Shared memory buffer
__shared__ TA smemA[cosize_v<ABlockLayout>]; //MM: cosize_v -> the length of the (flattened) array in terms of number of elements, not in terms of number of bytes

CuTe uses a Layout to describe the assignment of threads to work items.

Use of static layouts has two advantages. First, it makes it easier to prove correctness of the algorithm. If the code compiles, it's likely correct. (On the other hand, new CuTe users may find themselves doing more debugging at compile time than they have before.) Second, it makes it easier and faster for CuTe to dispatch to the correct optimized implementations (called "atoms" -- see below) for copying blocks and performing matrix multiplies.

// Define the thread layouts (static)
auto tA = make_layout(make_shape(Int<32>{}, Int< 8>{}));
auto tB = make_layout(make_shape(Int<32>{}, Int< 8>{}));
auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));

The "size" of CThreadLayout is the total number of threads, 16 * 16 = 256.(MM in the block). The block gemm function (in the loop over blocks) parallelizes over elements of the C block. The kernel launch uses the size of CThreadLayout as the block dimension.

dim3 dimBlock(size(tC));
dim3 dimGrid(ceil_div(size(M), size(bM)),
             ceil_div(size(N), size(bN)));

a thread layout maps from a "logical" coordinate space (possibly multidimensional tuples of indices) to (one-dimensional) integer indices. In this case, CThreadLayout maps from pairs of indices in the Cartesian product space {0, 1, 2, ..., 15} x {0, 1, 2, ..., 15}, to one-dimensional indices 0, 1, 2, ..., 255. The latter, the output of CThreadLayout, is the actual thread index threadIdx.x in this case.


