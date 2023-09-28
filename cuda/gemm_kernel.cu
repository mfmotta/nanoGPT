/*
  This kernel is based on Cutlass-1.3 sgemm: 
  https://github.com/NVIDIA/cutlass/blob/master/examples/00_basic_gemm/basic_gemm.cu

  The CUTLASS Gemm template is instantiated in the function CutlassSgemmNN. This kernel:
  
  > computes the general matrix product (GEMM) using single-precision floating-point arithmetic
  > assumes all matrices have column-major layout.

  Threadblock tile size is chosen to be 128x128x8 

  To view the full gemm device API interface, see `cutlass/gemm/device/gemm.h` 
  (https://github.com/NVIDIA/cutlass/blob/master/include/cutlass/gemm/device/gemm.h)
*/

#include <iostream>
#include <sstream>
#include <vector>
#include "helper.h"
#include "cutlass/gemm/device/gemm.h"

// This function defines a CUTLASS GEMM kernel instantiation, constructs its parameters object,
// and launches it on the CUDA device.

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSgemmNN(
    int M,
    int N,
    int K,
    float alpha,
    float const *A,
    int lda,
    float const *B,
    int ldb,
    float beta,
    float *C,
    int ldc) {

    //MM at compile time: maps data types and high-level structural parameters onto specific CUTLASS components
    using ColumnMajor = cutlass::layout::ColumnMajor;

    using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                    ColumnMajor,  // Layout of A matrix
                                                    float,        // Data-type of B matrix
                                                    ColumnMajor,  // Layout of B matrix
                                                    float,        // Data-type of C matrix
                                                    ColumnMajor>; // Layout of C matrix

    // Define a CUTLASS GEMM type
    CutlassGemm gemm_operator;

    // Construct the CUTLASS GEMM arguments object.
    //
    // gemm argument objects are constructible in host code and passed to kernels by value. 
    // These may include pointers, strides, scalars, and other arguments needed by Gemm and its components.
    //
    // Benefits of this pattern: (1.) structured, composable strategy for passing host-constructible
    // arguments to kernels and (2.) minimized initialization overhead on kernel entry.

    // MM At runtime, map logical arguments to GEMM problems to kernel parameters.
    CutlassGemm::Arguments args({M , N, K},  // Gemm Problem dimensions
                                {A, lda},    // Tensor-ref for source matrix A TODO:?with leading dimension lda=column (col major)
                                {B, ldb},    // Tensor-ref for source matrix B
                                {C, ldc},    // Tensor-ref for source matrix C
                                {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                {alpha, beta}); // Scalars used in the Epilogue

    // Launch the CUTLASS GEMM kernel on the device at runtime.
    cutlass::Status status = gemm_operator(args);

    // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    // Return success, if no errors were encountered.
    return cudaSuccess;
    }


// The source code after this point in the file is generic CUDA using the CUDA Runtime API
// and simple CUDA kernels to initialize matrices and compute the general matrix product.

/// Kernel to initialize a matrix with small integers.
__global__ void InitializeMatrix_kernel(
  float *matrix,
  int rows,
  int columns,
  int seed = 0) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    int offset = i + j * rows;

    // Generate arbitrary elements.
    int const k = 16807;
    int const m = 16;
    float value = float(((offset + seed) * k % m) - m / 2);

    matrix[offset] = value;
  }
}

// Function that uses kernel to initialize a matrix to arbitrary small integers.
cudaError_t InitializeMatrix(float *matrix, int rows, int columns, int seed = 0) {

  dim3 block(16, 16);
  dim3 grid(
    (rows + block.x - 1) / block.x,
    (columns + block.y - 1) / block.y
  );

  InitializeMatrix_kernel<<< grid, block >>>(matrix, rows, columns, seed);

  return cudaGetLastError();
}

// Will be used to create matrices A, B, C_cutlass, and C_reference (last two with same seed)
// Allocates device memory for a matrix then fill with arbitrary small integers.
cudaError_t AllocateMatrix(float **matrix, int rows, int columns, int seed = 0) {
  cudaError_t result;

  size_t sizeof_matrix = sizeof(float) * rows * columns;

  // Allocate device memory.
  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Clear the allocation.
  result = cudaMemset(*matrix, 0, sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to clear matrix device memory: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Initialize matrix elements to arbitrary small integers.
  result = InitializeMatrix(*matrix, rows, columns, seed);

  if (result != cudaSuccess) {
    std::cerr << "Failed to initialize matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  return result;
}



/// Reference GEMM computation.
__global__ void ReferenceGemm_kernel(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    float accumulator = 0;

    for (int k = 0; k < K; ++k) {
      accumulator += A[i + k * lda] * B[k + j * ldb];
    }

    C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
  }
}

/// Reference GEMM computation.
cudaError_t ReferenceGemm(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  dim3 block(16, 16);
  dim3 grid(
    (M + block.x - 1) / block.x,
    (N + block.y - 1) / block.y
  );

  ReferenceGemm_kernel<<< grid, block >>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

  return cudaGetLastError();
}


/// Allocate several matrices in GPU device memory and call a single-precision
/// CUTLASS GEMM kernel.
cudaError_t TestCutlassGemm(int M, int N, int K, float alpha, float beta) {
  cudaError_t result;

  //
  // Define several matrices to be used as operands to GEMM kernels.
  //

  // Compute leading dimensions for each matrix.
  int lda = M;
  int ldb = K;
  int ldc = M;

  // Compute size in bytes of the C matrix.
  size_t sizeof_C = sizeof(float) * ldc * N;

  // Define pointers to matrices in GPU device memory.
  float *A;
  float *B;
  float *C_cutlass;
  float *C_reference;

  //
  // Allocate matrices in GPU device memory with arbitrary seeds. 
  //and free memory if allocation was not successful
  //

  result = AllocateMatrix(&A, M, K, 0);

  if (result !=  cudaSuccess) {
    return result;
  }

  result = AllocateMatrix(&B, K, N, 17);

  if (result !=  cudaSuccess) {
    cudaFree(A);
    return result;
  }

  result = AllocateMatrix(&C_cutlass, M, N, 101);

  if (result != cudaSuccess) {
    cudaFree(A);
    cudaFree(B);
    return result;
  }

  result = AllocateMatrix(&C_reference, M, N, 101);

  if (result != cudaSuccess) {
    cudaFree(A);
    cudaFree(B);
    cudaFree(C_cutlass);
    return result;
  }

  result = cudaMemcpy(C_reference, C_cutlass, sizeof_C, cudaMemcpyDeviceToDevice);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy C_cutlass matrix to C_reference: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }






    
TODO: eventually replace allocation with reading some matrix defined elsewhere, maybe python ?