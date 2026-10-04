#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;
using namespace wmma;

// Tensor Core tile dimensions
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;


// ---------------------------------------------------------
// Initialize matrices A and B
// ---------------------------------------------------------
__global__ void init(half *A, half *B)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < 64 * 64)
    {
        A[tid] = __float2half((float)(tid % 8));
        B[tid] = __float2half((float)(tid % 8));
    }
}


// ---------------------------------------------------------
// Tensor Core Matrix Multiplication
//
// A : M x K
// B : K x N
// C : M x N
//
// Each warp calculates one 16 x 16 tile of C.
// ---------------------------------------------------------
__global__ void tensorCoreGemmKernel(
    half *A,
    half *B,
    float *C,
    int M,
    int N,
    int K)
{
    // One block = one warp = one output tile

    int warpM = blockIdx.x;
    int warpN = blockIdx.y;

    // Starting row and column of this 16x16 tile
    int mTile = warpM * WMMA_M;
    int nTile = warpN * WMMA_N;

    // Make sure tile is inside matrix
    if (mTile >= M || nTile >= N)
        return;


    // -----------------------------------------------------
    // Declare Tensor Core fragments
    // -----------------------------------------------------

    fragment<matrix_a,
             WMMA_M,
             WMMA_N,
             WMMA_K,
             half,
             row_major> aFrag;

    fragment<matrix_b,
             WMMA_M,
             WMMA_N,
             WMMA_K,
             half,
             row_major> bFrag;

    fragment<accumulator,
             WMMA_M,
             WMMA_N,
             WMMA_K,
             float> cFrag;


    // Start C tile with zero
    fill_fragment(cFrag, 0.0f);


    // -----------------------------------------------------
    // K = 64
    //
    // Tensor Core can multiply 16x16x16.
    //
    // Therefore:
    //
    // 64 / 16 = 4
    //
    // Tensor Core operations are required for each C tile.
    // -----------------------------------------------------

    for (int i = 0; i < K / WMMA_K; i++)
    {
        // Starting position of A tile

        int aind = mTile * K + i * WMMA_K;

        // Starting position of B tile

        int bind = i * WMMA_K * N + nTile;


        // Load 16x16 tile from A
        load_matrix_sync(
            aFrag,
            A + aind,
            K
        );


        // Load 16x16 tile from B
        load_matrix_sync(
            bFrag,
            B + bind,
            N
        );


        // Tensor Core multiplication
        //
        // cFrag = aFrag * bFrag + cFrag

        mma_sync(
            cFrag,
            aFrag,
            bFrag,
            cFrag
        );
    }


    // -----------------------------------------------------
    // Store the resulting 16x16 tile into C
    // -----------------------------------------------------

    int cind = mTile * N + nTile;

    store_matrix_sync(
        C + cind,
        cFrag,
        N,
        mem_row_major
    );
}


// ---------------------------------------------------------
// Main
// ---------------------------------------------------------
int main()
{
    // Matrix dimensions
    int M = 64;
    int N = 64;
    int K = 64;


    half *devA;
    half *devB;

    float *devC;
    float *hostC;


    // CUDA timing
    cudaEvent_t start;
    cudaEvent_t stop;

    float elapsedTime;


    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // -----------------------------------------------------
    // Allocate GPU memory
    // -----------------------------------------------------

    cudaMalloc(
        &devA,
        M * K * sizeof(half)
    );

    cudaMalloc(
        &devB,
        K * N * sizeof(half)
    );

    cudaMalloc(
        &devC,
        M * N * sizeof(float)
    );


    // -----------------------------------------------------
    // Allocate CPU memory
    // -----------------------------------------------------

    hostC = (float *)malloc(
        M * N * sizeof(float)
    );


    // -----------------------------------------------------
    // Initialize A and B
    // -----------------------------------------------------

    init<<<4, 1024>>>(devA, devB);

    cudaDeviceSynchronize();


    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf("Initialization error: %s\n",
               cudaGetErrorString(err));
        return 1;
    }


    // -----------------------------------------------------
    // Launch Tensor Core kernel
    //
    // 64 / 16 = 4 tiles in each direction
    //
    // Therefore grid = 4 x 4 = 16 blocks
    //
    // Each block contains one warp = 32 threads
    // -----------------------------------------------------

    dim3 grid(4, 4);
    dim3 block(32);


    cudaEventRecord(start, 0);


    tensorCoreGemmKernel<<<grid, block>>>(
        devA,
        devB,
        devC,
        M,
        N,
        K
    );


    cudaEventRecord(stop, 0);


    // Wait for kernel
    cudaEventSynchronize(stop);


    cudaDeviceSynchronize();


    // -----------------------------------------------------
    // Check kernel error
    // -----------------------------------------------------

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf("Kernel error: %s\n",
               cudaGetErrorString(err));
        return 1;
    }


    // -----------------------------------------------------
    // Calculate execution time
    // -----------------------------------------------------

    cudaEventElapsedTime(
        &elapsedTime,
        start,
        stop
    );


    printf(
        "Tensor Core kernel execution time: %f ms\n",
        elapsedTime
    );


    // -----------------------------------------------------
    // Copy result from GPU to CPU
    // -----------------------------------------------------

    cudaMemcpy(
        hostC,
        devC,
        M * N * sizeof(float),
        cudaMemcpyDeviceToHost
    );


    // -----------------------------------------------------
    // Print a small portion of result
    // -----------------------------------------------------

    printf("\nFirst 16x16 output tile:\n\n");

    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            printf(
                "%8.1f ",
                hostC[i * N + j]
            );
        }

        printf("\n");
    }


    // -----------------------------------------------------
    // Verify result using CPU
    // -----------------------------------------------------

    int errors = 0;


    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float expected = 0.0f;


            for (int k = 0; k < K; k++)
            {
                float a =
                    (float)((i * K + k) % 8);

                float b =
                    (float)((k * N + j) % 8);

                expected += a * b;
            }


            if (fabs(hostC[i * N + j] - expected) > 0.01f)
            {
                errors++;
            }
        }
    }


    // -----------------------------------------------------
    // Verification result
    // -----------------------------------------------------

    if (errors == 0)
    {
        printf(
            "\nVerification PASSED!\n"
        );
    }
    else
    {
        printf(
            "\nVerification FAILED!\n"
        );

        printf(
            "Number of errors = %d\n",
            errors
        );
    }


    // -----------------------------------------------------
    // Free memory
    // -----------------------------------------------------

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    free(hostC);


    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    return 0;
}