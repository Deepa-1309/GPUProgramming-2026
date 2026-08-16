%%writefile barrier.cu
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// Custom barrier using atomic operation and __syncthreads()
__device__ void customBarrier(int* arrived, int* released)
{
    // Each thread atomically records its arrival
    atomicAdd(arrived, 1);

    // Synchronize all threads in the block
    __syncthreads();

    // Thread 0 releases the barrier after all threads arrive
    if (threadIdx.x == 0)
    {
        if (*arrived == blockDim.x)
        {
            *released = 1;
        }
    }

    // Make sure all threads see the release
    __syncthreads();
}

__global__ void barrierKernel()
{
    __shared__ int arrived;
    __shared__ int released;

    // Initialize shared variables
    if (threadIdx.x == 0)
    {
        arrived = 0;
        released = 0;
    }

    // Make sure initialization is completed
    __syncthreads();

    int tid = threadIdx.x;

    printf("Thread %d reached the barrier\n", tid);

    // Custom barrier
    customBarrier(&arrived, &released);

    // Code after the barrier
    if (released)
    {
        printf("Thread %d passed the barrier\n", tid);
    }
}

int main()
{
    int threadsPerBlock = 8;

    // Launch one block containing 8 threads
    barrierKernel<<<1, threadsPerBlock>>>();

    // Check for kernel launch errors
    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        cout << "Kernel launch error: "
             << cudaGetErrorString(error)
             << endl;

        return 1;
    }

    // Wait for GPU to finish
    error = cudaDeviceSynchronize();

    if (error != cudaSuccess)
    {
        cout << "CUDA error: "
             << cudaGetErrorString(error)
             << endl;

        return 1;
    }

    return 0;
}