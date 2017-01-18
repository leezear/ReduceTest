// CUDA Runtime
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
// Utilities and system includes
#include <helper_cuda.h>
#include <helper_functions.h>

#include <stdio.h>
#include <iostream>
#include <random>
#include <math.h>

typedef unsigned int u32;
using namespace std;

#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

unsigned int nextPow2(unsigned int x)
{
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}

void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
	threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
	blocks = (n + (threads * 2 - 1)) / (threads * 2);
	blocks = MIN(maxBlocks, blocks);
}


// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
	__device__ inline operator T *()
	{
		extern __shared__ int __smem[];
		return (T *)__smem;
	}

	__device__ inline operator const T *() const
	{
		extern __shared__ int __smem[];
		return (T *)__smem;
	}
};

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6(T *g_idata, T *g_odata, unsigned int n)
{
	T *sdata = SharedMemory<T>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize * 2 + threadIdx.x;
	unsigned int gridSize = blockSize * 2 * gridDim.x;

	T mySum = 0;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		mySum += g_idata[i];

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			mySum += g_idata[i + blockSize];

		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = mySum;
	__syncthreads();


	// do reduction in shared mem
	if ((blockSize >= 512) && (tid < 256))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 256];
	}

	__syncthreads();

	if ((blockSize >= 256) && (tid < 128))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 128];
	}

	__syncthreads();

	if ((blockSize >= 128) && (tid < 64))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 64];
	}

	__syncthreads();

	if (tid < 32)
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >= 64) mySum += sdata[tid + 32];
		// Reduce final warp using shuffle
		for (int offset = warpSize / 2; offset > 0; offset /= 2)
		{
			mySum += __shfl_down(mySum, offset);
		}
	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = mySum;
}

__host__ __device__ __inline__ bool isPow2(unsigned int x)
{
	return ((x&(x - 1)) == 0);
}

__global__ void print(u32* in, const u32 n)
{
	const int gtid = blockDim.x*blockIdx.x + threadIdx.x;

	if (gtid == 1)
		printf("%d = %d\n", gtid, in[gtid]);
}

////////////////////////////////////////////////////////////////////////////////
//! Compute sum reduction on CPU
//! We use Kahan summation for an accurate sum of large arrays.
//! http://en.wikipedia.org/wiki/Kahan_summation_algorithm
//!
//! @param data       pointer to input data
//! @param size       number of input data elements
////////////////////////////////////////////////////////////////////////////////
template<class T>
T reduceCPU(T *data, int size)
{
	T sum = data[0];
	T c = (T)0.0;

	for (int i = 1; i < size; i++)
	{
		T y = data[i] - c;
		T t = sum + y;
		c = (t - sum) - y;
		sum = t;
	}

	return sum;
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T>
void
reduce(int size, int threads, int blocks, T *d_idata, T *d_odata)
{
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

	//如果是2的幂数
	if (isPow2(size))
	{
		switch (threads)
		{
		case 512:
			reduce6<T, 512, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
			break;

		case 256:
			reduce6<T, 256, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
			break;

		case 128:
			reduce6<T, 128, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
			break;

		case 64:
			reduce6<T, 64, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
			break;

		case 32:
			reduce6<T, 32, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
			break;

		case 16:
			reduce6<T, 16, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
			break;

		case  8:
			reduce6<T, 8, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
			break;

		case  4:
			reduce6<T, 4, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
			break;

		case  2:
			reduce6<T, 2, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
			break;

		case  1:
			reduce6<T, 1, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
			break;
		}
	}
	else
	{
		switch (threads)
		{
		case 512:
			reduce6<T, 512, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
			break;

		case 256:
			reduce6<T, 256, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
			break;

		case 128:
			reduce6<T, 128, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
			break;

		case 64:
			reduce6<T, 64, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
			break;

		case 32:
			reduce6<T, 32, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
			break;

		case 16:
			reduce6<T, 16, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
			break;

		case  8:
			reduce6<T, 8, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
			break;

		case  4:
			reduce6<T, 4, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
			break;

		case  2:
			reduce6<T, 2, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
			break;

		case  1:
			reduce6<T, 1, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
			break;
		}
	}
}

int main()
{
	//	是否将GPU数据写回
	bool needReadBack = true;
	bool cpuFinalReduction = false;

	typedef u32 T;
	// 归约数组长度
	const int size = 512;
	// 最大blocks数
	const int maxBlocks = 64;
	// 每block中最大线程数
	const int maxThreads = 256;
	// 输入数组byte长度
	u32 bytes = size * sizeof(u32);

	u32 sum = 0;
	int numBlocks = 0;
	int numThreads = 0;
	getNumBlocksAndThreads(size, maxBlocks, maxThreads, numBlocks, numThreads);

	cout << "size = " << size << endl;
	cout << "numBlocks = " << numBlocks << endl;
	cout << "numThreads = " << numThreads << endl;

	// 生成host数组，并赋初值
	T *h_idata = (T *)malloc(bytes);
	// allocate mem for the result on host side
	T *h_odata = (T *)malloc(numBlocks*sizeof(T));

	//生成随机数
	default_random_engine generator;
	uniform_int_distribution<u32> dis(0, 1000);
	for (u32 i = 0; i < size; ++i)
		h_idata[i] = dis(generator);

	// allocate device memory and data
	T *d_idata = NULL;
	T *d_odata = NULL;

	//	在设备内存上申请空间
	checkCudaErrors(cudaMalloc((void **)&d_idata, bytes));
	checkCudaErrors(cudaMalloc((void **)&d_odata, numBlocks*sizeof(T)));

	//	拷贝到设备内存
	checkCudaErrors(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_odata, h_idata, numBlocks*sizeof(T), cudaMemcpyHostToDevice));

	//	sample示例中 ----benchmarkReduce()----- 方法
	T gpu_result = 0;

	//	创建计时器
	StopWatchInterface *timer = 0;
	sdkCreateTimer(&timer);
	//	开启计时器
	cudaDeviceSynchronize();
	sdkStartTimer(&timer);

	// 一阶段归约内核
	reduce<T>(size, numThreads, numBlocks, d_idata, d_odata);

	// Clear d_idata for later use as temporary buffer.
	cudaMemset(d_idata, 0, size*sizeof(T));

	// 以CPU做最后的归约
	if (cpuFinalReduction)
	{
		// sum partial sums from each block on CPU
		// copy result from device to host
		checkCudaErrors(cudaMemcpy(h_odata, d_odata, numBlocks*sizeof(T), cudaMemcpyDeviceToHost));

		for (int i = 0; i < numBlocks; i++)
			gpu_result += h_odata[i];

		needReadBack = false;
	}
	else
	{
		// sum partial block sums on GPU
		int s = numBlocks;
		//	当block数降到多少时开启CPU归约
		int cpuFinalThreshold = 1;

		while (s > cpuFinalThreshold)
		{
			int threads = 0, blocks = 0;
			//	重新计算thread数和block数
			getNumBlocksAndThreads(s, maxBlocks, maxThreads, blocks, threads);
			cudaMemcpy(d_idata, d_odata, s*sizeof(T), cudaMemcpyDeviceToDevice);
			reduce<T>(s, threads, blocks, d_idata, d_odata);

			s = (s + (threads * 2 - 1)) / (threads * 2);
		}

		if (s > 1)
		{
			// copy result from device to host
			checkCudaErrors(cudaMemcpy(h_odata, d_odata, s * sizeof(T), cudaMemcpyDeviceToHost));

			for (int i = 0; i < s; i++)
			{
				gpu_result += h_odata[i];
			}

			needReadBack = false;
		}
	}

	cudaDeviceSynchronize();
	sdkStopTimer(&timer);

	if (needReadBack)
	{
		// copy final sum from device to host
		checkCudaErrors(cudaMemcpy(&gpu_result, d_odata, sizeof(T), cudaMemcpyDeviceToHost));
	}

	double reduceTime = sdkGetAverageTimerValue(&timer) * 1e-3;

	// compute reference solution
	T cpu_result = reduceCPU<T>(h_idata, size);

	printf("\nGPU result = %d\n", (int)gpu_result);
	printf("CPU result = %d\n\n", (int)cpu_result);
	cout << "reduceTime = " << reduceTime << endl;

	// cleanup
	sdkDeleteTimer(&timer);
	free(h_idata);
	free(h_odata);

	checkCudaErrors(cudaFree(d_idata));
	checkCudaErrors(cudaFree(d_odata));

	if (gpu_result == cpu_result)
		cout << "pass!" << endl;

	return 0;
}