/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

// Shuffle intrinsics CUDA Sample
// This sample demonstrates the use of the shuffle intrinsic
// First, a simple example of a prefix sum using the shuffle to
// perform a scan operation is provided.
// Secondly, a more involved example of computing an integral image
// using the shuffle intrinsic is provided, where the shuffle
// scan operation and shuffle xor operations are used

#include <stdio.h>
// CUDA Runtime
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
// Utilities and system includes
#include <helper_cuda.h>
#include <helper_functions.h>
//#include "shfl_integral_image.cuh"

// Scan using shfl - takes log2(n) steps
// This function demonstrates basic use of the shuffle intrinsic, __shfl_up,
// to perform a scan operation across a block.
// First, it performs a scan (prefix sum in this case) inside a warp
// Then to continue the scan operation across the block,
// each warp's sum is placed into shared memory.  A single warp
// then performs a shuffle scan on that shared memory.  The results
// are then uniformly added to each warp's threads.
// This pyramid type approach is continued by placing each block's
// final sum in global memory and prefix summing that via another kernel call, then
// uniformly adding across the input data via the uniform_add<<<>>> kernel.

__global__ void shfl_scan_test(int *data, int width, int *partial_sums = NULL)
{
	//	申请共享内存，block之间的共享内存不可相互访问
	extern __shared__ int sums[];
	//	获取全局id
	int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
	//	线程束内线程ID
	int lane_id = id % warpSize;
	//	block内的局部线程束ID
	int warp_id = threadIdx.x / warpSize;

	// Below is the basic structure of using a shfl instruction
	// for a scan.
	// Record "value" as a variable - we accumulate it along the way
	int value = data[id];

	// Now accumulate in log steps up the chain
	// compute sums, with another thread's value who is
	// distance delta away (i).  Note
	// those threads where the thread 'i' away would have
	// been out of bounds of the warp are unaffected.  This
	// creates the scan sum.
#pragma unroll

	//	步长越来越长
	for (int i = 1; i <= width; i *= 2)
	{
		//	__shfl_up(变量, [1, 2, 4, ...,32], 32)
		int n = __shfl_up(value, i, width);

		//	没对齐的部分不用相加
		if (lane_id >= i) value += n;
	}

	// value now holds the scan value for the individual thread
	// next sum the largest values for each warp

	// write the sum of the warp to smem
	// 每个线程束最后一个线程(31)将本线程束的归约值写入共享内存
	if (threadIdx.x % warpSize == warpSize - 1)
	{
		sums[warp_id] = value;
	}

	__syncthreads();

	//
	// scan sum the warp sums
	// the same shfl scan operation, but performed on warp sums
	//
	//	只有第0个线程工作
	//	且数据落入线程内的线程才工作
	//	最后归约结果写回共享内存中
	if (warp_id == 0 && lane_id < (blockDim.x / warpSize))
	{
		int warp_sum = sums[lane_id];

		for (int i = 1; i <= width; i *= 2)
		{
			int n = __shfl_up(warp_sum, i, width);

			if (lane_id >= i) warp_sum += n;
		}

		sums[lane_id] = warp_sum;
	}

	__syncthreads();

	// perform a uniform add across warps in the block
	// read neighboring warp's sum and add it to threads value
	//	
	int blockSum = 0;

	//	非第一个线程束的每个线程获得该线程束scan到的值（读的值串一个值）
	if (warp_id > 0)
	{
		blockSum = sums[warp_id - 1];
	}

	//	本线程持有值加上
	value += blockSum;

	//	写出自己的值
	data[id] = value;

	// last thread has sum, write write out the block's sum
	//	表明是第一次调用内核，需要写入部分核数组，供下次调用使用
	if (partial_sums != NULL && threadIdx.x == blockDim.x - 1)
	{
		partial_sums[blockIdx.x] = value;
	}
}

// Uniform add: add partial sums array
__global__ void uniform_add(int *data, int *partial_sums, int len)
{
	__shared__ int buf;
	int id = ((blockIdx.x * blockDim.x) + threadIdx.x);

	if (id > len) return;

	if (threadIdx.x == 0)
	{
		buf = partial_sums[blockIdx.x];
	}

	__syncthreads();
	data[id] += buf;
}

static unsigned int iDivUp(unsigned int dividend, unsigned int divisor)
{
	return ((dividend % divisor) == 0) ?
		(dividend / divisor) :
		(dividend / divisor + 1);
}


// This function verifies the shuffle scan result, for the simple
// prefix sum case.
bool CPUverify(int *h_data, int *h_result, int n_elements)
{
	// cpu verify
	for (int i = 0; i < n_elements - 1; i++)
	{
		h_data[i + 1] = h_data[i] + h_data[i + 1];
	}

	int diff = 0;

	for (int i = 0; i < n_elements; i++)
	{
		diff += h_data[i] - h_result[i];
	}

	printf("CPU verify result diff (GPUvsCPU) = %d\n", diff);
	bool bTestResult = false;

	if (diff == 0) bTestResult = true;

	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	for (int j = 0; j < 100; j++)
		for (int i = 0; i < n_elements - 1; i++)
		{
			h_data[i + 1] = h_data[i] + h_data[i + 1];
		}

	sdkStopTimer(&hTimer);
	double cput = sdkGetTimerValue(&hTimer);
	printf("CPU sum (naive) took %f ms\n", cput / 100);
	return bTestResult;
}


// this verifies the row scan result for synthetic data of all 1's
unsigned int verifyDataRowSums(unsigned int *h_image, int w, int h)
{
	unsigned int diff = 0;

	for (int j = 0; j < h; j++)
	{
		for (int i = 0; i < w; i++)
		{
			int gold = i + 1;
			diff += abs((int)gold - (int)h_image[j*w + i]);
		}
	}

	return diff;
}

bool shuffle_simple_test(int argc, char **argv)
{
	//	主机输入数组、部分和、结果
	int *h_data, *h_partial_sums, *h_result;
	//	设备输入数组、部分和
	int *d_data, *d_partial_sums;
	//	归约元素
	const int n_elements = 8192;
	//	归约元素的长度(bytes)
	int sz = sizeof(int)*n_elements;
	int cuda_device = 0;

	//	GPU分配内存，长度相同
	checkCudaErrors(cudaMallocHost((void **)&h_data, sizeof(int)*n_elements));
	checkCudaErrors(cudaMallocHost((void **)&h_result, sizeof(int)*n_elements));

	//	初始化数据
	for (int i = 0; i < n_elements; i++)
	{
		h_data[i] = 1;
	}

	//	block内线程数
	int blockSize = 1024;
	//	grid内block数 = N/blockSize
	int gridSize = n_elements / blockSize;
	//	一个block中warp的数
	int nWarps = blockSize / 32;
	//	共享内存长度(bytes) = warp数
	int shmem_sz = nWarps * sizeof(int);
	//	部分和数组长度 = grid内block数
	int n_partialSums = n_elements / blockSize;
	//	部分和数组长度(bytes)
	int partial_sz = n_partialSums*sizeof(int);

	printf("Scan summation for %d elements, %d partial sums\n",
		n_elements, n_elements / blockSize);

	//	第二次调用内核时的线程块数
	int p_blockSize = min(n_partialSums, blockSize);
	int p_gridSize = iDivUp(n_partialSums, p_blockSize);
	printf("Partial summing %d elements with %d blocks of size %d\n",
		n_partialSums, p_gridSize, p_blockSize);

	//	初始化计时器
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	float et = 0;
	float inc = 0;

	//	申请设备内存
	checkCudaErrors(cudaMalloc((void **)&d_data, sz));
	checkCudaErrors(cudaMalloc((void **)&d_partial_sums, partial_sz));
	checkCudaErrors(cudaMemset(d_partial_sums, 0, partial_sz));

	checkCudaErrors(cudaMallocHost((void **)&h_partial_sums, partial_sz));
	checkCudaErrors(cudaMemcpy(d_data, h_data, sz, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaEventRecord(start, 0));
	shfl_scan_test << <gridSize, blockSize, shmem_sz >> >(d_data, 32, d_partial_sums);
	shfl_scan_test << <p_gridSize, p_blockSize, shmem_sz >> >(d_partial_sums, 32);
	uniform_add << <gridSize - 1, blockSize >> >(d_data + blockSize, d_partial_sums, n_elements);
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&inc, start, stop));
	et += inc;

	checkCudaErrors(cudaMemcpy(h_result, d_data, sz, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_partial_sums, d_partial_sums, partial_sz,
		cudaMemcpyDeviceToHost));

	printf("Test Sum: %d\n", h_partial_sums[n_partialSums - 1]);
	printf("Time (ms): %f\n", et);
	printf("%d elements scanned in %f ms -> %f MegaElements/s\n",
		n_elements, et, n_elements / (et / 1000.0f) / 1000000.0f);

	bool bTestResult = CPUverify(h_data, h_result, n_elements);

	checkCudaErrors(cudaFreeHost(h_data));
	checkCudaErrors(cudaFreeHost(h_result));
	checkCudaErrors(cudaFreeHost(h_partial_sums));
	checkCudaErrors(cudaFree(d_data));
	checkCudaErrors(cudaFree(d_partial_sums));

	return bTestResult;
}

int main(int argc, char *argv[])
{
	// Initialization.  The shuffle intrinsic is not available on SM < 3.0
	// so waive the test if the hardware is not present.
	int cuda_device = 0;

	printf("Starting shfl_scan\n");

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	cuda_device = findCudaDevice(argc, (const char **)argv);

	cudaDeviceProp deviceProp;
	checkCudaErrors(cudaGetDevice(&cuda_device));

	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));

	printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n",
		deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

	// __shfl intrinsic needs SM 3.0 or higher
	if (deviceProp.major < 3)
	{
		printf("> __shfl() intrinsic requires device SM 3.0+\n");
		printf("> Waiving test.\n");
		exit(EXIT_WAIVED);
	}


	bool bTestResult = true;
	bool simpleTest = shuffle_simple_test(argc, argv);
	//bool intTest = shuffle_integral_image_test();

	//bTestResult = simpleTest & intTest;
	bTestResult = simpleTest;

	exit((bTestResult) ? EXIT_SUCCESS : EXIT_FAILURE);
}
