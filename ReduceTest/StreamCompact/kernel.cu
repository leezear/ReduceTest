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
#include <random>
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
	//	���빲���ڴ棬block֮��Ĺ����ڴ治���໥����
	extern __shared__ int sums[];
	//	��ȡȫ��id
	int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
	//	�߳������߳�ID
	int lane_id = id % warpSize;
	//	block�ڵľֲ��߳���ID
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

	//	����Խ��Խ��
	for (int i = 1; i <= width; i *= 2)
	{
		//	__shfl_up(����, [1, 2, 4, ...,32], 32)
		int n = __shfl_up(value, i, width);

		//	û����Ĳ��ֲ������
		if (lane_id >= i) value += n;
	}

	// value now holds the scan value for the individual thread
	// next sum the largest values for each warp

	// write the sum of the warp to smem
	// ÿ���߳������һ���߳�(31)�����߳����Ĺ�Լֵд�빲���ڴ�
	if (threadIdx.x % warpSize == warpSize - 1)
	{
		sums[warp_id] = value;
	}

	__syncthreads();

	//
	// scan sum the warp sums
	// the same shfl scan operation, but performed on warp sums
	//
	//	ֻ�е�0���̹߳���
	//	�����������߳��ڵ��̲߳Ź���
	//	����Լ���д�ع����ڴ���
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

	//	�ǵ�һ���߳�����ÿ���̻߳�ø��߳���scan����ֵ������ֵ��һ��ֵ��
	if (warp_id > 0)
	{
		blockSum = sums[warp_id - 1];
	}

	//	���̳߳���ֵ����
	value += blockSum;

	//	д���Լ���ֵ
	data[id] = value;

	// last thread has sum, write write out the block's sum
	//	�����ǵ�һ�ε����ںˣ���Ҫд�벿�ֺ����飬���´ε���ʹ��
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
			h_data[i + 1] = h_data[i] + h_data[i + 1];

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

int main(int argc, char *argv[])
{
	//	���������ж����顢���ֺ͡����
	int *h_pre_data, *h_pre_partial_sums, *h_pre_result;
	//	�豸�������顢���ֺ�
	int *d_data, *d_partial_sums;
	//	��ԼԪ��
	//	����blocksize = 256  1<<17��Ԫ�ؼ������Ǵ���ģ� �趨blocksize = 512�����ȷ
	const int N = 1 << 16;
	//	��ԼԪ�صĳ���(bytes)
	int sz = sizeof(int)*N;
	int cuda_device = 0;

	//	GPU�����ڴ棬������ͬ
	checkCudaErrors(cudaMallocHost((void **)&h_pre_data, sizeof(int)*N));
	checkCudaErrors(cudaMallocHost((void **)&h_pre_result, sizeof(int)*N));

	//���������
	std::default_random_engine generator;
	std::uniform_int_distribution<int> dis(0, 1);

	//	��ʼ������
	for (int i = 0; i < N; i++)
	{
		h_pre_data[i] = dis(generator);
	}

	//	block���߳���
	int blockSize = 256;
	//	grid��block�� = N/blockSize
	//	131072
	int gridSize = N / blockSize;
	//	һ��block��warp����
	int nWarps = blockSize / 32;
	//	�����ڴ泤��(bytes) = warp��
	int shmem_sz = nWarps * sizeof(int);
	//	���ֺ����鳤�� = grid��block��
	int n_partialSums = N / blockSize;
	//	���ֺ����鳤��(bytes)
	int partial_sz = n_partialSums*sizeof(int);

	printf("Scan summation for %d elements, %d partial sums\n", N, N / blockSize);

	//	�ڶ��ε����ں�ʱ���߳̿���
	int p_blockSize = min(n_partialSums, blockSize);
	int p_gridSize = iDivUp(n_partialSums, p_blockSize);
	printf("Partial summing %d elements with %d blocks of size %d\n", n_partialSums, p_gridSize, p_blockSize);

	//	��ʼ����ʱ��
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	float et = 0;
	float inc = 0;

	//	�����豸�ڴ�
	checkCudaErrors(cudaMalloc((void **)&d_data, sz));
	checkCudaErrors(cudaMalloc((void **)&d_partial_sums, partial_sz));
	checkCudaErrors(cudaMemset(d_partial_sums, 0, partial_sz));

	checkCudaErrors(cudaMallocHost((void **)&h_pre_partial_sums, partial_sz));
	checkCudaErrors(cudaMemcpy(d_data, h_pre_data, sz, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaEventRecord(start, 0));
	shfl_scan_test << <gridSize, blockSize, shmem_sz >> >(d_data, 32, d_partial_sums);
	shfl_scan_test << <p_gridSize, p_blockSize, shmem_sz >> >(d_partial_sums, 32);
	uniform_add << <gridSize - 1, blockSize >> >(d_data + blockSize, d_partial_sums, N);
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&inc, start, stop));
	et += inc;

	checkCudaErrors(cudaMemcpy(h_pre_result, d_data, sz, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_pre_partial_sums, d_partial_sums, partial_sz,
		cudaMemcpyDeviceToHost));

	printf("Test Sum: %d\n", h_pre_partial_sums[n_partialSums - 1]);
	printf("Time (ms): %f\n", et);
	printf("%d elements scanned in %f ms -> %f MegaElements/s\n", N, et, N / (et / 1000.0f) / 1000000.0f);

	bool bTestResult = CPUverify(h_pre_data, h_pre_result, N);
	printf("h_result = %d\n ", h_pre_result[N - 1]);

	checkCudaErrors(cudaFreeHost(h_pre_data));
	checkCudaErrors(cudaFreeHost(h_pre_result));
	checkCudaErrors(cudaFreeHost(h_pre_partial_sums));
	checkCudaErrors(cudaFree(d_data));
	checkCudaErrors(cudaFree(d_partial_sums));
}
