
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <time.h>

cudaError_t sumPrefix(unsigned long long* input, unsigned long long* results, long vector_size, int block_size);

bool checkForError(const cudaError_t cudaStatus, const char text[]) {
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\n\n%s \nError code: %d \nStatus: %s \n\n", text, cudaStatus, cudaGetErrorString(cudaStatus));
		return true;
	}
	return false;
}

__global__ 
void prefix_sum_kernel(unsigned long long* input, unsigned long long* results, unsigned long long exp, int vector_size) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < vector_size) {
		if (i < exp) {
			results[i] = input[i];
		}
		else {
			results[i] = input[i] + input[i - exp];
		}
	}
}

__global__
void copy_kernel(unsigned long long* input, unsigned long long* results, int size) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size) {
		input[i] = results[i];
	}
}

bool test(unsigned long long* results, unsigned long long* input, long arraySize) {
	bool prefix_sum_ok = true;
	unsigned long long sum = 1;
	printf("\n\n");

	for (int i = 1; i < arraySize; i++) {
		sum += input[i];
		if (!(results[i] == sum)) {
			printf("BLAD! NIE ZGADZA SIE! oczekiwana = %lld, dostalem = %lld\n", sum, results[i]);
			prefix_sum_ok = false;
		}
	}
	return prefix_sum_ok;
}

int main()
{
	long vector_size;
	long block_size;

	printf("Podaj rozmiar wektora >>> ");
	scanf("%d", &vector_size);
	printf("Podaj liczbe watkow w bloku >>> ");
	scanf("%d", &block_size);

	//const long vector_size = 67108864+2; // 8 digits, max 99999999
	//const long block_size = 512;
	
	unsigned long long* input = (unsigned long long*)malloc(vector_size * sizeof(unsigned long long));
	unsigned long long* results = (unsigned long long*)malloc(vector_size * sizeof(unsigned long long));
	
	for (int i = 0; i < vector_size; i++) {
		input[i] = i + 1;
	}
	
	cudaError_t cudaStatus = sumPrefix(input, results, vector_size, block_size);
	if (checkForError(cudaStatus, "sumPrefix failed!")) {
		return 1;
	}

	test(results, input, vector_size);
	printf("%lld, ", results[vector_size - 1]);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (checkForError(cudaStatus, "cudaDeviceReset failed!")) {
		return 1;
	}

	return 0;
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t sumPrefix(unsigned long long* input, unsigned long long* results, long vector_size, int block_size)
{
	int num_blocks = (vector_size + block_size - 1) / block_size;
	int iterations = ceil((int)log2(vector_size));
	unsigned long long exp = 1;
	int i;
	 
	printf("iterations = %d, vector_size = %d, block_size = %d, num_blocks = %d", iterations, vector_size, block_size, num_blocks);

	cudaError_t cudaStatus;
	unsigned long long* dev_input = 0;
	unsigned long long* dev_results = 0;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (checkForError(cudaStatus, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?")) {
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_input, vector_size * sizeof(unsigned long long));
	if (checkForError(cudaStatus, "cudaMalloc (dev_input) failed!")) {
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_results, vector_size * sizeof(unsigned long long));
	if (checkForError(cudaStatus, "cudaMalloc (dev_results) failed!")) {
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_input, input, vector_size * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	if (checkForError(cudaStatus, "cudaMemcpy (host -> dev, dev_input) failed!")) {
		goto Error;
	}
	
	printf("\n\nSTART");
	for (i = 0; i <= iterations; i++) {
		prefix_sum_kernel << <num_blocks, block_size >> > (dev_input, dev_results, exp, vector_size);
		exp *= 2;

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (checkForError(cudaStatus, "prefix_sum_kernel launch failed!")) {
			goto Error;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (checkForError(cudaStatus, "cudaDeviceSynchronize on \"prefix_sum_kernel\" returned error code.")) {
			goto Error;
		}

		copy_kernel << <num_blocks, block_size >> > (dev_input, dev_results, vector_size);
		cudaStatus = cudaGetLastError();
		if (checkForError(cudaStatus,  "copy_kernel launch failed")) {
			goto Error;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (checkForError(cudaStatus,  "cudaDeviceSynchronize on \"copy_kernel\" returned error code")) {
			goto Error;
		}
	}
	printf("\nSTOP");

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(results, dev_results, vector_size * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	if (checkForError(cudaStatus, "cudaMemcpy (dev -> host, dev_results) failed!")) {
		goto Error;
	}

Error:
	cudaFree(dev_input);
	cudaFree(dev_results);

	return cudaStatus;
}
