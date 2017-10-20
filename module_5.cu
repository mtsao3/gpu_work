#include <stdio.h>
#include <stdlib.h>
#include <iostream>


#define KERNEL_LOOP 4096 // Used to exacerbate runtime affects 
#define NUM_ELEMENTS 4096
#define MAX_THREADS_PER_BLOCK 1024
__constant__ unsigned int const_data_gpu[NUM_ELEMENTS];
static unsigned int const_data_host[NUM_ELEMENTS];

__device__ void manipulate_shared_data(unsigned int * const data,
																		 	  unsigned int * const s_data,
																				const unsigned int tid)
{	
	// Load data into shared memory while doing an operation at the same time
	unsigned int i = gridDim.x*(blockDim.x)/2;
	if(tid < i){
		s_data[tid] = data[tid] + data[tid+i];
	}
	__syncthreads();
	

	// Add up data through reduction
	// Divides the probelm in half, but half the threads are also idle.
	// I know this isn't the fastest way through, but it's a start
	// Then d
	for (unsigned int s=gridDim.x*blockDim.x/4; s>32; s>>=1) {
		if (tid < s) {
				s_data[tid] += s_data[tid + s];
		}
		__syncthreads();
	}

	// Unwrap last loop
	if (tid < 32) {
		s_data[tid] += s_data[tid+32];
		s_data[tid] += s_data[tid+16];
		s_data[tid] += s_data[tid+8];
		s_data[tid] += s_data[tid+4];
		s_data[tid] += s_data[tid+2];
		s_data[tid] += s_data[tid+1];
	}
	__syncthreads();
}

__device__ void manipulate_constant_data(unsigned int * const data,
																				const unsigned int tid)
{	
	// Load data into shared memory while doing an operation at the same time
	unsigned int i = gridDim.x*(blockDim.x)/2;
	if(tid < i){
		data[tid] = const_data_gpu[tid] + const_data_gpu[tid+i];
	}
	__syncthreads();
	

	// Add up data through reduction
	// Divides the probelm in half, but half the threads are also idle.
	// I know this isn't the fastest way through, but it's a start
	// Then d
	for (unsigned int s=gridDim.x*blockDim.x/4; s>32; s>>=1) {
		if (tid < s) {
				data[tid] += const_data_gpu[tid + s];
		}
		__syncthreads();
	}

	// Unwrap last loop
	if (tid < 32) {
		data[tid] += const_data_gpu[tid+32];
		data[tid] += const_data_gpu[tid+16];
		data[tid] += const_data_gpu[tid+8];
		data[tid] += const_data_gpu[tid+4];
		data[tid] += const_data_gpu[tid+2];
		data[tid] += const_data_gpu[tid+1];
	}
	__syncthreads();
}

__device__ void copy_data(const unsigned int * const src,
								unsigned int * const dst,
								const unsigned int num_elements,
								const unsigned int tid)
{
	if(tid < num_elements) {
		dst[tid] = src[tid];
	}
	__syncthreads();
}

__global__ void gpu_kernel(unsigned int * data, const unsigned int num_elements)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	__shared__ unsigned int shared[NUM_ELEMENTS];

	manipulate_shared_data(data, shared, thread_idx);

	copy_data(shared, data, num_elements, thread_idx);
}

__global__ void gpu_kernel_constants(unsigned int * data, const unsigned int num_elements)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	manipulate_constant_data(data, thread_idx);
}


__host__ void generate_rand_data(unsigned int * host_data_ptr)
{
	for(unsigned int i=0; i < NUM_ELEMENTS; i++)
	{
  	host_data_ptr[i] = (unsigned int) rand();
	}
}

__host__ void generate_sequetial_data(unsigned int * data) {
	for(unsigned int i=0; i < NUM_ELEMENTS; i++)
	{
  	data[i] = i;
	}
}

__host__ void start_measure(cudaEvent_t * start, cudaEvent_t *stop){
        cudaEventCreate(start,0);
        cudaEventCreate(stop,0);
        cudaEventRecord(*start, 0);
}

__host__ void stop_measure(cudaEvent_t* start, cudaEvent_t * stop, float &time) {
        cudaEventRecord(*stop, 0);
        cudaEventSynchronize(*stop);
        cudaEventElapsedTime(&time, *start, *stop);
}

// Create display function to ensure output is the same for both kernels
__host__ void display_data(const unsigned int * const in_data,
													 const unsigned int * const out_data,
													 const unsigned int num_iter) {
        for(int i = 0; i < num_iter; i ++){
                printf("i=%i, input_data = %u, output_data = %u\n", i, in_data[i], out_data[i]);
        }
}

int main ( int argc, char *argv[] )
{
  const unsigned int num_elements = NUM_ELEMENTS;
  const unsigned int num_threads = MAX_THREADS_PER_BLOCK;
  const unsigned int num_blocks = num_elements/num_threads;
  const unsigned int num_bytes = num_elements * sizeof(unsigned int);
  const unsigned int kernel_loop = KERNEL_LOOP;

  // Prep Host data
  unsigned int * host_pinned;
  unsigned int * host_pinned_final;
  cudaMallocHost((void**)&host_pinned, num_bytes);
  cudaMallocHost((void**)&host_pinned_final, num_bytes);
  generate_rand_data(const_data_host);
  generate_sequetial_data(host_pinned);

  // Prep Device data
  unsigned int * data_gpu;
  cudaMalloc(&data_gpu, num_bytes);

  // Define measurement 
  float time;
  cudaEvent_t kernel_start, kernel_stop;
  cudaEvent_t kernel_start1, kernel_stop1;

	// Copy new data to be manipulated
	cudaMemcpyToSymbol(const_data_gpu, const_data_host, num_bytes, 0, cudaMemcpyHostToDevice);
  cudaThreadSynchronize();
	start_measure(&kernel_start1, &kernel_stop1);
	cudaMemcpy(data_gpu, host_pinned, num_bytes, cudaMemcpyHostToDevice);
  // run gpu kernel
	gpu_kernel_constants <<<num_blocks, num_threads>>>(data_gpu, num_elements);
	/* Copy back the gpu results to the CPU */
	cudaMemcpy(host_pinned_final, data_gpu, num_bytes, cudaMemcpyDeviceToHost);	
	stop_measure(&kernel_start1, &kernel_stop1, time);
	std::cout << "Constant memory kernel took " << time << std::endl;

	display_data(const_data_host, host_pinned_final, 5);

	/* Allocate arrays on the GPU */
	cudaThreadSynchronize();
	start_measure(&kernel_start, &kernel_stop);
	cudaMemcpy(data_gpu, host_pinned, num_bytes, cudaMemcpyHostToDevice);
	// run gpu kernel
	gpu_kernel <<<num_blocks, num_threads>>>(data_gpu, num_elements);
	/* Copy back the gpu results to the CPU */
	cudaMemcpy(host_pinned_final, data_gpu, num_bytes, cudaMemcpyDeviceToHost);	
	stop_measure(&kernel_start, &kernel_stop, time);
	std::cout << "Shared memory kernel took " << time << std::endl;
	display_data(host_pinned, host_pinned_final, 5);



	/* Free the arrays on the GPU as now we're done with them */
	cudaFree(data_gpu);
  cudaFreeHost(host_pinned_final);
  cudaFreeHost(host_pinned);
}
