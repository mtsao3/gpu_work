#include <stdio.h>
#include <stdlib.h>

#define NUM_ELEMENTS 8192
#define MAX_THREADS_PER_BLOCK 1024
#define KERNEL_LOOP 100000

__host__ void generate_rand_data(unsigned int * host_data_ptr)
{
        for(unsigned int i=0; i < NUM_ELEMENTS; i++)
        {
                host_data_ptr[i] = (unsigned int) rand();
        }
}

__global__ void test_gpu_global(unsigned int * const data, 
                                        const unsigned int num_elements,
                                        const unsigned int loop_iter)
{
        const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
        if(tid < num_elements)
        {
                // compute the average of this thread's left and right neighbors and place in register
                // Place value directly in register
                float tmp = (data[tid > 0 ? tid - 1 : NUM_ELEMENTS-1] + data[tid < NUM_ELEMENTS-1 ? tid + 1 : 0]) * 0.5f;
                for(int i = 0; i < KERNEL_LOOP; i++) {
                        tmp += data[tid];
                }
                data[tid] = tmp;
                __syncthreads();
        }
}

__global__ void test_gpu_register(unsigned int * const data, 
                                        const unsigned int num_elements,
                                        const unsigned int loop_iter)
{
        const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
        if(tid < num_elements)
        {
                // compute the average of this thread's left and right neighbors and place in register
                // Place value directly in register
                float tmp = (data[tid > 0 ? tid - 1 : NUM_ELEMENTS-1] + data[tid < NUM_ELEMENTS-1 ? tid + 1 : 0]) * 0.5f;
                for(int i = 0; i < KERNEL_LOOP; i++) {
                        tmp += tmp;
                }
                data[tid] = tmp;
                __syncthreads();
        }
}

__global__ void test_gpu_shared(unsigned int * const data, 
                                        const unsigned int num_elements,
                                        const unsigned int loop_iter)
{
        const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
        __shared__ unsigned int tmp_0[NUM_ELEMENTS];
        if(tid < num_elements)
        {
                // compute the average of this thread's left and right neighbors and place in register
                // Place valude directly in shared
                tmp_0[tid] = (data[tid > 0 ? tid - 1 : NUM_ELEMENTS-1] + data[tid < NUM_ELEMENTS-1 ? tid + 1 : 0]) * 0.5f;
                float tmp;
                for(int i = 0; i < KERNEL_LOOP; i++) {
                        tmp += tmp_0[tid];
                }
                data[tid] = tmp;
                __syncthreads();
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
__host__ void display_data(const unsigned int * const in_data, const unsigned int * const out_data) {
        for(int i = 0; i < 20; i ++){
                printf("i=%i, input_data = %u, output_data = %u\n", i, in_data[i], out_data[i]);
        }
}

__host__ void gpu_kernel(void)
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
        generate_rand_data(host_pinned);

        // Prep Device data
        unsigned int * data_gpu;
        cudaMalloc(&data_gpu, num_bytes);

         // Define measurement 
        float time;
        cudaEvent_t kernel_start, kernel_stop;
        cudaEvent_t kernel_start1, kernel_stop1;
        cudaEvent_t kernel_start2, kernel_stop2;


        // Run Test with gpu shared
        cudaMemcpy(data_gpu, host_pinned, num_bytes, cudaMemcpyHostToDevice);
        start_measure(&kernel_start, &kernel_stop);
        test_gpu_global<<<num_blocks, num_threads>>>(data_gpu, num_elements, kernel_loop);
        stop_measure(&kernel_start, &kernel_stop, time);
        printf("test_gpu_global took %f\n", time);

        cudaMemcpy(host_pinned_final, data_gpu, num_bytes,cudaMemcpyDeviceToHost);
        display_data(host_pinned, host_pinned_final);

        
        // Run Test with gpu registers
        cudaMemcpy(data_gpu, host_pinned, num_bytes, cudaMemcpyHostToDevice);    // referesh data
        start_measure(&kernel_start1, &kernel_stop1);
        test_gpu_shared <<<num_blocks, num_threads>>>(data_gpu, num_elements, kernel_loop);
        stop_measure(&kernel_start1, &kernel_stop1, time);
        printf("test_gpu_shared took %f\n", time);

        cudaMemcpy(host_pinned_final, data_gpu, num_bytes,cudaMemcpyDeviceToHost);
        display_data(host_pinned, host_pinned_final);

        // Run Test with gpu global
        cudaMemcpy(data_gpu, host_pinned, num_bytes, cudaMemcpyHostToDevice);    // referesh data
        start_measure(&kernel_start2, &kernel_stop2);
        test_gpu_register <<<num_blocks, num_threads>>>(data_gpu, num_elements, kernel_loop);
        stop_measure(&kernel_start2, &kernel_stop2, time);
        printf("test_gpu_register took %f\n", time);

        cudaMemcpy(host_pinned_final, data_gpu, num_bytes,cudaMemcpyDeviceToHost);
        display_data(host_pinned, host_pinned_final);

        cudaFree((void* ) data_gpu);
        cudaFreeHost(host_pinned_final);
        cudaFreeHost(host_pinned);
        cudaDeviceReset();

}


/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(void) {
        // Find out statistics of GPU 
        int nDevices;
        cudaGetDeviceCount(&nDevices);
        for (int i = 0; i < nDevices; i++) {
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, i);
                printf("maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
        }
	gpu_kernel();

	return EXIT_SUCCESS;
}