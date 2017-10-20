#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>


#define NUM_ELEMENTS 8192

// Non interleaved structure definition
typedef unsigned int ARRAY_MEMBER_T[NUM_ELEMENTS];
typedef struct {
	ARRAY_MEMBER_T a;
	ARRAY_MEMBER_T b;
	ARRAY_MEMBER_T c;
	ARRAY_MEMBER_T d;
} NON_INTERLEAVED_T;


// Multiply kernel
__global__ void multiply_kernel(
		NON_INTERLEAVED_T * const dest_ptr,
		NON_INTERLEAVED_T * const src_ptr,
		const unsigned int num_elements) {
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(tid < num_elements)
	{
			dest_ptr->a[tid] *= src_ptr->a[tid];
			dest_ptr->b[tid] *= src_ptr->b[tid];
			dest_ptr->c[tid] *= src_ptr->c[tid];
			dest_ptr->d[tid] *= src_ptr->d[tid];
	}
}

int main(void)
{
  // Define structs
  int bytes = sizeof(NON_INTERLEAVED_T);
  NON_INTERLEAVED_T *x, *y, *x_pin, *y_pin, *d_x, *d_y;
  const unsigned int num_threads = 256;
	const unsigned int num_blocks = (NUM_ELEMENTS + (num_threads-1)) / num_threads;

    // Define measurement 
  cudaEvent_t kernel_start, kernel_stop;
  cudaEvent_t kernel_start1, kernel_stop1;
  cudaEventCreate(&kernel_start,0);
	cudaEventCreate(&kernel_stop,0);
  cudaEventCreate(&kernel_start1,0);
	cudaEventCreate(&kernel_stop1,0);

  // Allocate pageable memory
  x = (NON_INTERLEAVED_T*)malloc(bytes);
  y = (NON_INTERLEAVED_T*)malloc(bytes);
  
  // Allocate pinned memory
  cudaMallocHost((void**)&x_pin, bytes);
  cudaMallocHost((void**)&y_pin, bytes);
  
  // Allocate device memory
  cudaMalloc(&d_x, bytes); 
  cudaMalloc(&d_y, bytes);
  
  // Fill data
  float x_val = 3.0f;
  float y_val = 2.0f;
  for (int i = 0; i < NUM_ELEMENTS; i++) {
    x->a[i] = x_val;
    x->b[i] = x_val;
    x->c[i] = x_val;
    x->d[i] = x_val;
    y->a[i] = y_val;
    y->b[i] = y_val;
    y->c[i] = y_val;
    y->d[i] = y_val;
    x_pin->a[i] = x_val;
    x_pin->b[i] = x_val;
    x_pin->c[i] = x_val;
    x_pin->d[i] = x_val;
    y_pin->a[i] = y_val;
    y_pin->b[i] = y_val;
    y_pin->c[i] = y_val;
    y_pin->d[i] = y_val;
  }
  
  cudaEventRecord(kernel_start, 0);

  // Copy pageable memory
  cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, bytes, cudaMemcpyHostToDevice);
  
  // Apply Kernel
  multiply_kernel<<<num_blocks, num_threads>>>(d_y, d_x, NUM_ELEMENTS);

  // Copy data back
  cudaMemcpy(y, d_y, bytes, cudaMemcpyDeviceToHost);
  
  // Display metrics
  cudaEventRecord(kernel_stop, 0);
  cudaEventSynchronize(kernel_stop);
	float delta = 0.0F;
	cudaEventElapsedTime(&delta, kernel_start, kernel_stop);
  std::cout << "Pageable multiply took " << delta << std::endl;

  cudaEventRecord(kernel_start1, 0);

  // Copy pinned memory
  cudaMemcpy(d_x, x_pin, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y_pin, bytes, cudaMemcpyHostToDevice);
  
  // Apply kernel
  multiply_kernel<<<num_blocks, num_threads>>>(d_y, d_x, NUM_ELEMENTS);
  
  // Copy memory back
  cudaMemcpy(y_pin, d_y, bytes, cudaMemcpyDeviceToHost);
  
  // Display metrics
  cudaEventRecord(kernel_stop1, 0);
	cudaEventSynchronize(kernel_stop1);
  float delta1 = 0.0F;
	cudaEventElapsedTime(&delta1, kernel_start1, kernel_stop1);
  std::cout << "Pinned multiply took " << delta1 << std::endl;
  
  // // Print some values for validation
  // for(int i = NUM_ELEMENTS-3; i < NUM_ELEMENTS; i++) {
  //   std::cout << "y_pin.a[" << i << "] = " << y_pin->a[i] << std::endl;
  //   std::cout << "y.a[" << i << "] = " << y->a[i] << std::endl;
  //   std::cout << "y_pin.b[" << i << "] = " << y_pin->b[i] << std::endl;
  //   std::cout << "y.b[" << i << "] = " << y->b[i] << std::endl;
  //   std::cout << "y_pin.c[" << i << "] = " << y_pin->c[i] << std::endl;
  //   std::cout << "y.c[" << i << "] = " << y->c[i] << std::endl;
  //   std::cout << "y_pin.d[" << i << "] = " << y_pin->d[i] << std::endl;
  //   std::cout << "y.d[" << i << "] = " << y->d[i] << std::endl;
  // }

  // House keeping
  cudaEventDestroy(kernel_start);
	cudaEventDestroy(kernel_stop);
  cudaEventDestroy(kernel_start1);
	cudaEventDestroy(kernel_stop1);
  cudaFreeHost(x_pin);
  cudaFreeHost(x_pin);
  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}

