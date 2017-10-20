/*https://cdac.in/index.aspx?id=ev_hpc_gpu-comp-nvidia-cuda-streams#hetr-cuda-prog-cuda-streams*/

#include <stdio.h> 
#include <time.h> 
#include <cuda.h> 

#define BLOCKSIZE 256
#define SIZEOFARRAY 1048576*4
#define KENERL_LOOP 400
 
// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

__global__ void arrayAddition(int *device_a, int *device_b, int *device_result, const int offset)
{

	int threadId = threadIdx.x + blockIdx.x * blockDim.x ;
	int index = threadId + offset;

	if (threadId < SIZEOFARRAY)
			for (int i =0; i < KENERL_LOOP; i++)
        device_result[index]= device_a[index]+device_b[index]; 
} 

__host__ void generate_rand_data(unsigned int * host_data_ptr)
{
        for(unsigned int i=0; i < SIZEOFARRAY; i++)
        {
                host_data_ptr[i] = (unsigned int) rand();
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

/* Check for safe return of all calls to the device */ 
int main ( int argc, char **argv ) 
{ 

	// Get cuda properties
  cudaDeviceProp prop; 
  cudaSetDevice(0);
  cudaGetDeviceProperties( &prop, 0); 
  printf("maxThreadsPerBlock is %d \n", prop.maxThreadsPerBlock);

  // Allocate device and host memory
  const int num_streams = 4;
  const int stream_size = SIZEOFARRAY / num_streams;
  const int stream_bytes = stream_size * sizeof(int);
  const int num_bytes = SIZEOFARRAY * sizeof(int);
  int *host_a, *host_b, *host_result; 
  int *device_a, *device_b, *device_result; 

  checkCuda(cudaMalloc( ( void**)& device_a, num_bytes)); 
  checkCuda(cudaMalloc( ( void**)& device_b, num_bytes )); 
  checkCuda(cudaMalloc( ( void**)& device_result, num_bytes)); 

  checkCuda(cudaHostAlloc((void **)&host_a, num_bytes, cudaHostAllocDefault));
  checkCuda(cudaHostAlloc((void **)&host_b, num_bytes, cudaHostAllocDefault));
  checkCuda(cudaHostAlloc((void **)&host_result, num_bytes, cudaHostAllocDefault));
  
  // Instantiate cuda events and streams
  cudaEvent_t start, stop, start2, stop2;
  float elapsedTime, elapsedTime2; 

  // Create Streams
  cudaStream_t orig;
  cudaStream_t stream[num_streams];
  checkCuda( cudaStreamCreate(&orig));
  for (int i = 0; i < num_streams; ++i)
    checkCuda( cudaStreamCreate(&stream[i]) );

  // Instantiate host values
  for(int index = 0; index < SIZEOFARRAY; index++) 
  { 
  	host_a[index] = index;
  	host_b[index] = SIZEOFARRAY - index;
  } 

  // Run sequential version
  start_measure(&start, &stop);
  checkCuda(cudaMemcpyAsync(device_a, host_a, num_bytes, cudaMemcpyHostToDevice, orig)); 
	checkCuda(cudaMemcpyAsync(device_b, host_b, num_bytes, cudaMemcpyHostToDevice, orig)); 
  arrayAddition<<<SIZEOFARRAY/BLOCKSIZE, BLOCKSIZE>>>(device_a, device_b, device_result, 0);
  checkCuda(cudaMemcpyAsync(host_result, device_result, num_bytes, cudaMemcpyDeviceToHost, orig)); 
  stop_measure(&start, &stop, elapsedTime);

  // Run overlapped stream processing
  //		each stream processes portions of the data
  start_measure(&start2, &stop2);

  for (int i = 0; i < num_streams; ++i) {
	  int offset = i * stream_size;
  	checkCuda(cudaMemcpyAsync(&device_a[offset], &host_a[offset], stream_bytes, cudaMemcpyHostToDevice, stream[i])); 
  	checkCuda(cudaMemcpyAsync(&device_b[offset], &host_b[offset], stream_bytes, cudaMemcpyHostToDevice, stream[i]));
	}

	for (int i = 0; i < num_streams; ++i) {
	  int offset = i * stream_size;
	  arrayAddition<<<stream_size/BLOCKSIZE, BLOCKSIZE, 0, stream[i]>>>(device_a, device_b, device_result, offset);
	}

	for (int i = 0; i < num_streams; ++i) {
	  int offset = i * stream_size;
	  checkCuda(cudaMemcpyAsync(&host_result[offset], &device_result[offset], stream_bytes, cudaMemcpyDeviceToHost, stream[i])); 
	}
  stop_measure(&start2, &stop2, elapsedTime2);

  printf("\n Block size: %d \n", BLOCKSIZE);
  printf("\n Kernal loop size: %d \n", KENERL_LOOP); 
  printf("\n Size of array : %d \n", SIZEOFARRAY); 
  printf("\n Sequential Time taken: %3.1f ms \n", elapsedTime);
  printf("\n Streams Time taken: %3.1f ms \n", elapsedTime2);

  cudaFreeHost(host_a); 
  cudaFreeHost(host_b); 
  cudaFreeHost(host_result); 
  cudaFree(device_a); 
  cudaFree(device_b); 
  cudaFree(device_result); 

  return 0; 
}
