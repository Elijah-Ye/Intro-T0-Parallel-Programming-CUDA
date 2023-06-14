// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this
// #define SMALL_BLOCK_SIZE 128

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *X, float *Y, float *S, int InputSize) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  // I am using Brent Kung algorithm from pg.186 of the textbook
  
  __shared__ float XY[BLOCK_SIZE*2];
  int i = 2*blockIdx.x * blockDim.x + threadIdx.x;
  
  //loading data
  if(i < InputSize) XY[threadIdx.x] = X[i];
  else XY[threadIdx.x] = 0;
  if((i + blockDim.x) < InputSize) XY[threadIdx.x + blockDim.x] = X[i + blockDim.x];
  else XY[threadIdx.x + blockDim.x] = 0;

  // for(int stride = 1; stride <= blockDim.x; stride *= 2)
  // {
  //   __syncthreads();
  //   int index = (threadIdx.x + 1) * 2 * stride - 1;
  //   if(index < sectionSize && (index-stride >= 0)) XY[index] += XY[index - stride];
  // }

  int stride = 1;
  while(stride < 2*blockDim.x){
    __syncthreads();
    int index = (threadIdx.x + 1) * 2 * stride - 1;
    if(index < 2*blockDim.x && (index - stride) >= 0) XY[index] += XY[index - stride];
    stride = 2*stride;
  }

  // for(int stride = ceil(sectionSize/4.0); stride > 0; stride /= 2)
  // {
  //   __syncthreads();
  //   int index = (threadIdx.x + 1) * 2 * stride - 1;
  //   if((index + stride) < sectionSize) XY[index + stride] += XY[index];
  // }
  stride = blockDim.x/2;
  while(stride > 0)
  {
    __syncthreads();
    int index = (threadIdx.x + 1) * 2 * stride - 1;
    if((index + stride) < 2*blockDim.x) XY[index + stride] += XY[index];
    stride = stride / 2;
  }

  __syncthreads();
  if(i < InputSize) Y[i] = XY[threadIdx.x];
  if((i + blockDim.x) < InputSize) Y[i + blockDim.x] = XY[threadIdx.x + blockDim.x];

  // additional code for Hierarchical Parallel scan on pg.192 of the textbook
  if(S){
    __syncthreads();
    if(threadIdx.x == (blockDim.x - 1)) S[blockIdx.x] = XY[blockDim.x * 2 - 1];
  }

}

__global__ void addSum(float *S, float *Y, int InputSize)
{
  //@@ This kernel taes the S and Y arrays as inputs and writes
  //@@ its output back into Y
  //@@ pg.192 of the textbook
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < InputSize && blockIdx.x > 0){
    Y[i] += S[blockIdx.x - 1];
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *S; // helper array to store the sum of each section
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);
  int numSections = ceil(numElements/(BLOCK_SIZE*2.0));
  // I guess they did all the malloc and memcpy for us.
  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&S, numSections * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbCheck(cudaMemset(S, 0, numSections * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  // Phase 1: Finding the partial sum for each section
  dim3 dimGrid(ceil(numElements/(2.0 * BLOCK_SIZE)), 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, S, numElements);
  cudaDeviceSynchronize();

  // Phase 2: Finding the partial sum for the helper array
  dim3 dimGrid2(ceil(numSections/(BLOCK_SIZE*2.0)), 1, 1);
  dim3 dimBlock2(BLOCK_SIZE, 1, 1);
  scan<<<dimGrid2, dimBlock2>>>(S, S, NULL, numSections);
  cudaDeviceSynchronize();

  // Phase 3: Calculating the final partial sums base on the calculated
  dim3 dimGrid3(ceil(numElements/(BLOCK_SIZE*2.0)),1, 1);
  dim3 dimBlock3(BLOCK_SIZE*2, 1, 1);
  addSum<<<dimGrid3, dimBlock3>>>(S, deviceOutput, numElements);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(S);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
