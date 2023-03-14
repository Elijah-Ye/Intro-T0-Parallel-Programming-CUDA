#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASK_WIDTH 3
#define MASK_RADIUS 1
#define TILE_WIDTH 8
#define BLOCK_WIDTH (TILE_WIDTH + (MASK_WIDTH - 1))

//@@ Define constant memory for device kernel here
__constant__ float M_KDevice[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  int x_o = blockIdx.x * TILE_WIDTH + tx;
  int y_o = blockIdx.y * TILE_WIDTH + ty;
  int z_o = blockIdx.z * TILE_WIDTH + tz;

  int x_i = x_o - MASK_RADIUS;
  int y_i = y_o - MASK_RADIUS;
  int z_i = z_o - MASK_RADIUS;

  __shared__ float tile[BLOCK_WIDTH][BLOCK_WIDTH][BLOCK_WIDTH];

  // copy data from global memory to shared memory
  if ((x_i >= 0) && (x_i < x_size) &&
      (y_i >= 0) && (y_i < y_size) &&
      (z_i >= 0) && (z_i < z_size))
  {
      tile[tz][ty][tx] = input[z_i * (y_size * x_size) + y_i * (x_size) + x_i];
  }
  else
  {
      tile[tz][ty][tx] = 0.0f;
  }
  __syncthreads();

  if (tx < TILE_WIDTH && ty < TILE_WIDTH && tz < TILE_WIDTH && x_o < x_size && y_o < y_size && z_o < z_size)
  {
      float Pvalue = 0;
      for (int i = 0; i < MASK_WIDTH; i++)
          for (int j = 0; j < MASK_WIDTH; j++)
              for (int k = 0; k < MASK_WIDTH; k++)
                  Pvalue += M_KDevice[i][j][k] * tile[i + tz][j + ty][k + tx];

      output[z_o * (y_size * x_size) + y_o * (x_size) + x_o] = Pvalue;
  }

}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMalloc((void **)&deviceInput, (inputLength - 3) * sizeof(float));
  cudaMalloc((void **)&deviceOutput, (inputLength - 3) * sizeof(float));

  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, &hostInput[3], (inputLength - 3) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(M_KDevice, hostKernel, MASK_WIDTH * MASK_WIDTH * MASK_WIDTH * sizeof(float));

  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 dimGrid(ceil((x_size) / (TILE_WIDTH*1.0)), ceil((y_size) / (TILE_WIDTH*1.0)), ceil((z_size) / (TILE_WIDTH*1.0)));
  dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, BLOCK_WIDTH);
  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(&hostOutput[3], deviceOutput, (inputLength - 3) * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
