#include <CL/opencl.h>
#include <gputk.h>

#define checkStatus(status) \
  if (status != CL_SUCCESS) { \
    std::cout << "[" << __FILE__ << ":" << __LINE__ << "]" << " OpenCL error " << status << std::endl; \
    exit(status); \
  }

cl_int status;

cl_program createProgramFromFile(cl_context context, const char *file_name) {
  FILE *file = fopen(file_name, "rb");
  if(file == NULL) {
    std::cout << "Failed to open the kernel file." << std::endl;
    exit(1);
  }
  fseek(file, 0, SEEK_END);
  size_t sourceSize = ftell(file);
  rewind(file);
  char *sourceCode = (char *)malloc(sourceSize + 1);
  fread(sourceCode, sizeof(char), sourceSize, file);
  sourceCode[sourceSize] = '\0';
  fclose(file);
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&sourceCode, &sourceSize, &status);
  checkStatus(status);
  free(sourceCode);
  return program;
}

const char *kernelSource = "";

int main(int argc, char *argv[]) {
  gpuTKArg_t args;
  int inputLength;
  int filterLength;
  int inputLengthBytes;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  cl_mem deviceInput1;
  cl_mem deviceInput2;
  cl_mem deviceOutput;

  cl_platform_id cpPlatform; // OpenCL platform
  cl_device_id device_id;    // device ID
  cl_context context;        // context
  cl_command_queue queue;    // command queue
  cl_program program;        // program
  cl_kernel kernel;          // kernel

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 =
      (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &filterLength);
  inputLengthBytes = inputLength * sizeof(float);
  hostOutput       = (float *)malloc(inputLengthBytes);
  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  gpuTKLog(TRACE, "The input length is ", inputLength);
  gpuTKLog(TRACE, "The filter length is ", filterLength);
  gpuTKLog(TRACE, "The input size is ", inputLengthBytes, " bytes");

  //@@ Insert code here
  //@@ Initialize the workgroup dimensions
  cl_uint numPlatforms = 0;
  cl_uint numDevices = 0;
  std::cout << "Max work group size: " << CL_DEVICE_MAX_WORK_GROUP_SIZE << std::endl;
  size_t indexSpaceSize[1] = {1024};
  size_t workGroupSize[1] = {128};
  indexSpaceSize[0] = (inputLength + workGroupSize[0] - 1) / workGroupSize[0] * workGroupSize[0];
  std::cout << "Global: " << indexSpaceSize[0] << std::endl;
  std::cout << "Local: " << workGroupSize[0] << std::endl;

  //@@ Bind to platform
  checkStatus(clGetPlatformIDs(1, &cpPlatform, &numPlatforms));
  std::cout << "Number of platforms: " << numPlatforms << std::endl;

  //@@ Get ID for the device
  checkStatus(clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices));
  std::cout << "Number of Devices: " << numDevices << std::endl;
  checkStatus(clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_ALL, 1, &device_id, NULL));

  //@@ Create a context
  context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &status);
  checkStatus(status);

  //@@ Create a command queue
  queue = clCreateCommandQueue(context, device_id, 0, &status);
  checkStatus(status);

  //@@ Create the compute program from the source buffer
  program = createProgramFromFile(context, "kernel.cl");
  checkStatus(status);

  //@@ Build the program executable
  checkStatus(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  //@@ Create the compute kernel in the program we wish to run
  kernel = clCreateKernel(program, "tiled1DConvolution", &status);
  checkStatus(status);

  //@@ Create the input and output arrays in device memory for our
  //@@ calculation
  deviceInput1 = clCreateBuffer(context, CL_MEM_READ_ONLY, inputLengthBytes, NULL, &status);
  checkStatus(status);
  deviceInput2 = clCreateBuffer(context, CL_MEM_READ_ONLY, filterLength * sizeof(float), NULL, &status);
  checkStatus(status);
  deviceOutput = clCreateBuffer(context, CL_MEM_WRITE_ONLY, inputLengthBytes, NULL, &status);
  checkStatus(status);

  //@@ Write our data set into the input array in device memory
  checkStatus(clEnqueueWriteBuffer(queue, deviceInput1, CL_TRUE, 0, inputLengthBytes, hostInput1, 0, NULL, NULL));
  checkStatus(clEnqueueWriteBuffer(queue, deviceInput2, CL_TRUE, 0, filterLength * sizeof(float), hostInput2, 0, NULL, NULL));

  //@@ Set the arguments to our compute kernel
  checkStatus(clSetKernelArg(kernel, 0, sizeof(cl_mem), &deviceInput1));
  checkStatus(clSetKernelArg(kernel, 1, sizeof(cl_mem), &deviceInput2));
  checkStatus(clSetKernelArg(kernel, 2, sizeof(cl_mem), &deviceOutput));
  checkStatus(clSetKernelArg(kernel, 3, sizeof(int), &inputLength));

  //@@ Execute the kernel over the entire range of the data set
  checkStatus(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, indexSpaceSize, workGroupSize, 0, NULL, NULL));
  //@@ Wait for the command queue to get serviced before reading back results
  checkStatus(clFinish(queue));

  //@@ Read the results from the device
  checkStatus(clEnqueueReadBuffer(queue, deviceOutput, CL_TRUE, 0, inputLengthBytes, hostOutput, 0, NULL, NULL));

  gpuTKSolution(args, hostOutput, inputLength);

  // release OpenCL resources
  clReleaseMemObject(deviceInput1);
  clReleaseMemObject(deviceInput2);
  clReleaseMemObject(deviceOutput);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  // release host memory
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
