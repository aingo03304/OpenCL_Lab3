#define WORK_GROUP_SIZE 128
#define MASK_SIZE 5

__kernel void tiled1DConvolution(__constant float* inputArray, 
                                 __constant float* filterArray, 
                                 __global float* outputArray, 
                                 int inputLength) {
    int index = get_global_id(0);
    int localIndex = get_local_id(0);
    if (index >= inputLength) return;
    __local float shared[WORK_GROUP_SIZE + MASK_SIZE / 2 * 2];
    
    // check if local index is at boundaries.
    if (localIndex - MASK_SIZE / 2 < 0) {
        if (index - MASK_SIZE / 2 >= 0) {
            shared[localIndex] = inputArray[index - MASK_SIZE / 2];
        } else {
            shared[localIndex] = 0.0f;
        }
    } else if (localIndex + MASK_SIZE / 2 >= WORK_GROUP_SIZE) {
        if (index + MASK_SIZE / 2 < inputLength) {
            shared[localIndex + MASK_SIZE / 2 * 2] = inputArray[index + MASK_SIZE / 2];
        } else {
            shared[localIndex + MASK_SIZE / 2 * 2] = 0.0f;
        }
    }
    shared[localIndex + MASK_SIZE / 2] = inputArray[index];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    float outputValue = 0.0f;
    for (int i = 0; i < MASK_SIZE; i++) {
        outputValue += filterArray[i] * shared[localIndex + i];
    }
    outputArray[index] = outputValue;
}