__kernel void adam_map_36(__global float *A,__global float *B) {

        // local vars: Ad
    float Ad;
    // READ
    // WRITTEN
    // READ & WRITTEN
    // globalIdDeclaration
    int i;
    // globalIdInitialisation
        i = get_global_id(0);

    
    Ad = A[i];
    A[i] = 1.5*A[i] - 0.5*B[i];
    B[i] = A[i];
                }
