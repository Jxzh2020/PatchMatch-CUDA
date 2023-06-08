//
// Created by Hao on 2023/6/3.
//

#ifndef PATCHMATCH_MACRO_H
#define PATCHMATCH_MACRO_H

#define PRIME_CHANNELS 3
#define BLOCK_SIZE 16
#define SUB_BLOCK_SIZE 16
#define MAX_DISPLACEMENT 512
#define RANDOM_CNT 5
#define RANDOM_SEED 5206

#include <cstdio>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}

__global__ void random_search(float* a, float* b, float* dev_a_prime,
                              int width, int height, int channels, int patch_size, int u, int* nnf, float* distances);

__global__ void apply_nnf(float* dev_a_prime, float* dev_b_prime,
                          int A_width, int A_height, int B_width, int B_height,
                          int channels, int patch_size, int u, int iteration, const int* nnf);
__global__ void compute_patch_distances(float* a, float* b,
                                        float* dev_a_prime, float* dev_b_prime,
                                        int A_width, int A_height, int B_width, int B_height,
                                        int channels, int patch_size, int u, const int* nnf, float* distances);
__global__ void initialize_nnf(int* nnf, int A_width, int A_height, int B_width, int B_height,
                               int patch_size, int seed);
__global__ void propagate(float* a, float* b, float* dev_a_prime,
                          float* dev_b_prime, int A_width, int A_height, int B_width,
                          int B_height, int channels, float* distance,
                          int* nnf, int u, int patch_size, bool reversed);

#endif //PATCHMATCH_MACRO_H
