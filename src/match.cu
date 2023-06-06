#include "macro.h"

__device__ float patch_distance(float* a, float* b, int width, int patch_size) {
    float dist = 0.f;
    for (int i = 0; i < patch_size; i++) {
        for (int j = 0; j < patch_size; j++) {
            dist += fabs(a[i * width + j] - b[i * width + j]);
        }
    }
    return dist;
}

__global__ void initialize_nnf(int* nnf, int width, int height, int patch_size, int seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = x * width + y;
    curandState_t state;
    if (x < height - patch_size && y < width - patch_size) {
        curand_init(seed, tid, 0, &state);
        int dx = (int)(curand_uniform(&state) * (2 * MAX_DISPLACEMENT)) - MAX_DISPLACEMENT;
        int dy = (int)(curand_uniform(&state) * (2 * MAX_DISPLACEMENT)) - MAX_DISPLACEMENT;
        nnf[2 * (x * width + y)] = x + dx >= 0 && x + dx < height - patch_size ? x + dx : x;
        nnf[2 * (x * width + y) + 1] = y + dy >= 0 && y + dy < width - patch_size ? y + dy : y;
    }
}

__global__ void apply_nnf(float* dev_a_prime, float* dev_b_prime,
                          int width, int height, int patch_size, int u, const int* nnf) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = x * width + y;

    if (x >= height - patch_size || y >= width - patch_size) {
        return;
    }

    int target_x = nnf[2 * idx];
    int target_y = nnf[2 * idx + 1];
    int target_idx = target_x * width + target_y;

    if (target_x >= height - patch_size || target_y >= width - patch_size) {
        //printf("NNF Overflow! ");
        if (target_x >= height - patch_size)
            printf("target_x ");
        if (target_y >= width - patch_size)
            printf("target_y ");
    }

    // default copy
    dev_a_prime[idx] = dev_b_prime[target_idx];

    // bottom copy
    if (x == height - patch_size - 1) {
        for (int i = 1; i < patch_size; i++) {
            dev_a_prime[idx + width * i] = dev_b_prime[target_idx + width * i];
        }
    }
    if (y == width - patch_size - 1) {
        for (int i = 1; i < patch_size; i++) {
            dev_a_prime[idx + i] = dev_b_prime[target_idx + i];
        }
    }
}

/**
 * The function computes patch distances based on given four images and nnf.
 * $Dist = |A(p) - B(q)|^2 + u*|A'(p) - B'(q)|^2, p and q specified by nnf$
 *
 * @param a input array a
 * @param b input array b
 * @param dev_a_prime stylized input a_prime
 * @param dev_b_prime stylized input b_prime
 * @param width image width, assuming a and b are of the same size
 * @param height image height, assuming a and b are of the same size
 * @param patch_size selected calculation patch size
 * @param u a parameter deciding the effect of the style guidance
 * @param nnf Nearest-Neighbor Field, from a to b
 * @param distances the result to write on
 */
__global__ void compute_patch_distances(float* a, float* b, float* dev_a_prime, float* dev_b_prime,
                                        int width, int height, int patch_size, int u, const int* nnf, float* distances) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // patch overlapped the edge of images
    if (x >= height - patch_size || y >= width - patch_size) {
        return;
    }

    int idx = x * width + y;
    int target_x = nnf[2 * idx];
    int target_y = nnf[2 * idx + 1];
    int target_idx = target_x * width + target_y;

    float a_b_dist, a_b_prime_dist;

    a_b_dist = patch_distance(&a[idx], &b[target_idx], width, patch_size);
    a_b_prime_dist = patch_distance(&dev_a_prime[idx], &dev_b_prime[target_idx], width, patch_size);

    distances[x * width + y] = a_b_dist * a_b_dist + u * a_b_prime_dist * a_b_prime_dist;
}



/**
 *
 * @param a input array a
 * @param b input array b
 * @param width image width, assuming a and b are of the same size
 * @param height image height, assuming a and b are of the same size
 * @param distance based on a, records the current distance
 * @param nnf current Nearest Neighbor Field
 * @param patch_size given patch size
 * @param reversed scanning direction, left to right, up to down if false.
 */
__global__ void propagate(float* a, float* b, int width, int height, float* distance,
                          int* nnf, int patch_size, const bool reversed) {

    //    int bx = blockIdx.x * blockDim.x;
    //    int by = blockIdx.y * blockDim.y;
    //    int tx = threadIdx.x;
    //    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= height - patch_size || y >= width - patch_size) {
        return;
    }

    int idx = x * width + y;
    float dist = distance[idx];
    int best_x = x, best_y = y;
    float tmp_dist;
    // Forward propagation
    if (!reversed) {
        if (y > 0) {
            int b_left_x = nnf[2 * (idx - 1)];
            int b_left_y = nnf[2 * (idx - 1) + 1] + 1;
            if (b_left_y < width - patch_size) {
                tmp_dist = patch_distance(&a[idx], &b[b_left_x * width + b_left_y], width, patch_size);
                if (dist > tmp_dist) {
                    dist = tmp_dist;
                    best_x = b_left_x;
                    best_y = b_left_y;
                }
            }
        }
        if (x > 0) {
            int b_up_x = nnf[2 * (idx - width)] + 1;
            int b_up_y = nnf[2 * (idx - width) + 1];
            if (b_up_x < height - patch_size) {
                tmp_dist = patch_distance(&a[idx], &b[b_up_x * width + b_up_y], width, patch_size);
                if (dist > tmp_dist) {
                    dist = tmp_dist;
                    best_x = b_up_x;
                    best_y = b_up_y;
                }
            }
        }

    }

        // Backward propagation
    else {
        if (y < width - patch_size) {
            int b_right_x = nnf[2 * (idx + 1)];
            int b_right_y = nnf[2 * (idx + 1) + 1] - 1;
            if (b_right_y > 0) {
                tmp_dist = patch_distance(&a[idx], &b[b_right_x * width + b_right_y], width, patch_size);
                if (dist > tmp_dist) {
                    dist = tmp_dist;
                    best_x = b_right_x;
                    best_y = b_right_y;
                }
            }
        }
        if (x < height - patch_size) {
            int b_down_x = nnf[2 * (idx + width)] - 1;
            int b_down_y = nnf[2 * (idx + width) + 1];
            if (b_down_x > 0) {
                tmp_dist = patch_distance(&a[idx], &b[b_down_x * width + b_down_y], width, patch_size);
                if (dist > tmp_dist) {
                    dist = tmp_dist;
                    best_x = b_down_x;
                    best_y = b_down_y;
                }
            }
        }
    }

    distance[idx] = dist;
    nnf[2 * idx] = best_x;
    nnf[2 * idx + 1] = best_y;
}