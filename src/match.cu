#include "macro.h"

__device__ inline float my_vector_distance(float* a, float* b, int channels) {
    auto result = 0.f;
    for (int i = 0; i < channels; i++) {
        result += pow((a[i] - b[i]), 2);
    }
    return sqrt(result);
}

__device__ float patch_distance(float* a, float* b, int width, int channels, int patch_size) {
    float dist = 0.f;
    for (int i = 0; i < patch_size; i++) {
        for (int j = 0; j < patch_size; j++)
            for (int k = 0; k < channels; k++) {
                // naive way of distance
                dist += fabs(a[(i * width + j) * channels + k] - b[(i * width + j) * channels + k]);
            }
//            dist +=my_vector_distance(&a[(i * width + j) * channels], &b[(i * width + j) * channels], patch_size);

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
                          int width, int height, int channels, int patch_size, int u, const int* nnf) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = x * width + y;

    if (x >= height - patch_size || y >= width - patch_size) {
        return;
    }

    int target_x = nnf[2 * idx];
    int target_y = nnf[2 * idx + 1];

    idx *= channels;
    int target_idx = (target_x * width + target_y) * channels;

    if (target_x >= height - patch_size || target_y >= width - patch_size) {
        //printf("NNF Overflow! ");
        if (target_x >= height - patch_size)
            printf("target_x ");
        if (target_y >= width - patch_size)
            printf("target_y ");
    }

    // default copy
    for( int i = 0; i < channels; i++)
        dev_a_prime[idx + i] = dev_b_prime[target_idx + i];

    // bottom copy
    if (x == height - patch_size - 1) {
        for (int i = 1; i < patch_size; i++)
            for (int j = 0; j < channels; j++) {
                dev_a_prime[idx + width * i * channels + j] = dev_b_prime[target_idx + width * i * channels + j];
            }
    }
    if (y == width - patch_size - 1) {
        for (int i = 1; i < patch_size; i++)
            for (int j = 0; j < channels; j++) {
                dev_a_prime[idx +  i * channels + j] = dev_b_prime[target_idx +  i * channels + j];
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
                                        int width, int height, int channels, int patch_size, int u, const int* nnf, float* distances) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // patch overlapped the edge of images
    if (x >= height - patch_size || y >= width - patch_size) {
        return;
    }

    int idx = x * width + y;
    int target_x = nnf[2 * idx];
    int target_y = nnf[2 * idx + 1];

    idx *= channels;
    int target_idx = (target_x * width + target_y) * channels;

    float a_b_dist, a_b_prime_dist;

    a_b_dist = patch_distance(&a[idx], &b[target_idx], width, channels, patch_size);
    a_b_prime_dist = patch_distance(&dev_a_prime[idx], &dev_b_prime[target_idx], width, channels, patch_size);

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
__global__ void propagate(float* a, float* b, int width, int height, int channels, float* distance,
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
                tmp_dist = patch_distance(&a[idx * channels], &b[(b_left_x * width + b_left_y) * channels], width, channels, patch_size);
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
                tmp_dist = patch_distance(&a[idx * channels], &b[(b_up_x * width + b_up_y) * channels], width, channels, patch_size);
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
                tmp_dist = patch_distance(&a[idx * channels], &b[(b_right_x * width + b_right_y) * channels], width, channels, patch_size);
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
                tmp_dist = patch_distance(&a[idx * channels], &b[(b_down_x * width + b_down_y) * channels], width, channels, patch_size);
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
__global__ void random_search(float* a, float* b, float* dev_a_prime, int width, int height,
                              int channels, int patch_size, int u, int* nnf, float* distances) {
    //printf("Launching random_search");

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= height - patch_size || y >= width - patch_size) {
        return ;
    }

    int idx = x * width + y;

    int target_x = nnf[2 * idx];
    int target_y = nnf[2 * idx + 1];

    int offset_x = fabs((float)target_x - x);
    int offset_y = fabs((float)target_y - y);

    int offset_radius = offset_x > offset_y ? offset_x : offset_y;

    float dist = distances[idx];
    float tmp_dist, tmp_dist_prime;

    int tid = x * width + y;
    curandState_t state;
    curand_init(RANDOM_SEED, tid, 0, &state);

    for (int cnt = 0; cnt < offset_radius; cnt++) {
        if (offset_radius < patch_size)
            break;
        int dx = target_x + (int)(curand_uniform(&state) * (2 * offset_radius)) - offset_radius;
        int dy = target_y + (int)(curand_uniform(&state) * (2 * offset_radius)) - offset_radius;
        if (!(dx >= 0 && dx < height - patch_size && dy >= 0 && dy < width - patch_size)) {
            cnt--;
            continue;
        }

        tmp_dist_prime = patch_distance(&dev_a_prime[idx * channels], &dev_a_prime[(dx * width + dy) * channels], width, channels, patch_size);
        tmp_dist = patch_distance(&a[idx * channels], &b[(dx * width + dy) * channels], width, channels, patch_size);
        tmp_dist = u * tmp_dist_prime * tmp_dist_prime + tmp_dist * tmp_dist;
        if (tmp_dist < dist) {
            dist = tmp_dist;
            distances[idx] = dist;
            nnf[2 * idx] = dx;
            nnf[2 * idx + 1] = dy;
            target_x = dx;
            target_y = dy;
            offset_x = (int)fabs((float)dx - x);
            offset_y = (int)fabs((float)dy - y);
            offset_radius = offset_x > offset_y ? offset_x : offset_y;
            cnt = 0;
            return ;
        }
    }
}