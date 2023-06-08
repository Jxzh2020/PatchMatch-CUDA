#include "macro.h"

//__device__ inline float my_vector_distance(float* a, float* b, int channels) {
//    auto result = 0.f;
//    for (int i = 0; i < channels; i++) {
//        result += pow((a[i] - b[i]), 2);
//    }
//    return sqrt(result);
//}

__device__ void cal_range(int x, int y, int target_x, int target_y , int A_width, int A_height, int B_width, int B_height, int patch_size,
                          int* x_min, int* x_max, int *y_min, int* y_max) {

    int half_patch = patch_size >> 1;
    int d_x_max, d_y_max;

    *x_min = half_patch > x ? x > target_x ? target_x : x : half_patch;
    *y_min = half_patch > y ? y > target_y ? target_y : y : half_patch;

    d_x_max = half_patch > (A_height - x - 1) ? (A_height - x - 1) > (B_height - target_x - 1) ? (B_height - target_x - 1) : (A_height - x - 1) : half_patch;
    d_y_max = half_patch > (A_width - y - 1) ? (A_width - y - 1) > (B_width - target_y - 1) ? (B_width - target_y - 1) : (A_width - y - 1) : half_patch;

    *x_max = d_x_max + 1;
    *y_max = d_y_max + 1;
}
/**
* channels here means the channel of b_prime
*
*
*/
__device__ void average_patch(float* b_prime, int B_width, int channels, int patch_height, int patch_width, float* rgb) {
    int size = patch_height * patch_width;

    // TODO: Note changable channels here
    rgb[0] = 0.f;
    rgb[1] = 0.f;
    rgb[2] = 0.f;

    for (int i = 0; i < patch_height; i++) {
        for (int j = 0; j < patch_width; j++)
            for (int k = 0; k < channels; k++) {
                // naive way of distance
                rgb[k] += b_prime[(i * B_width + j) * channels + k];
            }
        /*dist +=my_vector_distance(&a[(i * width + j) * channels], &b[(i * width + j) * channels], patch_size);*/

    }
    rgb[0] /= size;
    rgb[1] /= size;
    rgb[2] /= size;
}

__device__ float patch_distance(float* a, float* b, int A_width, int B_width, int channels, int patch_height, int patch_width) {
    float dist = 0.f;
    for (int i = 0; i < patch_height; i++) {
        for (int j = 0; j < patch_width; j++)
            for (int k = 0; k < channels; k++) {
                // naive way of distance
                dist += fabs(a[(i * A_width + j) * channels + k] - b[(i * B_width + j) * channels + k]);
            }
        /*dist +=my_vector_distance(&a[(i * width + j) * channels], &b[(i * width + j) * channels], patch_size);*/

    }
    return dist / (float)(patch_height * patch_width * channels);
}

__global__ void initialize_nnf(int* nnf, int A_width, int A_height, int B_width, int B_height, int patch_size, int seed) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = x * A_width + y;
    curandState_t state;

    //TODO: Random radius points based on A's size, it should be based on B's
    if (x < A_height && y < A_width ) {
        curand_init(seed, tid, 0, &state);
        int dx = (int)(curand_uniform(&state) * (2 * MAX_DISPLACEMENT)) - MAX_DISPLACEMENT;
        int dy = (int)(curand_uniform(&state) * (2 * MAX_DISPLACEMENT)) - MAX_DISPLACEMENT;

        nnf[2 * (x * A_width + y)] = x + dx >= 0 && x + dx < B_height ? x + dx : x;
        nnf[2 * (x * A_width + y) + 1] = y + dy >= 0 && y + dy < B_width ? y + dy : y;
    }
}

__global__ void apply_nnf(float* dev_a_prime, float* dev_b_prime,
                          int A_width, int A_height, int B_width, int B_height, int channels, int patch_size, int u, int iteration, const int* nnf) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = x * A_width + y;

    if (x >= A_height || y >= A_width ) {
        return;
    }


    int target_x = nnf[2 * idx];
    int target_y = nnf[2 * idx + 1];

    idx *= channels;
    int target_idx = (target_x * B_width + target_y) * channels;

    //if (target_x >= height - patch_size || target_y >= width - patch_size) {
    //    //printf("NNF Overflow! ");
    //    if (target_x >= height - patch_size)
    //        printf("target_x ");
    //    if (target_y >= width - patch_size)
    //        printf("target_y ");
    //}
    int half_patch = patch_size / 2;
    int x_min = half_patch > target_x ? target_x : half_patch;
    int y_min = half_patch > target_y ? target_y : half_patch;

    int x_max = half_patch > (A_height - target_x - 1) ? (A_height - target_x - 1) : half_patch;
    int y_max = half_patch > (A_width - target_y - 1) ? (A_width - target_y - 1) : half_patch;

    ++x_max;
    ++y_max;

    // TODO: the length is actually x_min + x_max + 1
    float rgb[3];
    average_patch(&dev_b_prime[target_idx - x_min * B_width * PRIME_CHANNELS - y_min * PRIME_CHANNELS], B_width, PRIME_CHANNELS, x_min + x_max, y_min + y_max, rgb);
    if (iteration == 0) {
        for (int i = 0; i < PRIME_CHANNELS; i++)
            dev_a_prime[x * A_width * PRIME_CHANNELS + y * PRIME_CHANNELS + i] = rgb[i];
    }
    else {
        for (int i = 0; i < PRIME_CHANNELS; i++)
            dev_a_prime[x * A_width * PRIME_CHANNELS + y * PRIME_CHANNELS + i] = (dev_a_prime[x * A_width * PRIME_CHANNELS + y * PRIME_CHANNELS + i] + rgb[i]) / 2;
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
                                        int A_width, int A_height, int B_width, int B_height, int channels, int patch_size, int u, const int* nnf, float* distances) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // patch overlapped the edge of images
    if (x >= A_height || y >= A_width ) {
        return;
    }

    int idx = x * A_width + y;
    int target_x = nnf[2 * idx];
    int target_y = nnf[2 * idx + 1];

    int target_idx = (target_x * B_width + target_y);

    int x_min, y_min, x_max, y_max;

    cal_range(x, y, target_x, target_y, A_width, A_height, B_width, B_height, patch_size, &x_min, &x_max, &y_min, &y_max);

    float a_b_dist, a_b_prime_dist;

    a_b_dist = patch_distance(&a[(idx - x_min*A_width - y_min)*channels], &b[(target_idx - x_min*B_width-y_min)*channels], A_width, B_width, channels, x_min + x_max,y_min + y_max);
    a_b_prime_dist = patch_distance(&dev_a_prime[(idx - x_min * A_width - y_min) * PRIME_CHANNELS], &dev_b_prime[(target_idx - x_min * B_width - y_min) * PRIME_CHANNELS], A_width, B_width, PRIME_CHANNELS, x_min + x_max, y_min + y_max);

    distances[x * A_width + y] = u * a_b_dist * a_b_dist + a_b_prime_dist * a_b_prime_dist;
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
__global__ void propagate(float* a, float* b, float* dev_a_prime, float* dev_b_prime, int A_width, int A_height, int B_width, int B_height, int channels, float* distance,
                          int* nnf, int u, int patch_size, const bool reversed) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= A_height|| y >= A_width) {
        return;
    }

    int idx = x * A_width + y;
    float dist = distance[idx];

    int best_x = nnf[2 * idx], best_y = nnf[2 * idx + 1];
    float tmp_dist, tmp_prime_dist;
    int x_min, y_min, x_max, y_max;
    int target_idx;

    // Forward propagation
    if (!reversed) {
        if (y > 0) {
            int b_left_x = nnf[2 * (idx - 1)];
            int b_left_y = nnf[2 * (idx - 1) + 1] + 1;
            if (b_left_y < B_width) {
                target_idx = b_left_x * B_width + b_left_y;
                cal_range(x, y, b_left_x, b_left_y, A_width, A_height, B_width, B_height, patch_size, &x_min, &x_max, &y_min, &y_max);
                tmp_dist = patch_distance(&a[(idx - x_min * A_width - y_min) * channels], &b[(target_idx - x_min * B_width - y_min) * channels], A_width, B_width, channels, x_min + x_max, y_min + y_max);
                tmp_prime_dist = patch_distance(&dev_a_prime[(idx - x_min * A_width - y_min) * PRIME_CHANNELS], &dev_b_prime[(target_idx - x_min * B_width - y_min) * PRIME_CHANNELS - y_min * PRIME_CHANNELS], A_width, B_width, PRIME_CHANNELS, x_min + x_max, y_min + y_max);
                tmp_dist = u * tmp_dist * tmp_dist + tmp_prime_dist * tmp_prime_dist;
                if (dist > tmp_dist) {
                    dist = tmp_dist;
                    best_x = b_left_x;
                    best_y = b_left_y;
                }
            }
        }
        if (x > 0) {
            int b_up_x = nnf[2 * (idx - A_width)] + 1;
            int b_up_y = nnf[2 * (idx - A_width) + 1];
            if (b_up_x < B_height) {
                target_idx = b_up_x * B_width + b_up_y;
                cal_range(x, y, b_up_x, b_up_y, A_width, A_height, B_width, B_height, patch_size, &x_min, &x_max, &y_min, &y_max);
                tmp_dist = patch_distance(&a[(idx - x_min * A_width - y_min) * channels], &b[(target_idx - x_min * B_width - y_min) * channels], A_width, B_width, channels, x_min + x_max, y_min + y_max);
                tmp_prime_dist = patch_distance(&dev_a_prime[(idx - x_min * A_width - y_min) * PRIME_CHANNELS], &dev_b_prime[(target_idx - x_min * B_width - y_min) * PRIME_CHANNELS - y_min * PRIME_CHANNELS], A_width, B_width, PRIME_CHANNELS, x_min + x_max, y_min + y_max);
                tmp_dist = u * tmp_dist * tmp_dist + tmp_prime_dist * tmp_prime_dist;
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
        if (y < A_width - 1) {
            int b_right_x = nnf[2 * (idx + 1)];
            int b_right_y = nnf[2 * (idx + 1) + 1] - 1;
            if (b_right_y >= 0) {
                target_idx = b_right_x * B_width + b_right_y;
                cal_range(x, y, b_right_x, b_right_y, A_width, A_height, B_width, B_height, patch_size, &x_min, &x_max, &y_min, &y_max);
                tmp_dist = patch_distance(&a[(idx - x_min * A_width - y_min) * channels], &b[(target_idx - x_min * B_width - y_min) * channels], A_width, B_width, channels, x_min + x_max, y_min + y_max);
                tmp_prime_dist = patch_distance(&dev_a_prime[(idx - x_min * A_width - y_min) * PRIME_CHANNELS], &dev_b_prime[(target_idx - x_min * B_width - y_min) * PRIME_CHANNELS - y_min * PRIME_CHANNELS], A_width, B_width, PRIME_CHANNELS, x_min + x_max, y_min + y_max);
                tmp_dist = u * tmp_dist * tmp_dist + tmp_prime_dist * tmp_prime_dist;
                if (dist > tmp_dist) {
                    dist = tmp_dist;
                    best_x = b_right_x;
                    best_y = b_right_y;
                }
            }
        }
        if (x < A_height - 1) {
            int b_down_x = nnf[2 * (idx + A_width)] - 1;
            int b_down_y = nnf[2 * (idx + A_width) + 1];
            if (b_down_x >= 0) {
                target_idx = b_down_x * B_width + b_down_y;
                cal_range(x, y, b_down_x, b_down_y, A_width, A_height, B_width, B_height, patch_size, &x_min, &x_max, &y_min, &y_max);
                tmp_dist = patch_distance(&a[(idx - x_min * A_width - y_min) * channels], &b[(target_idx - x_min * B_width - y_min) * channels], A_width, B_width, channels, x_min + x_max, y_min + y_max);
                tmp_prime_dist = patch_distance(&dev_a_prime[(idx - x_min * A_width - y_min) * PRIME_CHANNELS], &dev_b_prime[(target_idx - x_min * B_width - y_min) * PRIME_CHANNELS - y_min * PRIME_CHANNELS], A_width, B_width, PRIME_CHANNELS, x_min + x_max, y_min + y_max);
                tmp_dist = u * tmp_dist * tmp_dist + tmp_prime_dist * tmp_prime_dist;
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


//__global__ void random_search(float* a, float* b, float* dev_a_prime, int width, int height,
//    int channels, int patch_size, int u, int* nnf, float* distances) {
//    //printf("Launching random_search");
//
//    int x = blockIdx.x * blockDim.x + threadIdx.x;
//    int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//    if (x >= height - patch_size || y >= width - patch_size) {
//        return ;
//    }
//
//    int idx = x * width + y;
//
//    int target_x = nnf[2 * idx];
//    int target_y = nnf[2 * idx + 1];
//
//    int offset_x = fabs((float)target_x - x);
//    int offset_y = fabs((float)target_y - y);
//
//    int offset_radius = offset_x > offset_y ? offset_x : offset_y;
//
//    float dist = distances[idx];
//    float tmp_dist, tmp_dist_prime;
//
//    int tid = x * width + y;
//    curandState_t state;
//    curand_init(RANDOM_SEED, tid, 0, &state);
//
//    for (int cnt = 0; cnt < offset_radius; cnt++) {
//        if (offset_radius < patch_size)
//            break;
//        int dx = target_x + (int)(curand_uniform(&state) * (2 * offset_radius)) - offset_radius;
//        int dy = target_y + (int)(curand_uniform(&state) * (2 * offset_radius)) - offset_radius;
//        if (!(dx >= 0 && dx < height - patch_size && dy >= 0 && dy < width - patch_size)) {
//            cnt--;
//            continue;
//        }
//
//        tmp_dist_prime = patch_distance(&dev_a_prime[idx * channels], &dev_a_prime[(dx * width + dy) * channels], width, channels, patch_size);
//        tmp_dist = patch_distance(&a[idx * channels], &b[(dx * width + dy) * channels], width, channels, patch_size);
//        tmp_dist = tmp_dist_prime * tmp_dist_prime + u * tmp_dist * tmp_dist;
//        if (tmp_dist < dist) {
//            dist = tmp_dist;
//            distances[idx] = dist;
//            nnf[2 * idx] = dx;
//            nnf[2 * idx + 1] = dy;
//            target_x = dx;
//            target_y = dy;
//            offset_x = (int)fabs((float)dx - x);
//            offset_y = (int)fabs((float)dy - y);
//            offset_radius = offset_x > offset_y ? offset_x : offset_y;
//            cnt = 0;
//            return ;
//        }
//    }
//}