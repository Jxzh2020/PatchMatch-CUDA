#include "macro.h"

/**
 * The function finds a nnf from a to b, it's not guaranteed that all pixels in b has a match in a.
 * a, b, b_prime are all known
 *
 * @param a
 * @param b
 * @param a_prime
 * @param b_prime
 * @param width
 * @param height
 * @param patch_size
 * @param u
 * @param num_iterations
 * @param nnf_from_a
 */
void patchMatch(float* a, float* b, float* a_prime, float* b_prime, int A_width, int A_height, int B_width,
                int B_height, int channels, int patch_size, int u,
                int num_iterations, int* nnf_from_a)
{
    // Allocate device memory
    float* dev_a;
    float* dev_b;
    float* dev_a_prime;
    float* dev_b_prime;

    cudaMalloc(&dev_a, A_width * A_height * channels * sizeof(float));
    cudaMalloc(&dev_b, B_width * B_height * channels * sizeof(float));
    cudaMalloc(&dev_a_prime, A_width * A_height * PRIME_CHANNELS * sizeof(float));
    cudaMalloc(&dev_b_prime, B_width * B_height * PRIME_CHANNELS * sizeof(float));

    cudaMemcpy(dev_a, a, A_width * A_height * channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, B_width * B_height * channels * sizeof(float), cudaMemcpyHostToDevice);

    // An initial value of A' is the same as A, meaning no style transfer
    cudaMemcpy(dev_a_prime, a_prime, A_width * A_height * PRIME_CHANNELS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b_prime, b_prime, B_width * B_height * PRIME_CHANNELS * sizeof(float), cudaMemcpyHostToDevice);

    int* dev_forward_nnf;
    cudaMalloc(&dev_forward_nnf, 2 * A_width * A_height * sizeof(int));

    /*initialize_nnf << <dim3((width - 1) / BLOCK_SIZE + 1, (height - 1) / BLOCK_SIZE + 1), dim3(BLOCK_SIZE, BLOCK_SIZE) >> > (
        dev_forward_nnf, width, height, patch_size, 5206);
        int* dev_backward_nnf;
        cudaMalloc(&dev_backward_nnf, 2 * width * height * sizeof(int));*/

    initialize_nnf<<<dim3((A_height - 1) / BLOCK_SIZE + 1, (A_width - 1) / BLOCK_SIZE + 1), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(
            dev_forward_nnf, A_width, A_height, B_width, B_height, patch_size, 5206);

    float* dev_distances;
    cudaMalloc(&dev_distances, A_width * A_height * sizeof(float));



    // Main loop
    for (int i = 0; i < num_iterations; i++) {

        compute_patch_distances << <dim3((A_height - 1) / BLOCK_SIZE + 1, (A_width - 1) / BLOCK_SIZE + 1), dim3(BLOCK_SIZE, BLOCK_SIZE) >> > (
                dev_a, dev_b, dev_a_prime, dev_b_prime, A_width, A_height, B_width, B_height, channels,
                        patch_size, i == 0 ? 10000 : u, dev_forward_nnf, dev_distances);
        for (int j = 0; j < 6; j++) {

            //cudaDeviceSynchronize();

            propagate <<<dim3((A_height - 1) / BLOCK_SIZE + 1, (A_width - 1) / BLOCK_SIZE + 1), dim3(BLOCK_SIZE, BLOCK_SIZE)>>> (
                    dev_a, dev_b, dev_a_prime, dev_b_prime, A_width, A_height, B_width, B_height, channels, dev_distances, dev_forward_nnf, i == 0 ? 10000 : u, patch_size, i % 2 == 1);

            random_search << <dim3((A_height - 1) / BLOCK_SIZE + 1, (A_width - 1) / BLOCK_SIZE + 1), dim3(BLOCK_SIZE, BLOCK_SIZE) >> > (
                    dev_a, dev_b, dev_a_prime, dev_b_prime, A_width, A_height, B_width, B_height, channels,
                            patch_size, i == 0 ? 10000 : u, dev_forward_nnf, dev_distances);

            //apply_nnf << <dim3((A_height - 1) / BLOCK_SIZE + 1, (A_width - 1) / BLOCK_SIZE + 1), dim3(BLOCK_SIZE, BLOCK_SIZE) >> > (
            //    dev_a_prime, dev_b_prime, A_width, A_height, B_width, B_height, channels, patch_size, u, 1, dev_forward_nnf);
            //if (j % 2 == 0) {
            //    random_search << <dim3((A_height - 1) / BLOCK_SIZE + 1, (A_width - 1) / BLOCK_SIZE + 1), dim3(BLOCK_SIZE, BLOCK_SIZE) >> > (
            //        dev_a, dev_b, dev_a_prime, dev_b_prime, A_width, A_height, B_width, B_height, channels,
            //        patch_size, u, dev_forward_nnf, dev_distances);
            //    apply_nnf << <dim3((A_height - 1) / BLOCK_SIZE + 1, (A_width - 1) / BLOCK_SIZE + 1), dim3(BLOCK_SIZE, BLOCK_SIZE) >> > (
            //        dev_a_prime, dev_b_prime, A_width, A_height, B_width, B_height, channels, patch_size, u, 1, dev_forward_nnf);
            //}

        }
        apply_nnf << <dim3((A_height - 1) / BLOCK_SIZE + 1, (A_width - 1) / BLOCK_SIZE + 1), dim3(BLOCK_SIZE, BLOCK_SIZE) >> > (
                dev_a_prime, dev_b_prime, A_width, A_height, B_width, B_height, channels, patch_size, u, i, dev_forward_nnf);
        //random_search << <dim3((A_height - 1) / BLOCK_SIZE + 1, (A_width - 1) / BLOCK_SIZE + 1), dim3(BLOCK_SIZE, BLOCK_SIZE) >> > (
        //    dev_a, dev_b, dev_a_prime, dev_b_prime, A_width, A_height, B_width, B_height, channels,
        //    patch_size, u, dev_forward_nnf, dev_distances);
        HANDLE_ERROR(cudaGetLastError());

    }

    /*apply_nnf <<<dim3((A_height - 1) / BLOCK_SIZE + 1, (A_width - 1) / BLOCK_SIZE + 1), dim3(BLOCK_SIZE, BLOCK_SIZE)>>> (
        dev_a_prime, dev_b_prime, A_width, A_height, B_width, B_height, channels, patch_size, u, 1, dev_forward_nnf);*/
    HANDLE_ERROR(cudaGetLastError());




    // Copy result back to host
    cudaMemcpy(nnf_from_a, dev_forward_nnf, 2 * A_width * A_height * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(a_prime, dev_a_prime, A_width * A_height * PRIME_CHANNELS * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_a_prime);
    cudaFree(dev_b_prime);
    cudaFree(dev_forward_nnf);
    //    cudaFree(dev_backward_nnf);
    cudaFree(dev_distances);

}
