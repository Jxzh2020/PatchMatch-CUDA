//
// Created by Hao on 2023/6/3.
//

#ifndef PATCHMATCH_PATCH_H
#define PATCHMATCH_PATCH_H

int gen_cpu();
void gen_gpu();
void enum_gpu();
void patchMatch(float* a, float* b, float* a_prime, float* b_prime, int width, int height, int patch_size, int u,
                int num_iterations, int* nnf_from_a);

#endif //PATCHMATCH_PATCH_H
