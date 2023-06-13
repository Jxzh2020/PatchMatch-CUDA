//
// Created by Hao on 2023/6/3.
//

#ifndef PATCHMATCH_PATCH_H
#define PATCHMATCH_PATCH_H
#include <opencv2/opencv.hpp>
#define L_DIVIDER 0
#define M_DIVIDER 30
#define R_DIVIDER 70
#define E_DIVIDER 103


int gen_cpu();
void gen_gpu();
void enum_gpu();
void patchMatch(float* a, float* b, float* a_prime, float* b_prime, int A_width, int A_height, int B_width,
    int B_height, int channels, int patch_size, float* u, int round,
    int num_iterations, int* nnf_from_a);
void apply_frame_pyramids(int i, cv::Mat& A_start_prime, cv::Mat& pre_prime, std::vector<cv::Mat>& B_prime_pyramids, int* nnf);

#endif //PATCHMATCH_PATCH_H
