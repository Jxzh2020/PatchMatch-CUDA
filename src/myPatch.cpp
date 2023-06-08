//
// Created by Hao on 2023/6/3.
//
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "patch.h"

using namespace cv;
using namespace std;

void gen_gpu() {
    // Read images
    Mat A = imread("img/source_fullgi.png");
    Mat B = imread("img/target_fullgi.png");
    Mat A_prime = imread("img/source_style.png");
    Mat B_prime = imread("img/target_fullgi.png");

    // Convert to float arrays
    int width = A.cols;
    int height = A.rows;
    int channels = A.channels();

    float* a = new float[width * height * channels];
    float* b = new float[width * height * channels];
    float* a_prime = new float[width * height * channels];
    float* b_prime = new float[width * height * channels];

    std::vector<cv::Mat> A_single_channel;
    cv::split(A, A_single_channel);


    std::vector<cv::Mat> B_single_channel;
    cv::split(B, B_single_channel);


    std::vector<cv::Mat> A_prime_single_channel;
    cv::split(A_prime, A_prime_single_channel);

    std::vector<cv::Mat> B_prime_single_channel;
    cv::split(B_prime, B_prime_single_channel);


    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < channels; k++) {
                a[(i * width + j) * channels + k] = static_cast<float>(A_single_channel[k].at<uchar>(i, j));
                b[(i * width + j) * channels + k] = static_cast<float>(B_single_channel[k].at<uchar>(i, j));
                a_prime[(i * width + j) * channels + k] = static_cast<float>(A_prime_single_channel[k].at<uchar>(i, j));
            }

        }
    }

    // b_prime := b
    // memcpy(b_prime,b,sizeof(float)*width*height);

    // Run PatchMatch algorithm

    int patch_size = 5;
    int num_iterations = 6;
    int u = 2;

    int* nnf_src_a = new int[2 * width * height];

    patchMatch(b, a, b_prime, a_prime, B.cols, B.rows, A.cols, A.rows, channels, patch_size, u, num_iterations, nnf_src_a);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < channels; k++)
                B_prime_single_channel[k].at<uchar>(i, j) = static_cast<uchar>(b_prime[(i * width + j) * channels + k]);
        }
    }
    cv::merge(B_prime_single_channel, B_prime);


    imwrite("output/A.jpeg", A);
    imwrite("output/B.jpeg", B);
    imwrite("output/A_prime.jpeg", A_prime);
    imwrite("output/B_prime.jpeg", B_prime);
    // Free memory
    delete[] a;
    delete[] b;
    delete[] a_prime;
    delete[] b_prime;
    delete[] nnf_src_a;

}