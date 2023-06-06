//
// Created by Hao on 2023/6/3.
//
#include <opencv2/opencv.hpp>
#include <iostream>
#include "patch.h"

using namespace cv;
using namespace std;

void gen_gpu() {
    // Read images
    Mat A = imread("../img/source_fullgi.png");
    Mat B = imread("../img/target_fullgi.png");
    Mat A_prime = imread("../img/source_style.png");
    Mat B_prime = imread("../img/target_fullgi.png");

    cv::Mat gray_A, gray_B, gray_A_prime, gray_B_prime;

    cv::cvtColor(A, gray_A, cv::COLOR_BGR2GRAY);
    cv::cvtColor(B, gray_B, cv::COLOR_BGR2GRAY);
    cv::cvtColor(A_prime, gray_A_prime, cv::COLOR_BGR2GRAY);
    cv::cvtColor(B_prime, gray_B_prime, cv::COLOR_BGR2GRAY);

    // Convert to float arrays
    int width = gray_A.cols;
    int height = gray_A.rows;
    
    float* a = new float[width * height];
    float* b = new float[width * height];
    float* a_prime = new float[width * height];
    float* b_prime = new float[width * height];

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            a[i * width + j] = static_cast<float>(gray_A.at<uchar>(i, j));
            b[i * width + j] = static_cast<float>(gray_B.at<uchar>(i, j));
            a_prime[i * width + j] = static_cast<float>(gray_A_prime.at<uchar>(i, j));
        }
    }

    // b_prime := b
    // memcpy(b_prime,b,sizeof(float)*width*height);

    // Run PatchMatch algorithm
    int patch_size = 5;
    int num_iterations = 100;

    int* nnf_src_a = new int[2 * width * height];

    patchMatch(b, a, b_prime, a_prime, width, height, patch_size, 2, num_iterations, nnf_src_a);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            gray_B_prime.at<uchar>(i, j) = (unsigned char)b_prime[i * width + j];//A_prime.at<uchar>((int)nnf_src_a[2*(i * width + j)], (int)nnf_src_a[2*(i * width + j) + 1]);
        }
    }

    imwrite("output/gray_A.jpeg", gray_A);
    imwrite("output/gray_B.jpeg", gray_B);
    imwrite("output/gray_A_prime.jpeg", gray_A_prime);
    imwrite("output/gray_B_prime.jpeg", gray_B_prime);
    // Free memory
    delete[] a;
    delete[] b;
    delete[] a_prime;
    delete[] b_prime;
    delete[] nnf_src_a;

}