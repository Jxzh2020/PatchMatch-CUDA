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
    const int pics_num = 4;
    cv::Mat A[pics_num], B[pics_num];

    std::string A_path[4] = { "img/source_fullgi.png", "img/source_indirb.png", "img/source_dirdif.png", "img/source_dirspc.png" };
    std::string B_path[4] = { "img/target_fullgi.png", "img/target_indirb.png", "img/target_dirdif.png", "img/target_dirspc.png" };
    for (int i = 0; i < pics_num; i++) {
        A[i] = imread(A_path[i]);
        B[i] = imread(B_path[i]);
    }
    //// Read images
    //A[0] = imread("img/source_fullgi.png");
    //B[0] = imread("img/target_fullgi.png");


    //A[1] = imread("img/source_indirb.png");
    //B[1] = imread("img/target_indirb.png");


    //A[2] = imread("img/source_dirdif.png");
    //B[2] = imread("img/target_dirdif.png");


    //A[3] = imread("img/source_dirspc.png");
    //B[3] = imread("img/target_dirspc.png");

    Mat A_prime = imread("img/source_style.png");
    Mat B_prime = imread("img/target_fullgi.png");

    // Convert to float arrays
    int A_width = A[0].cols;
    int A_height = A[0].rows;

    int B_width = B[0].cols;
    int B_height = B[0].rows;



    std::vector<cv::Mat> A_channel;
    std::vector<cv::Mat> B_channel;
    std::vector<cv::Mat> A_prime_channel;
    std::vector<cv::Mat> B_prime_channel;

    cv::split(A_prime, A_prime_channel);
    cv::split(B_prime, B_prime_channel);


    int channels = 0;
    std::vector<cv::Mat> temp;

    for (int i = 0; i < pics_num; i++) {
        channels += A[i].channels();
        cv::split(A[i], temp);
        A_channel.insert(A_channel.end(), temp.begin(), temp.end());
        temp.clear();
        cv::split(B[i], temp);
        B_channel.insert(B_channel.end(), temp.begin(), temp.end());
        temp.clear();

    }




    float* a = new float[A_width * A_height * channels];
    float* b = new float[B_width * B_height * channels];
    float* a_prime = new float[A_width * A_height * A_prime.channels()];
    float* b_prime = new float[B_width * B_height * A_prime.channels()];


    for (int i = 0; i < A_height; i++) {
        for (int j = 0; j < A_width; j++) {
            for (int k = 0; k < channels; k++) {
                a[(i * A_width + j) * channels + k] = static_cast<float>(A_channel[k].at<uchar>(i, j));
                if (k >= A_prime.channels())
                    continue;
                a_prime[(i * A_width + j) * A_prime.channels() + k] = static_cast<float>(A_prime_channel[k].at<uchar>(i, j));
            }

        }
    }
    for (int i = 0; i < B_height; i++) {
        for (int j = 0; j < B_width; j++) {
            for (int k = 0; k < channels; k++) {
                b[(i * B_width + j) * channels + k] = static_cast<float>(B_channel[k].at<uchar>(i, j));
            }

        }
    }


    // b_prime := b
    for (int i = 0; i < B_width * B_height * A_prime.channels(); i++)
        b_prime[i] = 255.f;

    // Run PatchMatch algorithm

    int patch_size = 5;
    int num_iterations = 2;
    int u = 1;

    int* nnf_src_a = new int[2 * B_width * B_height];

    patchMatch(b, a, b_prime, a_prime, B_width, B_height, A_width, A_height, channels, patch_size, u, num_iterations, nnf_src_a);

    for (int i = 0; i < B_height; i++) {
        for (int j = 0; j < B_width; j++) {
            for (int k = 0; k < A_prime.channels() ; k++)
                B_prime_channel[k].at<uchar>(i, j) = static_cast<uchar>(b_prime[(i * B_width + j) * A_prime.channels() + k]);
        }
    }
    cv::merge(B_prime_channel, B_prime);

    for (int i = 0; i < pics_num; i++) {
        imwrite("output/A_c" + std::to_string(i) + ".jpeg", A[i]);
        imwrite("output/B_c" + std::to_string(i) + ".jpeg", B[i]);

    }
    imwrite("output/A_prime.jpeg", A_prime);
    imwrite("output/B_prime.jpeg", B_prime);

    // Free memory
    delete[] a;
    delete[] b;
    delete[] a_prime;
    delete[] b_prime;
    delete[] nnf_src_a;

}