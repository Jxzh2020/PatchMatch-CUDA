//
// Created by Hao on 2023/6/3.
//
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include "patch.h"

using namespace cv;
using namespace std;

static const int pics_num = 5;
static const int gaussian_levels = 3;
static float u[5] = { 6, 16, 2, 0, 1 }; // col, edge, pos, temp, mask 3 16 0 0 1 / 4 16 1 0 1 . best ever { 6, 16, 2, 0, 1 };

void gaussian_pyramid(const cv::Mat& img, int levels, std::vector<cv::Mat>& pyramid)
{
    // 将多通道图像拆分成四个单通道图像
    std::vector<cv::Mat> channels;
    int channel = img.channels();
    cv::split(img, channels);

    pyramid.clear(); // 清空金字塔向量

    // 分别对每个通道计算高斯金字塔
    for (int i = 0; i < levels; i++) {
        cv::Mat merged_image; // 保存合并后的图像

        // 遍历四个通道，合并同一层级的图像
        for (int k = 0; k < channel; k++) {
            cv::Mat current = channels[k].clone();
            for (int j = 0; j < i; j++) {
                cv::pyrDown(current, current);
            }
            // 调整图像大小与第一个通道的图像相匹配
            if (k == 0) {
                merged_image = current.clone();
            }
            else {
                cv::resize(current, current, merged_image.size());
                cv::merge(std::vector<cv::Mat>{merged_image, current}, merged_image);
            }
        }

        pyramid.push_back(merged_image.clone()); // 将合并后的图像存储到金字塔中
    }
}

/// <summary>
/// This function up-scale the NNF, according cur_b_prime's size
/// </summary>
/// <param name="nnf"></param> input nnf
/// <param name="cur_b_prime"></param> input Mat this NNF is now based on
int* pyrUp_NNF(int* nnf, const cv::Mat& cur_b_prime, int n_width, int n_height) {
    assert(nnf && "Neareast Neighbor Field input array is nullptr");
    int width = cur_b_prime.cols;
    int height = cur_b_prime.rows;
    int n_i, n_j, n_idx, t_i, t_j;
    int* n_nnf = new int[2 * n_width * n_height];

    for( int i = 0; i < height; i++)
        for (int j = 0; j < width; j++) {

            //*********
            n_i = 2 * (i + 1) - 1;
            n_j = 2 * (j + 1) - 1;
            if (n_i > n_height || n_j > n_width) {
                printf("Severe Error!!\n");
            }
            n_idx = n_i * n_width + n_j;

            t_i = nnf[2 * (i * width + j)];
            t_j = nnf[2 * (i * width + j) + 1];

            t_i = 2 * (t_i + 1) - 1;
            t_j = 2 * (t_j + 1) - 1;



            n_nnf[2 * (n_idx - n_width - 1)] = t_i - 1;
            n_nnf[2 * (n_idx - n_width - 1) + 1] = t_j - 1;

            if (n_i >= n_height || n_j >= n_width) {

                if (t_i >= n_height || t_j >= n_width) {
                    printf("Severe Error!!\n");
                    exit(0);
                }

                if (n_i < n_height) {
                    n_nnf[2 * (n_idx - 1)] = t_i;
                    n_nnf[2 * (n_idx - 1) + 1] = t_j - 1;
                }
                if (n_j < n_width) {
                    n_nnf[2 * (n_idx - n_width)] = t_i - 1;
                    n_nnf[2 * (n_idx - n_width) + 1] = t_j;
                }
            }
            else {
                //
                //    1  2
                //    3  #
                //
                if (t_i >= n_height || t_j >= n_width) {
                    printf("Severe Error!!\n");
                    exit(0);
                }
                n_nnf[2 * (n_idx)] = t_i;
                n_nnf[2 * (n_idx)+1] = t_j;


                n_nnf[2 * (n_idx - n_width)] = t_i - 1;
                n_nnf[2 * (n_idx - n_width) + 1] = t_j;

                n_nnf[2 * (n_idx - 1)] = t_i;
                n_nnf[2 * (n_idx - 1) + 1] = t_j - 1;
            }
        }
    delete[] nnf;
    return n_nnf;
}

// do one complete patch match on scaled img
int* pyramid_pass(const cv::Mat& A, const cv::Mat& B, const cv::Mat& A_prime, cv::Mat& B_prime, int* pre_nnf, int round, bool GenFirst) {

    int A_width = A.cols;
    int A_height = A.rows;

    int A_prime_width = A_prime.cols;
    int A_prime_height = A_prime.rows;


    if (A_width != A_prime_width || A_height != A_prime_height) {
        printf("Pyramids Error!\n");
        exit(1);
    }

    int B_width = B.cols;
    int B_height = B.rows;


    std::vector<cv::Mat> A_channel;
    std::vector<cv::Mat> B_channel;
    std::vector<cv::Mat> A_prime_channel;
    std::vector<cv::Mat> B_prime_channel;

    cv::split(A, A_channel);
    cv::split(B, B_channel);

    cv::split(A_prime, A_prime_channel);
    cv::split(B_prime, B_prime_channel);


    int channels = A.channels();


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
                if (k >= A_prime.channels())
                    continue;
                b_prime[(i * B_width + j) * B_prime.channels() + k] = static_cast<float>(B_prime_channel[k].at<uchar>(i, j));
            }

        }
    }


    // b_prime := b
    if(GenFirst)
        for (int i = 0; i < B_width * B_height * A_prime.channels(); i++)
            b_prime[i] = 255.f;

    // Run PatchMatch algorithm

    int patch_size = 5;
    int num_iterations = 6;

    int* nnf_src_a;
    if (!pre_nnf)
        nnf_src_a = new int[2 * B_width * B_height];
    else
        nnf_src_a = pre_nnf;
    nnf_src_a[0] = !pre_nnf ? -1 : nnf_src_a[0];

    float u_temp[5];
    for (int i = 0; i < pics_num; i++) {
        u_temp[i] = u[i] * pow(16 * round, round);
    }

    patchMatch(b, a, b_prime, a_prime, B_width, B_height, A_width, A_height, channels, patch_size, u_temp, round,num_iterations, nnf_src_a);


    for (int i = 0; i < B_height; i++) {
        for (int j = 0; j < B_width; j++) {
            for (int k = 0; k < A_prime.channels(); k++)
                B_prime_channel[k].at<uchar>(i, j) = static_cast<uchar>(b_prime[(i * B_width + j) * A_prime.channels() + k]);
        }
    }
    cv::merge(B_prime_channel, B_prime);

    delete[] a;
    delete[] b;
    delete[] a_prime;
    delete[] b_prime;
    return nnf_src_a;
}


void gen_gpu() {

    const std::string src_name = "multi_frame_src/video/";
    std::string index[4] = { "0", "30", "70", "103" };
    int key_cnt = 140;
    int* nnf = nullptr;

    cv::Mat A_merge[4], A_prime_merge[4];
    std::vector<cv::Mat> A_frame_pyramids[4], A_prime_pyramids[4];

    for (int i = 0; i < 4; i++) {


        std::string A_start_path[5] = { src_name + index[i] + ".png",
                                        "multi_frame_src/edge/edge" + index[i] + ".jpg",
                                        "multi_frame_src/pos/pos" + index[i] + ".jpg",
                                        "multi_frame_src/keyframe/" + index[i] + ".png",
                                        "multi_frame_src/mask/mask0.jpg" };

        cv::Mat A_prime = imread("multi_frame_src/keyframe/" + index[i] + ".png");
        A_prime_merge[i] = A_prime.clone();
        cv::Mat A;

        for (int j = 0; j < pics_num; j++) {
            if (j == 0) {
                A = cv::imread(A_start_path[j]);
                A_merge[i] = A.clone();
            }
            else {
                cv::merge(std::vector<cv::Mat>{ A, cv::imread(A_start_path[j]) }, A);
                cv::merge(std::vector<cv::Mat>{ A_merge[i], cv::imread(A_start_path[j]) }, A_merge[i]);
            }

        }
        gaussian_pyramid(A_prime, gaussian_levels, A_prime_pyramids[i]);
        gaussian_pyramid(A, gaussian_levels, A_frame_pyramids[i]);

    }





    int* pre_nnf = nullptr;

    int should_be = 0;

    std::vector<cv::Mat> A_pre_prime_pyramids, A_pre_pyramids;

    gaussian_pyramid(A_prime_merge[0], gaussian_levels, A_pre_prime_pyramids);
    gaussian_pyramid(A_merge[0], gaussian_levels, A_pre_pyramids);

    for (int i = 1; i < key_cnt; i++) {
        if (i % 2 == 1) {
            if (i < (L_DIVIDER + M_DIVIDER)/2) {
                gaussian_pyramid(A_prime_merge[0], gaussian_levels, A_pre_prime_pyramids);
                gaussian_pyramid(A_merge[0], gaussian_levels, A_pre_pyramids);
                should_be = 0;
            }
            else if( i < M_DIVIDER) {
                gaussian_pyramid(A_prime_merge[1], gaussian_levels, A_pre_prime_pyramids);
                gaussian_pyramid(A_merge[1], gaussian_levels, A_pre_pyramids);
                should_be = 1;
            }
            else if (i < (M_DIVIDER + R_DIVIDER) / 2) {
                gaussian_pyramid(A_prime_merge[1], gaussian_levels, A_pre_prime_pyramids);
                gaussian_pyramid(A_merge[1], gaussian_levels, A_pre_pyramids);
                should_be = 1;
            }
            else if (i < R_DIVIDER) {
                gaussian_pyramid(A_prime_merge[2], gaussian_levels, A_pre_prime_pyramids);
                gaussian_pyramid(A_merge[2], gaussian_levels, A_pre_pyramids);
                should_be = 2;
            }
            else if (i < (R_DIVIDER + E_DIVIDER) / 2) {
                gaussian_pyramid(A_prime_merge[2], gaussian_levels, A_pre_prime_pyramids);
                gaussian_pyramid(A_merge[2], gaussian_levels, A_pre_pyramids);
                should_be = 2;
            }
            else {
                gaussian_pyramid(A_prime_merge[3], gaussian_levels, A_pre_prime_pyramids);
                gaussian_pyramid(A_merge[3], gaussian_levels, A_pre_pyramids);
                should_be = 3;
            }
        }

        printf("processing % d picture \n", i);
        auto index = std::to_string(i);
        cv::Mat B_merge;


        std::string B_path[5] = { src_name + index + ".png",
                                  "multi_frame_src/edge/edge" + index + ".jpg",
                                  "multi_frame_src/pos/pos" + index + ".jpg",
                                  src_name + index + ".png",
                                  "multi_frame_src/mask/mask" + index + ".jpg" };


        cv::Mat B_prime = imread(B_path[0]);
        std::vector<cv::Mat> B_pyramids, B_prime_pyramids;
        int B_width = B_prime.cols;
        int B_height = B_prime.rows;

        for (int i = 0; i < pics_num; i++) {
            if (i == 0) {
                B_merge = cv::imread(B_path[i]);
            }
            else {
                cv::merge(std::vector<cv::Mat>{ B_merge, cv::imread(B_path[i]) }, B_merge);
            }
        }

        gaussian_pyramid(B_merge, gaussian_levels, B_pyramids);
        gaussian_pyramid(B_prime, gaussian_levels, B_prime_pyramids);


        for (int level = gaussian_levels - 1; level >= 0; level--) {
            // do one Patch Match
            nnf = pyramid_pass(A_pre_pyramids[level], B_pyramids[level], A_pre_prime_pyramids[level], B_prime_pyramids[level], nnf, level, level == gaussian_levels - 1);
            if (level > 0) {
                cv::pyrUp(B_prime_pyramids[level], B_prime_pyramids[level - 1]);
                nnf = pyrUp_NNF(nnf, B_prime_pyramids[level], B_prime_pyramids[level - 1].cols, B_prime_pyramids[level - 1].rows);
            }
        }


        // the first frame
        if (pre_nnf == nullptr || i%2 == 1) {
            // if allocated here, it will be freed immediately

            /*pre_nnf = new int[2 * B_width * B_height];
            memcpy(pre_nnf, nnf, 2 * B_width * B_height * sizeof(float));*/
        }
        else {
            for( int i = 0; i < B_height; i++)
                for (int j = 0; j < B_width; j++) {
                    int n_x = nnf[2 * (i * B_width + j)];
                    int n_y = nnf[2 * (i * B_width + j) + 1];
                    // here use nnf directly, not changing pre_nnf to avoid super-step in repeatition usage
                    nnf[2 * (i * B_width + j)] = pre_nnf[2 * (n_x * B_width + n_y)];
                    nnf[2 * (i * B_width + j) + 1] = pre_nnf[2 * (n_x * B_width + n_y) + 1];
                }
        }
        delete[] pre_nnf;
        pre_nnf = nnf;
        nnf = nullptr;
        /*imshow("Before APPLY", B_prime_pyramids[0]);*/
        // refresh the B_prime_pyramids with A_prime
        apply_frame_pyramids(should_be, A_prime_merge[should_be], B_prime_pyramids[0], A_pre_prime_pyramids, pre_nnf);
        /*imshow("After APPLY", B_prime_pyramids[0]);
        waitKey();
        cv::destroyAllWindows();*/
        cv::imwrite("multi_frame_src/output/B_prime" + std::to_string(i) + ".jpeg", B_prime_pyramids[0]);

        A_pre_pyramids = std::move(B_pyramids);

    }
    delete[] nnf;
}
void apply_frame_pyramids(int i, cv::Mat& A_start_prime, cv::Mat& b_prime, std::vector<cv::Mat>& B_prime_pyramids, int* nnf) {
    int B_width = b_prime.cols;
    int B_height = b_prime.rows;

    int idx;

    // now just leave it as is, since a blend of patch algorithm maybe implement
    for( int j = 0; j < B_height; j++)
        for (int k = 0; k < B_width; k++) {
            idx = j * B_width + k;

            b_prime.at<Vec3b>(j, k) = A_start_prime.at<Vec3b>(nnf[2 * idx], nnf[2 * idx + 1]);

        }
    gaussian_pyramid(b_prime, gaussian_levels, B_prime_pyramids);
}