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
static float u[5] = { 4, 12, 1, 0, 1 }; // col, edge, pos, temp, mask 3 16 0 0 1 / 4 16 1 0 1

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

        //if (channel == 1)
        //{
        //    merged_image = img.clone();
        //    for (int j = 0; j < i; j++) {
        //        cv::pyrDown(merged_image.clone(), merged_image);
        //    }
        //}
        //else
        //{
        //    // 遍历四个通道，合并同一层级的图像
        //    for (int k = 0; k < channel; ++k) {
        //        cv::Mat current = channels[k].clone();
        //        for (int j = 0; j < i; j++) {
        //            cv::pyrDown(current, current);
        //        }

        //        // 调整图像大小与第一个通道的图像相匹配
        //        if (k == 0) {
        //            merged_image = current.clone();
        //        }
        //        else {
        //            cv::resize(current, current, merged_image.size());
        //            cv::merge(std::vector<cv::Mat>{merged_image, current}, merged_image);
        //        }
        //    }
        //}
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
        //if (channel == 3) {
        //    cv::imshow("pyramids", merged_image);
        //    waitKey(0);
        //    cv::destroyAllWindows();
        //}
        
        pyramid.push_back(merged_image.clone()); // 将合并后的图像存储到金字塔中
    }
}


void laplacian_pyramid(const std::vector<cv::Mat>& pyramid, int levels, std::vector<cv::Mat>& l_pyramid)
{
    l_pyramid.clear();
    for (int i = 0; i < levels - 1; i++) {
        cv::Mat up;
        cv::pyrUp(pyramid[i + 1], up, pyramid[i].size());
        cv::Mat laplacian = pyramid[i] - up;
        l_pyramid.push_back(laplacian);
    }
}

void recover_img(cv::Mat& img, cv::Mat& result, int level, const std::vector<cv::Mat>& l_pyramid)
{
    result = img.clone();
    for (int i = level; i > 0; i--)
    {
        cv::Mat up;
        cv::pyrUp(result, up, l_pyramid[i - 1].size());
        result = up + l_pyramid[i - 1];
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
    
    patchMatch(b, a, b_prime, a_prime, B_width, B_height, A_width, A_height, channels, patch_size, u, round,num_iterations, nnf_src_a);

    //float* new_b_prime = new float[B_width * B_height * A_prime.channels()];
    //memcpy(new_b_prime, b_prime, sizeof(float) * B_width * B_height * A_prime.channels());

    //patchMatch(b_prime, a_prime, new_b_prime, a_prime, B_width, B_height, A_width, A_height, A_prime.channels(), patch_size, u, num_iterations, nnf_src_a);


    for (int i = 0; i < B_height; i++) {
        for (int j = 0; j < B_width; j++) {
            for (int k = 0; k < A_prime.channels(); k++)
                B_prime_channel[k].at<uchar>(i, j) = static_cast<uchar>(b_prime[(i * B_width + j) * A_prime.channels() + k]);
        }
    }
    cv::merge(B_prime_channel, B_prime);
    /*cv::imshow("kk", B_prime);
    waitKey(0);
    cv::destroyAllWindows();*/
    delete[] a;
    delete[] b;
    delete[] a_prime;
    delete[] b_prime;
    return nnf_src_a;
}


void gen_gpu() {

    const std::string src_name = "video/data/test/video/";
    const std::string guide_name = "video/guides/";
    int key_cnt = 99;
    int* nnf = nullptr;
    std::string index;

    cv::Mat A_start_merge, A_end_merge;

    std::string A_start_path[5] = { src_name + "000.jpg",
                              guide_name + "edge0.png",
                              guide_name + "pos0.png",
        src_name + "000.jpg",
                              //guide_name + "temp0.png",
                              "video/img/mask/000.jpg" };
    std::string A_end_path[5] = { src_name + "099.jpg",
                              guide_name + "edge99.png",
                              guide_name + "pos99.png",
        src_name + "099.jpg",
                              //guide_name + "temp99.png",
                              "video/img/mask/099.jpg" };

    cv::Mat A_start_prime = imread("video/data/test/keys/000.jpg");
    cv::Mat A_end_prime = imread("video/data/test/keys/099.jpg");

    for (int i = 0; i < pics_num; i++) {
        if (i == 0) {
            A_start_merge = cv::imread(A_start_path[i]);
            A_end_merge = cv::imread(A_end_path[i]);
        }
        else {
            cv::merge(std::vector<cv::Mat>{ A_start_merge, cv::imread(A_start_path[i]) }, A_start_merge);
            cv::merge(std::vector<cv::Mat>{ A_end_merge, cv::imread(A_end_path[i]) }, A_end_merge);
        }

    }

    std::vector<cv::Mat> A_start_pyramids, A_end_pyramids, A_start_prime_pyramids, A_end_prime_pyramids;

    gaussian_pyramid(A_start_merge, gaussian_levels, A_start_pyramids);
    gaussian_pyramid(A_end_merge, gaussian_levels, A_end_pyramids);
    gaussian_pyramid(A_start_prime, gaussian_levels, A_start_prime_pyramids);
    gaussian_pyramid(A_end_prime, gaussian_levels, A_end_prime_pyramids);
    


    for (int i = 1; i < key_cnt; i++) {
        //if (i != 5)
        //    continue;
        index = std::to_string(i);
        delete[] nnf;
        nnf = nullptr;

        cv::Mat B_merge;
        

        std::string B_path[5] = { src_name + "0" + (i < 10 ? ("0" + index) : index) + ".jpg",
                                  guide_name + "edge" + index + ".png", 
                                  guide_name + "pos" + index + ".png",
                                  guide_name + "temp" + index + ".png",
                                  "video/img/mask/0" + (i < 10 ? ("0" + index) : index) + ".jpg" };

        
        cv::Mat B_prime = imread(B_path[0]);
        std::vector<cv::Mat> B_pyramids, B_prime_pyramids;


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

        if (i < 20) {
            for (int level = gaussian_levels - 1; level >= 0; level--) {
                // do one Patch Match
                nnf = pyramid_pass(A_start_pyramids[level], B_pyramids[level], A_start_prime_pyramids[level], B_prime_pyramids[level], nnf, level, level == gaussian_levels - 1);
                if (level > 0) {
                    cv::pyrUp(B_prime_pyramids[level], B_prime_pyramids[level - 1]);
                    nnf = pyrUp_NNF(nnf, B_prime_pyramids[level], B_prime_pyramids[level - 1].cols, B_prime_pyramids[level - 1].rows);
                }
            }
        }
        else {
            for (int level = gaussian_levels - 1; level >= 0; level--) {
                // do one Patch Match
                nnf = pyramid_pass(A_end_pyramids[level], B_pyramids[level], A_end_prime_pyramids[level], B_prime_pyramids[level], nnf, level, level == gaussian_levels - 1);
                if (level > 0) {
                    cv::pyrUp(B_prime_pyramids[level], B_prime_pyramids[level - 1]);
                    nnf = pyrUp_NNF(nnf, B_prime_pyramids[level], B_prime_pyramids[level - 1].cols, B_prime_pyramids[level - 1].rows);
                }
            }
        }
        
        imwrite("output/B_prime" + std::to_string(i) + ".jpeg", B_prime_pyramids[0]);
    }
    delete[] nnf;
}