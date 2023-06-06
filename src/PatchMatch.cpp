//
// Created by Hao on 2023/6/3.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "stylit.h"

#define channel 4
#define gaussian_levels 1

void gaussian_pyramid(const cv::Mat& img, int levels, std::vector<cv::Mat>& pyramid)
{
    // 将多通道图像拆分成四个单通道图像
    std::vector<cv::Mat> channels;
    if (channel > 1)
    {
        cv::split(img, channels);
    }


    pyramid.clear(); // 清空金字塔向量

    // 分别对每个通道计算高斯金字塔
    for (int i = 0; i < levels; i++) {
        cv::Mat merged_image; // 保存合并后的图像

        if (channel == 1)
        {
            merged_image = img.clone();
            for (int j = 0; j < i; j++) {
                cv::pyrDown(merged_image.clone(), merged_image);
            }
            //cv::imshow("1", merged_image);
            //cv::waitKey(0);
        }
        else
        {
            // 遍历四个通道，合并同一层级的图像
            for (int k = 0; k < channels.size(); ++k) {
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
        }

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
        cv::pyrUp(result, up, l_pyramid[i-1].size());
        result = up + l_pyramid[i - 1];
    }
}

int gen_cpu()
{
    std::string path = "../img/";

    std::cout << "Reading images begins..." << std::endl;

    cv::Mat imgAprime = cv::imread(path + "source_style_2.png");
    //cvtColor(imgAprime, imgAprime, COLOR_BGR2GRAY);

    std::vector<cv::Mat> imgAv, imgBv;
    cv::Mat imgA0 = cv::imread(path + "source_fullgi_2.png");
    cv::Mat imgB0 = cv::imread(path + "target_fullgi_2.png");
    //cvtColor(imgA0, imgA0, COLOR_BGR2GRAY);
    //cvtColor(imgB0, imgB0, COLOR_BGR2GRAY);
    imgAv.push_back(imgA0);
    imgBv.push_back(imgB0);

    cv::Mat imgA1 = cv::imread(path + "source_dirdif_2.png");
    cv::Mat imgB1 = cv::imread(path + "target_dirdif_2.png");
    //cvtColor(imgA1, imgA1, COLOR_BGR2GRAY);
    //cvtColor(imgB1, imgB1, COLOR_BGR2GRAY);
    imgAv.push_back(imgA1);
    imgBv.push_back(imgB1);

    cv::Mat imgA2 = cv::imread(path + "source_dirspc_2.png");
    cv::Mat imgB2 = cv::imread(path + "target_dirspc_2.png");
    //cvtColor(imgA2, imgA2, COLOR_BGR2GRAY);
    //cvtColor(imgB2, imgB2, COLOR_BGR2GRAY);
    imgAv.push_back(imgA2);
    imgBv.push_back(imgB2);

    cv::Mat imgA3 = cv::imread(path + "source_indirb_2.png");
    cv::Mat imgB3 = cv::imread(path + "target_indirb_2.png");
    //cvtColor(imgA3, imgA3, COLOR_BGR2GRAY);
    //cvtColor(imgB3, imgB3, COLOR_BGR2GRAY);
    imgAv.push_back(imgA3);
    imgBv.push_back(imgB3);


    std::cout << "Reading images finished..." << std::endl;

    cv::Mat imgA;
    cv::merge(imgAv, imgA);

    cv::Mat imgB;
    cv::merge(imgBv, imgB);

    cv::Mat imgBprime(imgB0.size(), CV_8UC3, cv::Scalar(255, 255, 255));

    std::vector<cv::Mat> A_pyramids;
    std::vector<cv::Mat> B_pyramids;
    std::vector<cv::Mat> Aprime_pyramids;
    std::vector<cv::Mat> Bprime_pyramids;

    gaussian_pyramid(imgA, gaussian_levels, A_pyramids);
    gaussian_pyramid(imgB, gaussian_levels, B_pyramids);
    gaussian_pyramid(imgAprime, gaussian_levels, Aprime_pyramids);

    std::vector<cv::Mat> l_pyramids;


    for (int level = 0; level < gaussian_levels; level++) {
        std::cout << "Stylit begins..." << std::endl;
        imgBprime = cv::Mat(B_pyramids[level].size(), CV_8UC3, cv::Scalar(255, 255, 255));
        stylit(A_pyramids[level], B_pyramids[level], Aprime_pyramids[level], imgBprime, channel);
        Bprime_pyramids.push_back(imgBprime.clone());
        std::cout << "Stylit finished..." << std::endl;
    }

    laplacian_pyramid(Bprime_pyramids, gaussian_levels, l_pyramids);

    cv::Mat imgBprime_merge = Bprime_pyramids[0].clone();
    cv::imshow("img", imgBprime_merge);
    cv::waitKey(0);
    imgBprime_merge.convertTo(imgBprime_merge, CV_32FC3);

    for (int level = gaussian_levels - 1; level > 0; level--) {
        cv::Mat current;
        recover_img(Bprime_pyramids[level], current, level, l_pyramids);
        current.convertTo(current, CV_32FC3);
        imgBprime_merge += current;
    }

    imgBprime_merge /= gaussian_levels;
    imgBprime_merge.convertTo(imgBprime_merge, CV_8UC3);

    cv::imwrite(path + "result.png", imgBprime_merge);
    cv::imshow("img", imgBprime_merge);
    cv::waitKey(0);

    return 0;
}

