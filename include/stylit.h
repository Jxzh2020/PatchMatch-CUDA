#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>
#include <omp.h>

using namespace std;
using namespace cv;

#define w 5
#define iteration 6
#define errorbuget 100000


class NNF {
private:
	Mat A, B, Aprime, Bprime, Afloat, Bfloat, Apfloat, Bpfloat;
	int Arows, Acols, Brows, Bcols;
	Mat nnf;
	Mat nnfer;
	Mat cp;
	std::pair<int, int> reverse_NNF_para(const cv::Mat& A, const cv::Mat& B);

public:
	Mat bmap;
	Mat bmaper;

	NNF(const Mat& A, const Mat& B, const Mat& Aprime, const Mat& Bprime);

	void nnf_init(int K, int R, int u);


	int nnf_iter(int R, int K, int u, int iterations = 6);

	void address_conflict(int x, int y);

	void propagation(int A_x, int A_y, bool odd_flag, int u);

	void random_search(int A_x, int A_y, int u);

	void dealer_reverse_NNF(int u, int iter = 6);

	float calculate(int A_x, int A_y, int B_x, int B_y, int u, float best = std::numeric_limits<float>::infinity());
};

cv::Mat average(const cv::Mat& A, const cv::Mat& Aprime, const cv::Mat& nnf, int B_x, int B_y);

void stylit(cv::Mat& A, cv::Mat& B, cv::Mat& Aprime, cv::Mat& Bprime, int channel);

//float calculate(const cv::Mat& A, const cv::Mat& B, const cv::Mat& Aprime, const cv::Mat& Bprime, int A_x, int A_y, int B_x, int B_y, int u, float best);