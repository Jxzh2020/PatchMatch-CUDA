#include "stylit.h"

int Channel;
std::pair<int, int> NNF::reverse_NNF_para(const cv::Mat& A, const cv::Mat& B) {
	int size_A = A.cols * A.rows;
	int size_B = B.cols * B.rows;
	int K = static_cast<int>(std::floor(size_B / size_A));
	int R = size_B % size_A;
	return std::make_pair(K, R);
}

NNF::NNF(const Mat& A, const Mat& B, const Mat& Aprime, const Mat& Bprime) {
    this->A = A;
    this->B = B;
    this->Aprime = Aprime;
    this->Bprime = Bprime;
    A.convertTo(this->Afloat, CV_32FC3);
    B.convertTo(this->Bfloat, CV_32FC3);
    Aprime.convertTo(this->Apfloat, CV_32FC3);
    Bprime.convertTo(this->Bpfloat, CV_32FC3);
    this->Arows = A.rows;
    this->Acols = A.cols;
    this->Brows = B.rows;
    this->Bcols = B.cols;
    this->nnf = Mat(Arows, Acols, CV_32SC2, Scalar(0, 0));
    this->bmap = Mat(Brows, Bcols, CV_32SC2, Scalar(0, 0));
    this->nnfer = Mat(Arows, Acols, CV_32F, Scalar(0));
    this->bmaper = Mat(Brows, Bcols, CV_32F, Scalar(0));
    this->cp = Mat(Arows, Acols, CV_32S, Scalar(0));
}

void NNF::nnf_init(int K, int R, int u) {
    cout << "NNF init begins..." << endl;

    vector<Point> available;
    for (int x = 0; x < Brows; x++) {
        for (int y = 0; y < Bcols; y++) {
            //cout << bmap.type();
            if (bmaper.at<float>(x, y) == 0) {
                available.push_back(Point(x, y));
            }
        }
    }

    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::shuffle(available.begin(), available.end(), gen);

    int index = 0;
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < Arows; i++) {
        for (int j = 0; j < Acols; j++) {
            if ((cp.at<int>(i, j) <= K && R != 0) || cp.at<int>(i, j) < K) {
                int x_b, y_b;
                //#pragma omp critical
                //{
                x_b = available[index].x;
                y_b = available[index].y;
                index = (index + 1) % available.size();
                //}

                nnf.at<Vec2i>(i, j)[0] = x_b;
                nnf.at<Vec2i>(i, j)[1] = y_b;
                /*nnfer.at<float>(i, j) = calculate(A, B, Aprime, Bprime, i, j, x_b, y_b, u);*/
                nnfer.at<float>(i, j) = calculate(i, j, x_b, y_b, u);
            }
        }
    }

    std::cout << "NNF init finished..." << endl;
}


int NNF::nnf_iter(int R, int K, int u, int iterations) {
    cout << "Iteration begins..." << endl;
    int R_local = R;

    for (int i = 0; i < iterations; i++) {
        cout << "Iteration Num. " << i + 1 << " begins..." << endl;

        // Odd time: order from up down, left to right
        if (i % 2 == 1) {
            for (int x = 0; x < Arows; x++) {
                for (int y = 0; y < Acols; y++) {
                    if ((cp.at<int>(x, y) <= K && R != 0) || cp.at<int>(x, y) < K) {
                        propagation(x, y, i % 2, u);
                        random_search(x, y, u);

                        // At the last iteration time, we increase cp and assign the patch of B to A
                        if (i == iterations - 1) {
                            if (cp.at<int>(x, y) == K) {
                                R_local -= 1;
                            }
                            address_conflict(x, y);
                        }
                    }
                }
            }
        }
        // Even time: reversed scanning order
        else {
            for (int x = Arows - 1; x >= 0; x--) {
                for (int y = Acols - 1; y >= 0; y--) {
                    if ((cp.at<int>(x, y) <= K && R != 0) || cp.at<int>(x, y) < K) {
                        propagation(x, y, i % 2, u);
                        random_search(x, y, u);

                        if (i == iterations - 1) {
                            if (cp.at<int>(x, y) == K) {
                                R_local -= 1;
                            }
                            address_conflict(x, y);
                        }
                    }
                }
            }
        }

        cout << "Iteration Num. " << i + 1 << " finished..." << endl;
    }

    cout << "Iteration finished..." << endl;
    return R_local;
}

void NNF::address_conflict(int x, int y) {
    if (nnfer.at<float>(x, y) < bmaper.at<float>(nnf.at<Vec2i>(x, y)[0], nnf.at<Vec2i>(x, y)[1])) {
        cp.at<int>(bmap.at<Vec2i>(nnf.at<Vec2i>(x, y)[0], nnf.at<Vec2i>(x, y)[1])[0],
            bmap.at<Vec2i>(nnf.at<Vec2i>(x, y)[0], nnf.at<Vec2i>(x, y)[1])[1]) -= 1;
        bmap.at<Vec2i>(nnf.at<Vec2i>(x, y)[0], nnf.at<Vec2i>(x, y)[1])[0] = x;
        bmap.at<Vec2i>(nnf.at<Vec2i>(x, y)[0], nnf.at<Vec2i>(x, y)[1])[1] = y;
        bmaper.at<float>(nnf.at<Vec2i>(x, y)[0], nnf.at<Vec2i>(x, y)[1]) = nnfer.at<float>(x, y);
        cp.at<int>(x, y) += 1;
    }
    else if (bmaper.at<float>(nnf.at<Vec2i>(x, y)[0], nnf.at<Vec2i>(x, y)[1]) == 0) {
        bmap.at<Vec2i>(nnf.at<Vec2i>(x, y)[0], nnf.at<Vec2i>(x, y)[1])[0] = x;
        bmap.at<Vec2i>(nnf.at<Vec2i>(x, y)[0], nnf.at<Vec2i>(x, y)[1])[1] = y;
        bmaper.at<float>(nnf.at<Vec2i>(x, y)[0], nnf.at<Vec2i>(x, y)[1]) = nnfer.at<float>(x, y);
        cp.at<int>(x, y) += 1;
    }
}

void NNF::propagation(int A_x, int A_y, bool odd_flag, int u) {
    int B_x = nnf.at<Vec2i>(A_x, A_y)[0];
    int B_y = nnf.at<Vec2i>(A_x, A_y)[1];
    float B_diff = nnfer.at<float>(A_x, A_y);

    if (!odd_flag) {
        // Left neighbor
        if (A_x - 1 >= 0) {
            int temp_bx = nnf.at<Vec2i>(A_x - 1, A_y)[0];
            int temp_by = nnf.at<Vec2i>(A_x - 1, A_y)[1];
            if (temp_bx + 1 < Brows  && bmaper.at<float>(temp_bx + 1, temp_by) == 0) {
                //float temp_bdiff = calculate(A, B, Aprime, Bprime, A_x, A_y, temp_bx + 1, temp_by, u);
                float temp_bdiff = calculate(A_x, A_y, temp_bx + 1, temp_by, u);
                if (temp_bdiff < B_diff && temp_bdiff < Channel*errorbuget) {
                    B_x = temp_bx + 1;
                    B_y = temp_by;
                    B_diff = temp_bdiff;
                }
            }
        }
        // Up neighbor
        if (A_y - 1 >= 0) {
            int temp_bx = nnf.at<Vec2i>(A_x, A_y - 1)[0];
            int temp_by = nnf.at<Vec2i>(A_x, A_y - 1)[1];
            if (temp_by + 1 < Brows && bmaper.at<float>(temp_bx, temp_by + 1) == 0) {
                //float temp_bdiff = calculate(A, B, Aprime, Bprime, A_x, A_y, temp_bx, temp_by + 1, u);
                float temp_bdiff = calculate(A_x, A_y, temp_bx, temp_by + 1, u);
                if (temp_bdiff < B_diff && temp_bdiff < Channel*errorbuget) {
                    B_x = temp_bx;
                    B_y = temp_by + 1;
                    B_diff = temp_bdiff;
                }
            }
        }
    }
    else {
        // Left neighbor
        if (A_x + 1 < Arows) {
            int temp_bx = nnf.at<Vec2i>(A_x + 1, A_y)[0];
            int temp_by = nnf.at<Vec2i>(A_x + 1, A_y)[1];
            if (temp_bx - 1 >= 0 && bmaper.at<float>(temp_bx - 1, temp_by) == 0) {
                //float temp_bdiff = calculate(A, B, Aprime, Bprime, A_x, A_y, temp_bx - 1, temp_by, u);
                float temp_bdiff = calculate(A_x, A_y, temp_bx - 1, temp_by, u);
                if (temp_bdiff < B_diff && temp_bdiff < Channel*errorbuget) {
                    B_x = temp_bx - 1;
                    B_y = temp_by;
                    B_diff = temp_bdiff;
                }
            }
        }
        // Up neighbor
        if (A_y + 1 < Acols) {
            int temp_bx = nnf.at<Vec2i>(A_x, A_y + 1)[0];
            int temp_by = nnf.at<Vec2i>(A_x, A_y + 1)[1];
            if (temp_by - 1 >= 0 && bmaper.at<float>(temp_bx, temp_by - 1) == 0) {
                //float temp_bdiff = calculate(A, B, Aprime, Bprime, A_x, A_y, temp_bx, temp_by - 1, u);
                float temp_bdiff = calculate(A_x, A_y, temp_bx, temp_by - 1, u);
                if (temp_bdiff < B_diff && temp_bdiff < Channel*errorbuget) {
                    B_x = temp_bx;
                    B_y = temp_by - 1;
                    B_diff = temp_bdiff;
                }
            }
        }
    }

    nnf.at<Vec2i>(A_x, A_y)[0] = B_x;
    nnf.at<Vec2i>(A_x, A_y)[1] = B_y;
    nnfer.at<float>(A_x, A_y) = B_diff;
}

void NNF::random_search(int A_x, int A_y, int u) {
    float radius = 8;
    float alpha = 0.5;
    cv::RNG random;

    int B_x = nnf.at<Vec2i>(A_x, A_y)[0] + static_cast<int>(radius * random.uniform(-1.0, 1.0));
    int B_y = nnf.at<Vec2i>(A_x, A_y)[1] + static_cast<int>(radius * random.uniform(-1.0, 1.0));

    while (radius >= 1) {
        while (B_x < 0 || B_x >= Brows || B_y < 0 || B_y >= Bcols) {
            B_x = nnf.at<Vec2i>(A_x, A_y)[0] + static_cast<int>(radius * random.uniform(-1.0, 1.0));
            B_y = nnf.at<Vec2i>(A_x, A_y)[1] + static_cast<int>(radius * random.uniform(-1.0, 1.0));
        }

        //float diff = calculate(A, B, Aprime, Bprime, A_x, A_y, B_x, B_y, u, nnfer.at<float>(A_x, A_y));
        float diff = calculate(A_x, A_y, B_x, B_y, u, nnfer.at<float>(A_x, A_y));
        if (diff < nnf.at<float>(A_x, A_y) && diff < Channel*errorbuget && bmaper.at<float>(B_x, B_y) == 0) {
            nnf.at<Vec2i>(A_x, A_y)[0] = B_x;
            nnf.at<Vec2i>(A_x, A_y)[1] = B_y;
            nnfer.at<float>(A_x, A_y) = diff;
        }
        radius *= alpha;
    }
}

void NNF:: dealer_reverse_NNF(int u, int iter) {
    std::pair<int, int> parameters = reverse_NNF_para(A, B);
    int K = parameters.first;
    int R = parameters.second;
    int cover = 0;
    int total = B.cols * B.rows;

    while (cover < total) {
        std::cout << "Cover pixels in B are " << cover << "/" << total << std::endl;
        nnf_init(K, R, u);
        R = nnf_iter(R, K, u, iter);
        cover = int(cv::sum(cp)[0]);
    }
    std::cout << "Cover pixels in B are " << cover << "/" << total << std::endl;
}

cv::Mat average(const cv::Mat& A, const cv::Mat& Apfloat, const cv::Mat& nnf, int B_x, int B_y) {
	int A_x = nnf.at<Vec2i>(B_x, B_y)[0];
	int A_y = nnf.at<Vec2i>(B_x, B_y)[1];
	int Arows = A.rows;
	int Acols = A.cols;
	cv::Mat cost(1, 1, CV_32FC3, cv::Scalar(0));
	int cnt = 0;

	int xmin = std::min(w / 2, A_x);
	int xmax = std::min(w / 2, Arows - A_x - 1) + 1;
	int ymin = std::min(w / 2, A_y);
	int ymax = std::min(w / 2, Acols - A_y - 1) + 1;

	for (int i = A_x - w / 2; i < A_x + w / 2 + 1; i++) {
		for (int j = A_y - w / 2; j < A_y + w / 2 + 1; j++) {
			if (i < 0 || j < 0 || i > Arows - 1 || j > Acols - 1) {
				continue;
			}
			cnt++;
			cost.at<Vec3f>(0, 0) += Apfloat.at<Vec3f>(i, j);
		}
	}

	cv::Mat num(1, 1, CV_32FC3, Scalar(cnt, cnt, cnt));
	cost = cost / num;
	cost.convertTo(cost, CV_8UC3);
	//cout << cost.at<Vec3b>(0, 0) << std::endl;
	return cost;
}

float NNF::calculate(int A_x, int A_y, int B_x, int B_y, int u, float best) {
	float cost = 0;

	int xmin = static_cast<int>(std::min(std::min(w / 2, A_x), B_x));
	int xmax = static_cast<int>(std::min(std::min(w / 2, Arows - A_x - 1), Brows - B_x - 1) + 1);
	int ymin = static_cast<int>(std::min(std::min(w / 2, A_y), B_y));
	int ymax = static_cast<int>(std::min(std::min(w / 2, Acols - A_y - 1), Bcols - B_y - 1) + 1);

	cv::Mat Af = Afloat(cv::Range(A_x - xmin, A_x + xmax), cv::Range(A_y - ymin, A_y + ymax));
	cv::Mat Bf = Bfloat(cv::Range(B_x - xmin, B_x + xmax), cv::Range(B_y - ymin, B_y + ymax));
	cv::Mat Apf = Apfloat(cv::Range(A_x - xmin, A_x + xmax), cv::Range(A_y - ymin, A_y + ymax));
	cv::Mat Bpf = Bpfloat(cv::Range(B_x - xmin, B_x + xmax), cv::Range(B_y - ymin, B_y + ymax));

	cv::Mat sub, subprime;
	cv::absdiff(Af, Bf, sub);
	cv::absdiff(Apf, Bpf, subprime);
	cv::pow(sub, 2, sub);
	cv::pow(subprime, 2, subprime);

	cv::Scalar sum_sub, sum_subprime;
	#pragma omp parallel sections
	{
		#pragma omp section
		sum_sub = cv::sum(sub.reshape(1, 1));
		#pragma omp section
		sum_subprime = cv::sum(subprime.reshape(1, 1));
	}

	cost = sum_sub[0] * u / Channel + sum_subprime[0];

	if (cost == 0) {
		return 0;
	}

	cost += 0.0001f;
	cost /= (xmin + xmax) * (ymin + ymax);
	return cost;
}



//float calculate(const cv::Mat& A, const cv::Mat& B, const cv::Mat& Aprime, const cv::Mat& Bprime, int A_x, int A_y, int B_x, int B_y, int u, float best) {
//	int Arows = A.rows;
//	int Acols = A.cols;
//	int Brows = B.rows;
//	int Bcols = B.cols;
//	float cost = 0;
//
//	int xmin = static_cast<int>(std::min(std::min(w / 2, A_x), B_x));
//	int xmax = static_cast<int>(std::min(std::min(w / 2, Arows - A_x - 1), Brows - B_x - 1) + 1);
//	int ymin = static_cast<int>(std::min(std::min(w / 2, A_y), B_y));
//	int ymax = static_cast<int>(std::min(std::min(w / 2, Acols - A_y - 1), Bcols - B_y - 1) + 1);
//
//	cv::Mat Afloat(A.size(), CV_32FC3);
//	cv::Mat Bfloat(B.size(), CV_32FC3);
//	cv::Mat Apfloat(Aprime.size(), CV_32FC3);
//	cv::Mat Bpfloat(Bprime.size(), CV_32FC3);
//
//	A.convertTo(Afloat, CV_32FC3);
//	B.convertTo(Bfloat, CV_32FC3);
//	Aprime.convertTo(Apfloat, CV_32FC3);
//	Bprime.convertTo(Bpfloat, CV_32FC3);
//
//	Afloat = Afloat(cv::Range(A_x - xmin, A_x + xmax), cv::Range(A_y - ymin, A_y + ymax)).clone();
//	Bfloat = Bfloat(cv::Range(B_x - xmin, B_x + xmax), cv::Range(B_y - ymin, B_y + ymax)).clone();
//	Apfloat = Apfloat(cv::Range(A_x - xmin, A_x + xmax), cv::Range(A_y - ymin, A_y + ymax)).clone();
//	Bpfloat = Bpfloat(cv::Range(B_x - xmin, B_x + xmax), cv::Range(B_y - ymin, B_y + ymax)).clone();
//
//	cv::Mat sub, subprime;
//	sub = Afloat - Bfloat;
//	subprime = Apfloat - Bpfloat;
//	cv::pow(sub, 2, sub);
//	cv::pow(subprime, 2, subprime);
//	cv::Scalar result = cv::sum((sub*u + subprime));
//	
//	//cost = cv::sum(u * cv::pow(A(cv::Range(0, Channel), cv::Range(A_x - xmin, A_x + xmax + 1), cv::Range(A_y - ymin, A_y + ymax + 1)).reshape(1, -1).t() - B(cv::Range(0, Channel), cv::Range(B_x - xmin, B_x + xmax + 1), cv::Range(B_y - ymin, B_y + ymax + 1)).reshape(1, -1).t(), 2))[0] / Channel +cv::sum(cv::pow(Aprime(cv::Range(A_x - xmin, A_x + xmax + 1), cv::Range(A_y - ymin, A_y + ymax + 1)) - Bprime(cv::Range(B_x - xmin, B_x + xmax + 1), cv::Range(B_y - ymin, B_y + ymax + 1)), 2));
//	for (int i = 0; i < Channel * 3; i++)
//	{
//		cost += float(result[i]);
//	}
//	
//	if (cost == 0) {
//		return 0;
//	}
//
//	cost += 0.0001f;
//	cost /= (xmin + xmax) * (ymin + ymax);
//	return cost;
//}

void stylit(cv::Mat& A, cv::Mat& B, cv::Mat& Aprime, cv::Mat& Bprime, int C) {
	cv::Mat avg(3, 1, CV_32F, cv::Scalar(2));
	Channel = C;
	for (int iter_time = 0; iter_time < iteration; iter_time++) {
		std::cout << "Round " << iter_time + 1 << " begins..." << std::endl;
		int u = 2;
		NNF deal_nnf(A, B, Aprime, Bprime);
		deal_nnf.dealer_reverse_NNF(u, iteration);
		cv::Mat nnf = deal_nnf.bmap;

		cv::Mat Apfloat;
		Aprime.convertTo(Apfloat, CV_32FC3);
		for (int i = 0; i < Bprime.rows; i++) {
			std::cout << "Row " << i + 1 << " generate in result..." << std::endl;
			for (int j = 0; j < Bprime.cols; j++) {
				if (iter_time == 0) {
					Bprime.at<cv::Vec3b>(i, j) = average(A, Apfloat, nnf, i, j).at<Vec3b>(0, 0);
				}
				else {
					//cv::Mat result = (average(A, Aprime, nnf, i, j).at<Vec3b>(0, 0) + Bprime.at<cv::Vec3b>(i, j)) / 2;
					Bprime.at<cv::Vec3b>(i, j) = average(A, Apfloat, nnf, i, j).at<Vec3b>(0, 0)/2 + Bprime.at<cv::Vec3b>(i, j)/2;
					//cout << Bprime.at<cv::Vec3b>(i, j) << endl;
				}
			}
		}
	}
}

