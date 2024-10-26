#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <opencv2/opencv.hpp>
#include <omp.h>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>

// Hàm tiện ích để giới hạn giá trị trong khoảng từ 0 đến 255
inline int clamp(int value) {
    return value > 255 ? 255 : (value < 0 ? 0 : value);
}
// Các hàm khác (applySobel, applyPrewitt, applyLaplacian, v.v.) giữ nguyên
// Cập nhật hàm phát hiện cạnh chính để sử dụng bộ lọc mới và ngưỡng động

void applySobel(const cv::Mat& grayImage, cv::Mat& sobelImage) {
    static const int kernelX[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}
    };

    static const int kernelY[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    #pragma omp parallel for collapse(2) schedule(dynamic, 16)
    for (int i = 1; i < grayImage.rows - 1; i++) {
        for (int j = 1; j < grayImage.cols - 1; j++) {
            int gx = 0, gy = 0;
            
            // Unroll the inner loops for better performance
            // Row 1
            gx += grayImage.at<uchar>(i-1, j-1) * kernelX[0][0];
            gy += grayImage.at<uchar>(i-1, j-1) * kernelY[0][0];
            gx += grayImage.at<uchar>(i-1, j) * kernelX[0][1];
            gy += grayImage.at<uchar>(i-1, j) * kernelY[0][1];
            gx += grayImage.at<uchar>(i-1, j+1) * kernelX[0][2];
            gy += grayImage.at<uchar>(i-1, j+1) * kernelY[0][2];
            
            // Row 2
            gx += grayImage.at<uchar>(i, j-1) * kernelX[1][0];
            gy += grayImage.at<uchar>(i, j-1) * kernelY[1][0];
            gx += grayImage.at<uchar>(i, j) * kernelX[1][1];
            gy += grayImage.at<uchar>(i, j) * kernelY[1][1];
            gx += grayImage.at<uchar>(i, j+1) * kernelX[1][2];
            gy += grayImage.at<uchar>(i, j+1) * kernelY[1][2];
            
            // Row 3
            gx += grayImage.at<uchar>(i+1, j-1) * kernelX[2][0];
            gy += grayImage.at<uchar>(i+1, j-1) * kernelY[2][0];
            gx += grayImage.at<uchar>(i+1, j) * kernelX[2][1];
            gy += grayImage.at<uchar>(i+1, j) * kernelY[2][1];
            gx += grayImage.at<uchar>(i+1, j+1) * kernelX[2][2];
            gy += grayImage.at<uchar>(i+1, j+1) * kernelY[2][2];

            sobelImage.at<uchar>(i, j) = clamp(std::sqrt(gx * gx + gy * gy));
        }
    }
}

void applyPrewitt(const cv::Mat& grayImage, cv::Mat& prewittImage) {
    static const int kernelX[3][3] = {
        {-1, -1, -1},
        {0, 0, 0},
        {1, 1, 1}
    };

    static const int kernelY[3][3] = {
        {-1, 0, 1},
        {-1, 0, 1},
        {-1, 0, 1}
    };

    #pragma omp parallel for collapse(2) schedule(dynamic, 16)
    for (int i = 1; i < grayImage.rows - 1; i++) {
        for (int j = 1; j < grayImage.cols - 1; j++) {
            int gx = 0, gy = 0;
            
            // Unrolled kernel operations
            #pragma omp simd reduction(+:gx,gy)
            for (int k = 0; k < 9; k++) {
                int kx = k / 3 - 1;
                int ky = k % 3 - 1;
                int pixel = grayImage.at<uchar>(i + kx, j + ky);
                gx += pixel * kernelX[kx + 1][ky + 1];
                gy += pixel * kernelY[kx + 1][ky + 1];
            }

            prewittImage.at<uchar>(i, j) = clamp(std::sqrt(gx * gx + gy * gy));
        }
    }
}

void applyLaplacian(const cv::Mat& grayImage, cv::Mat& laplacianImage) {
    static const int kernel[3][3] = {
        {0, 1, 0},
        {1, -4, 1},
        {0, 1, 0}
    };

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < grayImage.rows - 1; i++) {
        for (int j = 1; j < grayImage.cols - 1; j++) {
            int sum = 0;
            
            // Direct calculation instead of loops
            sum = grayImage.at<uchar>(i-1, j) +
                  grayImage.at<uchar>(i+1, j) +
                  grayImage.at<uchar>(i, j-1) +
                  grayImage.at<uchar>(i, j+1) -
                  4 * grayImage.at<uchar>(i, j);

            laplacianImage.at<uchar>(i, j) = clamp(std::abs(sum));
        }
    }
}


// Hàm bộ lọc thống kê (Cập nhật so với Gaussian Blurr)
void applyStatisticalFilter(const cv::Mat& grayImage, cv::Mat& filteredImage) {
    #pragma omp parallel for collapse(2) schedule(dynamic, 8)
    for (int i = 1; i < grayImage.rows - 1; i++) {
        for (int j = 1; j < grayImage.cols - 1; j++) {
            int sum = 0, sumSq = 0, n = 9;

            // Tính tổng và tổng bình phương các giá trị lân cận
            for (int x = -1; x <= 1; x++) {
                for (int y = -1; y <= 1; y++) {
                    int pixel = grayImage.at<uchar>(i + x, j + y);
                    sum += pixel;
                    sumSq += pixel * pixel;
                }
            }

            // Tính trung bình và phương sai
            float mean = sum / float(n);
            float variance = (sumSq / float(n)) - (mean * mean);

            // Sử dụng phương sai để lọc nhiễu
            int pixel = grayImage.at<uchar>(i, j);
            filteredImage.at<uchar>(i, j) = std::abs(pixel - mean) > variance ? mean : pixel;
        }
    }
}

// Thuật toán di truyền để tối ưu hóa ngưỡng động
std::pair<uchar, uchar> geneticThresholdOptimization(const cv::Mat& gradientImage) {
    const int populationSize = 20;
    const int generations = 100;
    const uchar minThreshold = 50;
    const uchar maxThreshold = 200;
    
    std::vector<std::pair<uchar, uchar>> population(populationSize);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(minThreshold, maxThreshold);

    // Khởi tạo quần thể ban đầu
    for (auto& thresholds : population) {
        thresholds.first = dist(gen);
        thresholds.second = dist(gen);
    }

    // Hàm đánh giá chất lượng (fitness) của ngưỡng
    auto fitness = [&](uchar low, uchar high) {
        int strongEdges = 0, weakEdges = 0;
        for (int i = 0; i < gradientImage.rows; i++) {
            for (int j = 0; j < gradientImage.cols; j++) {
                uchar pixel = gradientImage.at<uchar>(i, j);
                if (pixel >= high) strongEdges++;
                else if (pixel >= low) weakEdges++;
            }
        }
        return strongEdges + weakEdges * 0.5; // Cân bằng giữa cạnh mạnh và cạnh yếu
    };

    // Thực hiện tiến hóa qua nhiều thế hệ
    for (int gen = 0; gen < generations; ++gen) {
        // Sắp xếp quần thể theo chất lượng
        std::sort(population.begin(), population.end(), [&](auto& a, auto& b) {
            return fitness(a.first, a.second) > fitness(b.first, b.second);
        });

        // Phép lai (crossover)
        for (int i = populationSize / 2; i < populationSize; i++) {
            uchar parent1_low = population[i - 1].first;
            uchar parent1_high = population[i - 1].second;
            uchar parent2_low = population[i - 2].first;
            uchar parent2_high = population[i - 2].second;

            population[i].first = (parent1_low + parent2_low) / 2;
            population[i].second = (parent1_high + parent2_high) / 2;
        }

        // Phép đột biến (mutation)
        for (int i = 0; i < populationSize; i++) {
            if (gen % 10 == 0) {
                population[i].first = dist(gen);
                population[i].second = dist(gen);
            }
        }
    }
    
    // Trả về cặp ngưỡng tốt nhất
    return population.front();
}

// Áp dụng ngưỡng động tối ưu
void applyDynamicThreshold(const cv::Mat& sobelImage, cv::Mat& dtImage) {
    auto [lowThreshold, highThreshold] = geneticThresholdOptimization(sobelImage);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < sobelImage.rows - 1; i++) {
        for (int j = 1; j < sobelImage.cols - 1; j++) {
            const uchar pixel = sobelImage.at<uchar>(i, j);
            dtImage.at<uchar>(i, j) = pixel > highThreshold ? 255 : (pixel < lowThreshold ? 0 : pixel);
        }
    }
}

#endif
