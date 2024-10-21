#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <opencv2/opencv.hpp>
#include <omp.h>

// Utility function to clamp values between 0 and 255
inline int clamp(int value) {
    return value > 255 ? 255 : (value < 0 ? 0 : value);
}

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

void applyGaussianBlur(const cv::Mat& grayImage, cv::Mat& gaussianBlurImage) {
    static const int kernel[5][5] = {
        {1, 4, 7, 4, 1},
        {4, 16, 26, 16, 4},
        {7, 26, 41, 26, 7},
        {4, 16, 26, 16, 4},
        {1, 4, 7, 4, 1}
    };
    static const int kernelSum = 273;

    #pragma omp parallel for collapse(2) schedule(dynamic, 8)
    for (int i = 2; i < grayImage.rows - 2; i++) {
        for (int j = 2; j < grayImage.cols - 2; j++) {
            int sum = 0;
            
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < 25; k++) {
                int kx = k / 5 - 2;
                int ky = k % 5 - 2;
                sum += grayImage.at<uchar>(i + kx, j + ky) * kernel[kx + 2][ky + 2];
            }

            gaussianBlurImage.at<uchar>(i, j) = clamp(sum / kernelSum);
        }
    }
}

void nonMaximumSuppression(const cv::Mat& sobelImage, cv::Mat& nmsImage) {
    // Calculate gradients for direction
    cv::Mat gradX, gradY;
    cv::Sobel(sobelImage, gradX, CV_32F, 1, 0, 3);
    cv::Sobel(sobelImage, gradY, CV_32F, 0, 1, 3);

    #pragma omp parallel for collapse(2) schedule(dynamic, 16)
    for (int i = 1; i < sobelImage.rows - 1; i++) {
        for (int j = 1; j < sobelImage.cols - 1; j++) {
            float gx = gradX.at<float>(i, j);
            float gy = gradY.at<float>(i, j);
            
            // Calculate gradient direction
            float angle = std::atan2(gy, gx) * 180.0 / CV_PI;
            // Normalize angle to positive values
            if (angle < 0) angle += 180;
            
            // Get the current pixel value
            int pixel = sobelImage.at<uchar>(i, j);
            
            // Initialize neighbors for interpolation
            float pixel1, pixel2;
            
            // Round angle to nearest 45 degrees and get corresponding neighbors
            if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
                // Horizontal direction
                pixel1 = sobelImage.at<uchar>(i, j+1);
                pixel2 = sobelImage.at<uchar>(i, j-1);
            }
            else if (angle >= 22.5 && angle < 67.5) {
                // Diagonal direction (/)
                pixel1 = sobelImage.at<uchar>(i+1, j-1);
                pixel2 = sobelImage.at<uchar>(i-1, j+1);
            }
            else if (angle >= 67.5 && angle < 112.5) {
                // Vertical direction
                pixel1 = sobelImage.at<uchar>(i+1, j);
                pixel2 = sobelImage.at<uchar>(i-1, j);
            }
            else if (angle >= 112.5 && angle < 157.5) {
                // Diagonal direction (\)
                pixel1 = sobelImage.at<uchar>(i-1, j-1);
                pixel2 = sobelImage.at<uchar>(i+1, j+1);
            }
            
            // Perform non-maximum suppression
            if (pixel >= pixel1 && pixel >= pixel2) {
                nmsImage.at<uchar>(i, j) = pixel;
            } else {
                nmsImage.at<uchar>(i, j) = 0;
            }
        }
    }
}

void doubleThreshold(const cv::Mat& sobelImage, cv::Mat& dtImage) {
    static const uchar highThreshold = 130;
    static const uchar lowThreshold = 65;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < sobelImage.rows - 1; i++) {
        for (int j = 1; j < sobelImage.cols - 1; j++) {
            const uchar pixel = sobelImage.at<uchar>(i, j);
            dtImage.at<uchar>(i, j) = pixel > highThreshold ? 255 : (pixel < lowThreshold ? 0 : pixel);
        }
    }
}

void edgeTracking(const cv::Mat& dtImage, cv::Mat& etImage) {
    static const uchar lowThreshold = 50;
    static const uchar highThreshold = 125;

    // First pass: mark strong edges
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < dtImage.rows - 1; i++) {
        for (int j = 1; j < dtImage.cols - 1; j++) {
            etImage.at<uchar>(i, j) = (dtImage.at<uchar>(i, j) >= highThreshold) ? 255 : 0;
        }
    }

    // Second pass: trace edges (cannot be parallelized due to dependencies)
    bool changed;
    do {
        changed = false;
        for (int i = 1; i < dtImage.rows - 1; i++) {
            for (int j = 1; j < dtImage.cols - 1; j++) {
                if (etImage.at<uchar>(i, j) == 255) {
                    for (int x = -1; x <= 1; x++) {
                        for (int y = -1; y <= 1; y++) {
                            if (x == 0 && y == 0) continue;
                            
                            uchar& neighbor = etImage.at<uchar>(i + x, j + y);
                            uchar neighborOriginal = dtImage.at<uchar>(i + x, j + y);
                            
                            if (neighbor != 255 && neighborOriginal >= lowThreshold && 
                                neighborOriginal < highThreshold) {
                                neighbor = 255;
                                changed = true;
                            }
                        }
                    }
                }
            }
        }
    } while (changed);
}

#endif