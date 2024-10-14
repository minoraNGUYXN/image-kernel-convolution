#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <opencv2/opencv.hpp>

void applySobel(const cv::Mat grayImage, cv::Mat sobelImage) {
    int kernelX[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}
    };

    int kernelY[3][3] = {
        {-1, 0 , 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int i, j, kx, ky;
    for (i = 1; i < grayImage.rows - 1; i++) {
        for (j = 1; j < grayImage.cols - 1; j++) {
            int gx = 0;
            int gy = 0;
            for (kx = -1; kx < 2; kx++) {
                for (ky = -1; ky < 2; ky++) {
                    gx += grayImage.at<uchar>(i + kx, j + ky) * kernelX[kx + 1][ky + 1];
                    gy += grayImage.at<uchar>(i + kx, j + ky) * kernelY[kx + 1][ky + 1];
                }
            }
            int G = std::sqrt(gx * gx + gy * gy);
            G = G > 255 ? 255 : G;
            sobelImage.at<uchar>(i, j) = G;
        }
    }
}

void applyPrewitt(const cv::Mat grayImage, cv::Mat prewittImage) {
    int kernelX[3][3] = {
        {-1, -1, -1},
        {0, 0, 0},
        {1, 1, 1}
    };

    int kernelY[3][3] = {
        {-1, 0, 1},
        {-1, 0, 1},
        {-1, 0, 1}
    };

    int i, j, kx, ky;
    for (i = 1; i < grayImage.rows - 1; i++) {
        for (j = 1; j < grayImage.cols - 1; j++) {
            int gx = 0;
            int gy = 0;
            for (kx = -1; kx < 2; kx++) {
                for (ky = -1; ky < 2; ky++) {
                    gx += grayImage.at<uchar>(i + kx, j + ky) * kernelX[kx + 1][ky + 1];
                    gy += grayImage.at<uchar>(i + kx, j + ky) * kernelY[kx + 1][ky + 1];
                }
            }
            int G = std::sqrt(gx * gx + gy * gy);
            G = G > 255 ? 255 : G;
            prewittImage.at<uchar>(i, j) = G;
        }
    }
}

void applyLaplacian(const cv::Mat grayImage, cv::Mat laplacianImage) {
    int kernel[3][3] = {
        {0, 1, 0},
        {1, -4, 1},
        {0, 1, 0}
    };

    int i, j, kx, ky;
    for (i = 1; i < grayImage.rows - 1; i++) {
        for (j = 1; j < grayImage.cols - 1; j++) {
            int gx = 0;
            int gy = 0;
            for (kx = -1; kx < 2; kx++) {
                for (ky = -1; ky < 2; ky++) {
                    gx += grayImage.at<uchar>(i + kx, j + ky) * kernel[kx + 1][ky + 1];
                }
            }
            int G = std::abs(gx);
            G = G > 255 ? 255 : G;
            laplacianImage.at<uchar>(i, j) = G;
        }
    }
}

void applyGaussianBlur(const cv::Mat grayImage, cv::Mat gaussianBlurImage) {
    int kernel[5][5] = {
        {1, 4, 7, 4, 1},
        {4, 16, 26, 16, 4},
        {7, 26, 41, 26, 7},
        {4, 16, 26, 16, 4},
        {1, 4, 7, 4, 1}
    };

    int i, j, kx, ky;
    for (i = 2; i < grayImage.rows - 2; i++) {
        for (j = 2; j < grayImage.cols - 2; j++) {
            int gx = 0;
            for (kx = -2; kx < 3; kx++) {
                for (ky = -2; ky < 3; ky++) {
                    gx += grayImage.at<uchar>(i + kx, j + ky) * kernel[kx + 2][ky + 2];
                }
            }
            int G = gx / 273;
            G = G > 255 ? 255 : G;
            gaussianBlurImage.at<uchar>(i, j) = G;
        }
    }
}

void applyCanny(const cv::Mat grayImage, cv::Mat cannyImage) {
    cv::Canny(grayImage, cannyImage, 50, 150);
}

void nonMaximumSuppression(const cv::Mat sobelImage, cv::Mat nmsImage) {
    int i, j;
    for (i = 1; i < sobelImage.rows; i++) {
        for (j = 1; j < sobelImage.cols; j++) {
            for (int x = -1; x <= 1; x++) {
                for (int y = -1; y <= 1; y++) {
                    if (sobelImage.at<uchar>(i, j) <= sobelImage.at<uchar>(i+x, j+y)) {
                        nmsImage.at<uchar>(i, j) = 0;
                        continue;
                    } else {
                        nmsImage.at<uchar>(i, j) = sobelImage.at<uchar>(i, j);
                        continue;
                    }
                }
            }
        }
    }
}

void doubleThreshold(const cv::Mat sobelImage, cv::Mat dtImage) {
    int i, j;
    for (i = 1; i < sobelImage.rows - 1; i++) {
        for (j = 1; j < sobelImage.cols - 1; j++) {
            int G = sobelImage.at<uchar>(i, j);
            dtImage.at<uchar>(i, j) = G > 130 ? 255 : (G < 65 ? 0 : G);
        }
    }
}

void edgeTracking(const cv::Mat dtImage, cv::Mat etImage) {
    // Define thresholds for strong and weak edges
    const uchar lowThreshold = 30;
    const uchar highThreshold = 60;

    // First pass: mark strong edges
    for (int i = 1; i < dtImage.rows - 1; i++) {
        for (int j = 1; j < dtImage.cols - 1; j++) {
            if (dtImage.at<uchar>(i, j) >= highThreshold) {
                etImage.at<uchar>(i, j) = 255;  // Strong edge
            }
        }
    }

    // Second pass: trace edges
    for (int i = 1; i < dtImage.rows - 1; i++) {
        for (int j = 1; j < dtImage.cols - 1; j++) {
            if (etImage.at<uchar>(i, j) == 255) {
                // Check 8-connected neighbors
                for (int x = -1; x <= 1; x++) {
                    for (int y = -1; y <= 1; y++) {
                        if (x == 0 && y == 0) continue;  // Skip the center pixel
                        
                        uchar neighborPixel = dtImage.at<uchar>(i + x, j + y);
                        if (neighborPixel >= lowThreshold && neighborPixel < highThreshold) {
                            etImage.at<uchar>(i + x, j + y) = 255;  // Connect weak edge
                        }
                    }
                }
            }
        }
    }

    // Set non-edge pixels to 0
    for (int i = 0; i < etImage.rows; i++) {
        for (int j = 0; j < etImage.cols; j++) {
            if (etImage.at<uchar>(i, j) != 255) {
                etImage.at<uchar>(i, j) = 0;
            }
        }
    }
}

#endif