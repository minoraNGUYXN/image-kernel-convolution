#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <opencv2/opencv.hpp>

void applySobel(const cv::Mat grayImage, cv::Mat sobelImage);
void applyPrewitt(const cv::Mat grayImage, cv::Mat prewittImage);
void applyLaplacian(const cv::Mat grayImage, cv::Mat laplacianImage);
void applyGaussianBlur(const cv::Mat grayImage, cv::Mat gaussianBlurImage);
void applyCanny(const cv::Mat grayImage, cv::Mat cannyImage);
void nonMaximumSuppression(const cv::Mat sobelImage, cv::Mat nmsImage);
void doubleThreshold(const cv::Mat sobelImage, cv::Mat dtImage);
void edgeTracking(const cv::Mat dtImage, cv::Mat etImage);

#endif