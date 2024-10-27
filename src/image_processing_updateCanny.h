// image_processing.h
#ifndef IMAGE_PROCESSING_UPDATE_H
#define IMAGE_PROCESSING_UPDATE_H

#include <opencv2/opencv.hpp>

// Chỉ khai báo prototype của các hàm
void applySobel(const cv::Mat& grayImage, cv::Mat& sobelImage);
void applyPrewitt(const cv::Mat& grayImage, cv::Mat& prewittImage);
void applyLaplacian(const cv::Mat& grayImage, cv::Mat& laplacianImage);
void applyStatisticalFilter(const cv::Mat& grayImage, cv::Mat& filteredImage);
void applyDynamicThreshold(const cv::Mat& sobelImage, cv::Mat& dtImage);

#endif