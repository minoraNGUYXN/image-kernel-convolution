#include <opencv2/opencv.hpp>
#include <iostream>
#include "image_processing.h"
#include "opencv2/core.hpp"
#include <omp.h>

int main() {
    omp_set_num_threads(omp_get_max_threads());
    // load the image
    cv::Mat image = cv::imread("./image/image.jpg");
    if (image.empty()) {
        std::cerr << "There is no such image in the directory" << std::endl;
        return -1;
    }

    // turn the loaded image in to grayscale
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    // // create a black image with the size of grayImage and apply Laplacian filter
    // cv::Mat laplacianImage = cv::Mat::zeros(grayImage.size(), CV_8U);
    // applyLaplacian(grayImage, laplacianImage);

    double itime, ftime, exec_time;
    itime = omp_get_wtime();

    // create a black image with the size of grayImage and apply Gaussian blur
    cv::Mat blurredImage = cv::Mat::zeros(grayImage.size(), CV_8U);
    applyGaussianBlur(grayImage, blurredImage);

    // create a black image with the size of grayImage and apply Sobel filter
    cv::Mat sobelImage = cv::Mat::zeros(grayImage.size(), CV_8U);
    applySobel(blurredImage, sobelImage);

    // // create a black image with the size of grayImage and apply Canny filter
    // cv::Mat cannyImage = cv::Mat::zeros(grayImage.size(), CV_8U);
    // applyCanny(blurredImage, cannyImage);

    // create a black image with the size of grayImage and apply non-maximum suppression
    cv::Mat nmsImage = cv::Mat::zeros(grayImage.size(), CV_8U);
    nonMaximumSuppression(sobelImage, nmsImage);

    // // create a black image with the size of grayImage and apply double threshold
    // cv::Mat dtImage = cv::Mat::zeros(grayImage.size(), CV_8U);
    // doubleThreshold(nmsImage, dtImage);
    
    // create a black image with the size of grayImage and apply edge tracking
    // cv::Mat etImage = cv::Mat::zeros(grayImage.size(), CV_8U);
    // edgeTracking(nmsImage, etImage);

    // // create a black image with the size of grayImage and apply Sobel filter then double threshold
    // cv::Mat sobelImage = cv::Mat::zeros(grayImage.size(), CV_8U);
    // applySobel(grayImage, sobelImage);
    // doubleThreshold(sobelImage, sobelImage);

    ftime = omp_get_wtime();
    exec_time = ftime - itime;
    printf("\n\nTime taken is %f", exec_time);

    // Set the display window name
    std::string window_name = "test display";

    // Display the image
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::imshow(window_name, nmsImage);

    // wait for key press indefinitely
    cv::waitKey(0);

    // destroy the display window
    cv::destroyWindow(window_name);

    return 0;
}