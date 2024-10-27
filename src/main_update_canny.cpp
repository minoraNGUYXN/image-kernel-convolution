#include <opencv2/opencv.hpp>
#include <iostream>
#include "image_processing_updateCanny.h"
#include "opencv2/core.hpp"
#include <omp.h>

int main() {
    // Thiết lập số luồng tối đa cho OpenMP
    omp_set_num_threads(omp_get_max_threads());

    // Đọc ảnh đầu vào
    cv::Mat image = cv::imread("./image/image.jpg");
    if (image.empty()) {
        std::cerr << "There is no such image in the directory" << std::endl;
        return -1;
    }

    // Chuyển ảnh sang ảnh xám
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    double itime, ftime, exec_time;
    itime = omp_get_wtime();

    // Áp dụng các bộ lọc
    cv::Mat sobelImage = cv::Mat::zeros(grayImage.size(), CV_8U);
    applySobel(grayImage, sobelImage);

    // cv::Mat prewittImage = cv::Mat::zeros(grayImage.size(), CV_8U);
    // applyPrewitt(grayImage, prewittImage);

    // cv::Mat laplacianImage = cv::Mat::zeros(grayImage.size(), CV_8U);
    // applyLaplacian(grayImage, laplacianImage);
    
    cv::Mat filteredImage = cv::Mat::zeros(grayImage.size(), CV_8U);
    applyStatisticalFilter(grayImage, filteredImage);

    cv::Mat dtImage = cv::Mat::zeros(grayImage.size(), CV_8U);
    applyDynamicThreshold(sobelImage, dtImage);
    

    ftime = omp_get_wtime();
    exec_time = ftime - itime;
    printf("\n\nTime taken is %f", exec_time);

    // Hiển thị kết quả
    cv::namedWindow("Edge Detection Results", cv::WINDOW_NORMAL);
    cv::imshow("Edge Detection Results", dtImage); // hoặc prewittImage hoặc laplacianImage

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}