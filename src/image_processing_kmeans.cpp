#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <omp.h>
#include <cmath>
#include <vector>

// K-Means parameters
const int K = 3;  // Số cụm (segments)

// Hàm áp dụng K-Means cục bộ trên mỗi vùng nhỏ của ảnh
void applyLocalKMeans(const cv::Mat& grayImage, cv::Mat& kmeansImage, int windowSize) {
    // Chia ảnh thành các subimages
    for (int i = 0; i < grayImage.rows; i += windowSize) {
        for (int j = 0; j < grayImage.cols; j += windowSize) {
            // Xác định vùng cục bộ (local region)
            int x_end = std::min(i + windowSize, grayImage.rows);
            int y_end = std::min(j + windowSize, grayImage.cols);
            cv::Rect localRegion(j, i, y_end - j, x_end - i);
            cv::Mat subImage = grayImage(localRegion);

            // Chuẩn bị dữ liệu cho K-Means
            cv::Mat reshapedImage = subImage.reshape(1, subImage.total());
            reshapedImage.convertTo(reshapedImage, CV_32F);

            // Áp dụng K-Means
            cv::Mat labels, centers;
            cv::kmeans(reshapedImage, K, labels, 
                       cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
                       3, cv::KMEANS_PP_CENTERS, centers);

            // Gán lại giá trị cho subImage
            labels = labels.reshape(0, subImage.rows);
            cv::Mat segmented(subImage.size(), subImage.type());
            for (int x = 0; x < labels.rows; ++x) {
                for (int y = 0; y < labels.cols; ++y) {
                    int label = labels.at<int>(x, y);
                    segmented.at<uchar>(x, y) = static_cast<uchar>(centers.at<float>(label, 0));
                }
            }

            // Lưu kết quả của subImage vào kmeansImage
            segmented.copyTo(kmeansImage(localRegion));
        }
    }
}

// Hàm phát hiện cạnh dựa trên sự khác biệt cường độ giữa các phân đoạn k-Means
void detectEdgesKMeans(const cv::Mat& kmeansImage, cv::Mat& edgeImage) {
    edgeImage = cv::Mat::zeros(kmeansImage.size(), CV_8UC1);
    
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < kmeansImage.rows - 1; i++) {
        for (int j = 1; j < kmeansImage.cols - 1; j++) {
            // Lấy giá trị pixel hiện tại và các pixel lân cận
            uchar current = kmeansImage.at<uchar>(i, j);
            uchar right = kmeansImage.at<uchar>(i, j + 1);
            uchar down = kmeansImage.at<uchar>(i + 1, j);

            // Nếu có sự khác biệt giữa các cụm, coi đó là cạnh
            if (std::abs(current - right) > 10 || std::abs(current - down) > 10) {
                edgeImage.at<uchar>(i, j) = 255;
            }
        }
    }
}

int main() {
    // Đọc ảnh xám
    cv::Mat grayImage = cv::imread("input.jpg", cv::IMREAD_GRAYSCALE);
    if (grayImage.empty()) {
        std::cerr << "Failed to load image!" << std::endl;
        return -1;
    }

    // Ảnh kết quả K-Means và phát hiện cạnh
    cv::Mat kmeansImage = cv::Mat::zeros(grayImage.size(), CV_8UC1);
    cv::Mat edgeImage;

    // Áp dụng K-Means cục bộ
    int windowSize = 32;  // Kích thước cửa sổ cục bộ
    applyLocalKMeans(grayImage, kmeansImage, windowSize);

    // Phát hiện cạnh từ K-Means
    detectEdgesKMeans(kmeansImage, edgeImage);

    // Hiển thị và lưu kết quả
    cv::imshow("Original Image", grayImage);
    cv::imshow("K-Means Segmentation", kmeansImage);
    cv::imshow("Edge Detection", edgeImage);
    cv::imwrite("kmeans_edges.jpg", edgeImage);
    
    cv::waitKey(0);
    return 0;
}
