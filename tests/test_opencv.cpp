#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Create a 256x256 image with 3 channels (BGR)
    cv::Mat image(256, 256, CV_8UC3);

    // Fill the image with a gradient
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            // Create a gradient based on x and y coordinates
            image.at<cv::Vec3b>(y, x) = cv::Vec3b(x, y, 128);
        }
    }

    // Create a window to display the image
    cv::namedWindow("OpenCV Test - Gradient", cv::WINDOW_NORMAL);

    // Display the image
    cv::imshow("OpenCV Test - Gradient", image);

    // Save the image to a file
    cv::imwrite("gradient.png", image);

    std::cout << "Image saved as 'gradient.png'" << std::endl;

    // Wait for a key press
    cv::waitKey(0);

    // Destroy the window
    cv::destroyAllWindows();

    std::cout << "OpenCV is working properly!" << std::endl;

    return 0;
}
