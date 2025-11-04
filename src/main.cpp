#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::cout << "Majid said OpenCV version: "<< std::endl;
    cv::Mat image = cv::Mat::zeros(400, 400, CV_8UC3);
    cv::circle(image, cv::Point(200, 200), 50, cv::Scalar(0, 255, 0), -1);
    cv::imshow("Test", image);
    cv::waitKey(0);
    return 0;
}