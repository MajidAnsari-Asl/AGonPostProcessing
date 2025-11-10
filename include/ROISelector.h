#ifndef ROISELECTOR_H
#define ROISELECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <array>

class ROISelector {
public:
    struct PatchROI {
        cv::Rect rect;
        cv::Scalar averageValue;
        int patchId;
        cv::Point2f center;
    };
    
    struct ChartCorners {
        cv::Point2f topLeft;
        cv::Point2f topRight;
        cv::Point2f bottomRight;
        cv::Point2f bottomLeft;
    };
    
    ROISelector();

    ChartCorners corners;
    std::vector<PatchROI> patches; // For Nano Target chart (5x6 patches), brown patch at top-left is patchId 0
    
    // Interactive ROI selection
    void selectROICorners(const cv::Mat& image);

    // ROI rectification
    cv::Mat rectifyROI(const cv::Mat& image);
    
    // Calculate patch ROIs from selected corners
    void calculatePatchROIs(int rows = 5, int cols = 6);
    
    // Extract average values from patches
    void calculatePatchAverages(const cv::Mat& image);
    
    // Visualize ROIs on image
    cv::Mat visualizeROIs(const cv::Mat& image);
    
    // Get transformation matrix for the chart
    cv::Mat getPerspectiveTransform(const ChartCorners& corners, const cv::Size& outputSize);
    
private:

    // Mouse callback data
    struct SelectionData {
        cv::Mat image;
        std::vector<cv::Point2f> corners;
        bool completed;
        int currentCorner;
    };
    
    static void mouseCallback(int event, int x, int y, int flags, void* userdata);
    static cv::Point2f findClosestCorner(const std::vector<cv::Point2f>& corners, const cv::Point2f& point, float threshold = 10.0f);
    
    const std::array<std::string, 4> cornerNames = {"Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"};
};

#endif