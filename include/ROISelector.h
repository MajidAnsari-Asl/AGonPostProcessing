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
    
    // Interactive ROI selection
    ChartCorners selectChartCorners(const cv::Mat& image);
    
    // Calculate patch ROIs from selected corners
    std::vector<PatchROI> calculatePatchROIs(const ChartCorners& corners, int rows = 6, int cols = 5);
    
    // Extract average values from patches
    void calculatePatchAverages(const cv::Mat& image, std::vector<PatchROI>& patches);
    
    // Visualize ROIs on image
    cv::Mat visualizeROIs(const cv::Mat& image, const std::vector<PatchROI>& patches, 
                         const ChartCorners& corners = ChartCorners());
    
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