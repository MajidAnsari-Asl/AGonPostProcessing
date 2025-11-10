#include "ROISelector.h"
#include <iostream>
#include <numeric>

ROISelector::ROISelector() {}

void ROISelector::selectROICorners(const cv::Mat& image) {
    SelectionData data;
    data.image = image.clone();
    data.corners.resize(4);
    data.completed = false;
    data.currentCorner = 0;
    
    // Create window
    cv::namedWindow("Select Chart Corners (F5 to confirm)");
    cv::setMouseCallback("Select Chart Corners (F5 to confirm)", mouseCallback, &data);
    
    std::cout << "Select 4 corners in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left\n";
    std::cout << "Press F5 when done, ESC to cancel\n";
    
    while (!data.completed) {
        cv::Mat display = data.image.clone();
        
        // Draw instructions
        cv::putText(display, "Select: " + cornerNames[data.currentCorner], 
                   cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(display, "F5: Confirm, ESC: Cancel", 
                   cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 1);
        
        // Draw selected corners
        for (int i = 0; i < data.currentCorner; ++i) {
            cv::circle(display, data.corners[i], 8, cv::Scalar(0, 255, 0), -1);
            cv::putText(display, std::to_string(i+1), 
                       data.corners[i] + cv::Point2f(10, -10), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        }
        
        // Draw connecting lines
        if (data.currentCorner >= 2) {
            for (int i = 1; i < data.currentCorner; ++i) {
                cv::line(display, data.corners[i-1], data.corners[i], 
                        cv::Scalar(255, 0, 0), 2);
            }
        }
        
        cv::imshow("Select Chart Corners (F5 to confirm)", display);
        
        int key = cv::waitKey(10);
        if (key == 27) { // ESC
            cv::destroyWindow("Select Chart Corners (F5 to confirm)");
            return ChartCorners(); // Return empty
        }
        if (key == 65474) { // F5
            if (data.currentCorner == 4) {
                data.completed = true;
            }
        }
    }
    
    cv::destroyWindow("Select Chart Corners (F5 to confirm)");
    
    corners.topLeft = data.corners[0];
    corners.topRight = data.corners[1];
    corners.bottomRight = data.corners[2];
    corners.bottomLeft = data.corners[3];
    
}

void ROISelector::mouseCallback(int event, int x, int y, int flags, void* userdata) {
    SelectionData* data = static_cast<SelectionData*>(userdata);
    
    if (event == cv::EVENT_LBUTTONDOWN && data->currentCorner < 4) {
        cv::Point2f newPoint(x, y);
        
        // Check if clicking near existing corner (for adjustment)
        cv::Point2f closest = findClosestCorner(data->corners, newPoint);
        if (cv::norm(closest - newPoint) < 10.0f && closest != cv::Point2f(-1, -1)) {
            // Find which corner to adjust
            for (int i = 0; i < data->currentCorner; ++i) {
                if (cv::norm(data->corners[i] - closest) < 1.0f) {
                    data->currentCorner = i;
                    break;
                }
            }
        }
        
        data->corners[data->currentCorner] = newPoint;
        data->currentCorner = std::min(data->currentCorner + 1, 4);
    }
}

cv::Point2f ROISelector::findClosestCorner(const std::vector<cv::Point2f>& corners, 
                                          const cv::Point2f& point, float threshold) {
    float minDist = std::numeric_limits<float>::max();
    cv::Point2f closest(-1, -1);
    
    for (const auto& corner : corners) {
        float dist = cv::norm(corner - point);
        if (dist < minDist && dist < threshold) {
            minDist = dist;
            closest = corner;
        }
    }
    
    return closest;
}

cv::Mat ROISelector::rectifyROI(const cv::Mat& image) {
    // Estimate chart dimensions
    float width1 = cv::norm(corners.topRight - corners.topLeft);
    float width2 = cv::norm(corners.bottomRight - corners.bottomLeft);
    float height1 = cv::norm(corners.bottomLeft - corners.topLeft);
    float height2 = cv::norm(corners.bottomRight - corners.topRight);
    
    float avgWidth = (width1 + width2) / 2.0f;
    float avgHeight = (height1 + height2) / 2.0f;
    
    cv::Size outputSize(static_cast<int>(avgWidth), static_cast<int>(avgHeight));
    
    // Get perspective transform
    cv::Mat transform = getPerspectiveTransform(corners, outputSize);
    
    // Warp image
    cv::Mat rectified;
    cv::warpPerspective(image, rectified, transform, outputSize);
    
    return rectified;
}

void ROISelector::calculatePatchROIs(int rows, int cols) {
    // std::vector<PatchROI> patches;
    
    // Create perspective transform to rectangular grid
    std::vector<cv::Point2f> srcCorners = {
        corners.topLeft, corners.topRight, corners.bottomRight, corners.bottomLeft
    };
    
    // Estimate chart dimensions
    float width1 = cv::norm(corners.topRight - corners.topLeft);
    float width2 = cv::norm(corners.bottomRight - corners.bottomLeft);
    float height1 = cv::norm(corners.bottomLeft - corners.topLeft);
    float height2 = cv::norm(corners.bottomRight - corners.topRight);
    
    float avgWidth = (width1 + width2) / 2.0f;
    float avgHeight = (height1 + height2) / 2.0f;
    
    // Calculate patch dimensions
    float patchWidth = avgWidth / cols;
    float patchHeight = avgHeight / rows;
    
    // Create grid points in the warped space
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            PatchROI patch;
            patch.patchId = row * cols + col;
            
            // Calculate patch center in warped coordinates
            float centerX = (col + 0.5f) * patchWidth;
            float centerY = (row + 0.5f) * patchHeight;
            
            // Use 50% of central pixels
            int roiWidth = static_cast<int>(patchWidth * 0.5f);
            int roiHeight = static_cast<int>(patchHeight * 0.5f);
            
            // Create ROI in warped space (centered)
            cv::Rect warpedROI(centerX - roiWidth/2, centerY - roiHeight/2, roiWidth, roiHeight);
            patch.rect = warpedROI;
            patch.center = cv::Point2f(centerX, centerY);
            
            patches.push_back(patch);
        }
    }
    
    // return patches;
}

void ROISelector::calculatePatchAverages(const cv::Mat& image) {
    
    for (auto& patch : patches) {
        // Transform ROI from warped space to image space
        std::vector<cv::Point2f> srcPoints;
        srcPoints.push_back(cv::Point2f(patch.rect.x, patch.rect.y));
        srcPoints.push_back(cv::Point2f(patch.rect.x + patch.rect.width, patch.rect.y));
        srcPoints.push_back(cv::Point2f(patch.rect.x + patch.rect.width, patch.rect.y + patch.rect.height));
        srcPoints.push_back(cv::Point2f(patch.rect.x, patch.rect.y + patch.rect.height));
        
        // Create mask for the patch area
        cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
        std::vector<cv::Point> polyPoints;
        for (const auto& p : srcPoints) {
            polyPoints.push_back(cv::Point(static_cast<int>(p.x), static_cast<int>(p.y)));
        }
        cv::fillConvexPoly(mask, polyPoints, 255);
        
        // Calculate average of masked area
        cv::Scalar meanVal = cv::mean(image, mask);
        patch.averageValue = meanVal;
    }
}

cv::Mat ROISelector::getPerspectiveTransform(const ChartCorners& corners, const cv::Size& outputSize) {
    std::vector<cv::Point2f> src = {
        corners.topLeft, corners.topRight, corners.bottomRight, corners.bottomLeft
    };
    
    std::vector<cv::Point2f> dst = {
        cv::Point2f(0, 0),
        cv::Point2f(outputSize.width, 0),
        cv::Point2f(outputSize.width, outputSize.height),
        cv::Point2f(0, outputSize.height)
    };
    
    return cv::getPerspectiveTransform(src, dst);
}

cv::Mat ROISelector::visualizeROIs(const cv::Mat& image) {
    cv::Mat display = image.clone();
    if (display.channels() == 1) {
        cv::cvtColor(display, display, cv::COLOR_GRAY2BGR);
    }
    
    // Draw patch ROIs
    for (const auto& patch : patches) {
        cv::Rect visRect = patch.rect;
        
        cv::rectangle(display, visRect, cv::Scalar(0, 255, 0), 2);
        cv::putText(display, std::to_string(patch.patchId), 
                   visRect.tl() + cv::Point(5, 15), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
        
        // Draw center point
        cv::circle(display, patch.center, 3, cv::Scalar(255, 0, 0), -1);
    }
    
    return display;
}