#include "ImageProcessor.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

// MetadataReader implementation
std::vector<ImagingGeometry> MetadataReader::readMetadata(const std::string& filepath) {
    std::vector<ImagingGeometry> geometries;
    std::ifstream file(filepath);
    std::string line;
    
    // Skip header
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        ImagingGeometry geo;
        std::string token;
        
        std::getline(iss, geo.filename, '\t');
        std::getline(iss, token, '\t'); geo.theta_i = std::stod(token);
        std::getline(iss, token, '\t'); geo.phi_i = std::stod(token);
        std::getline(iss, token, '\t'); geo.theta_r = std::stod(token);
        std::getline(iss, token, '\t'); geo.phi_r = std::stod(token);
        std::getline(iss, token, '\t'); geo.motor_x = std::stod(token);
        std::getline(iss, token, '\t'); geo.motor_y = std::stod(token);
        std::getline(iss, token, '\t'); geo.motor_z = std::stod(token);
        std::getline(iss, token, '\t'); geo.motor_a = std::stod(token);
        std::getline(iss, token, '\t'); geo.hdr_bracketing = std::stoi(token);
        std::getline(iss, token, '\t'); geo.filter_iterations = std::stoi(token);
        std::getline(iss, token, '\t'); geo.filter_num = std::stoi(token);
        std::getline(iss, geo.timestamp, '\t');
        std::getline(iss, token, '\t'); geo.exposure_time = std::stod(token);
        
        geometries.push_back(geo);
    }
    
    return geometries;
}

std::vector<ImagingGeometry> MetadataReader::filterGeometries(
    const std::vector<ImagingGeometry>& allGeometries,
    const std::string& analyzeGeometriesFile) {
    
    // Read geometries to analyze
    auto targetGeometries = readMetadata(analyzeGeometriesFile);
    
    std::vector<ImagingGeometry> filtered;
    
    for (const auto& imageGeo : allGeometries) {
        for (const auto& targetGeo : targetGeometries) {
            if (imageGeo.theta_i == targetGeo.theta_i &&
                imageGeo.phi_i == targetGeo.phi_i &&
                imageGeo.theta_r == targetGeo.theta_r &&
                imageGeo.phi_r == targetGeo.phi_r) {
                filtered.push_back(imageGeo);
                break;
            }
        }
    }
    
    return filtered;
}

// CameraCalibration implementation
CameraCalibration::CameraCalibration() : isLoaded(false) {}

bool CameraCalibration::loadCalibration(const std::string& filepath) {
    cv::FileStorage fs(filepath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open calibration file: " << filepath << std::endl;
        return false;
    }
    
    fs["camera_matrix"] >> calibrationMatrix;
    fs.release();
    
    isLoaded = !calibrationMatrix.empty();
    return isLoaded;
}

cv::Mat CameraCalibration::calibrateImage(const cv::Mat& image) const {
    if (!isLoaded || image.empty()) {
        return image.clone();
    }
    
    cv::Mat calibrated;
    // Apply calibration matrix - implementation depends on calibration type
    // This is a placeholder - adjust based on your specific calibration needs
    cv::transform(image, calibrated, calibrationMatrix);
    return calibrated;
}

// HDRConstructor implementation
HDRConstructor::HDRConstructor(const HDRParams& params) : params(params) {}

cv::Mat HDRConstructor::constructHDR(const std::vector<cv::Mat>& images,
                                    const std::vector<double>& exposureTimes,
                                    const std::vector<cv::Mat>& darkImages) {
    
    if (images.size() != exposureTimes.size() || images.size() != darkImages.size()) {
        throw std::invalid_argument("Images, exposure times, and dark images must have same size");
    }
    
    cv::Size imageSize = images[0].size();
    radianceMap = cv::Mat::zeros(imageSize, CV_32FC(images[0].channels()));
    cv::Mat weightSum = cv::Mat::zeros(imageSize, CV_32FC(images[0].channels()));
    
    // Process from longest to shortest exposure
    std::vector<size_t> indices(images.size());
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return exposureTimes[a] > exposureTimes[b];
    });
    
    for (size_t idx : indices) {
        const cv::Mat& img = images[idx];
        const cv::Mat& dark = darkImages[idx];
        double expTime = exposureTimes[idx];
        
        cv::Mat corrected = img - dark;
        cv::Mat radiance;
        corrected.convertTo(radiance, CV_32F);
        radiance /= expTime;
        
        // Create weight mask
        cv::Mat weightMask = createMask(corrected, expTime);
        
        // Accumulate weighted radiance
        cv::Mat weightedRadiance;
        cv::multiply(radiance, weightMask, weightedRadiance);
        radianceMap += weightedRadiance;
        weightSum += weightMask;
    }
    
    // Normalize
    cv::divide(radianceMap, weightSum, radianceMap);
    
    // Handle remaining zeros (if any)
    cv::Mat zeroMask = (weightSum == 0);
    radianceMap.setTo(0, zeroMask);
    
    return radianceMap;
}

cv::Mat HDRConstructor::createMask(const cv::Mat& image, double exposureTime) const {
    cv::Mat mask;
    image.convertTo(mask, CV_32F);
    
    // Simple triangular weighting function
    cv::Mat weight = cv::Mat::zeros(mask.size(), mask.type());
    
    for (int i = 0; i < mask.rows; ++i) {
        for (int j = 0; j < mask.cols; ++j) {
            float pixelValue = mask.at<float>(i, j);
            
            if (pixelValue <= params.minDigCount + params.downDCMargin) {
                weight.at<float>(i, j) = 0.0f;
            } else if (pixelValue >= params.maxDigCount - params.upDCMargin) {
                weight.at<float>(i, j) = 0.0f;
            } else {
                // Triangular weighting - peak at middle of valid range
                float midRange = (params.minDigCount + params.maxDigCount) / 2.0f;
                weight.at<float>(i, j) = 1.0f - std::abs(pixelValue - midRange) / 
                                        ((params.maxDigCount - params.minDigCount) / 2.0f);
            }
        }
    }
    
    return weight;
}

// ImageRectifier implementation (simplified - you'll integrate your existing code)
bool ImageRectifier::calibrateFromImages(const std::vector<std::string>& imagePaths) {
    // Placeholder - integrate your existing ChArUco calibration code here
    std::cout << "Image rectification calibration would be performed here" << std::endl;
    return true;
}

cv::Mat ImageRectifier::rectifyImage(const cv::Mat& image) {
    // Placeholder - integrate your existing rectification code here
    return image.clone();
}

// ROISelector implementation
std::vector<ROISelector::PatchROI> ROISelector::selectROIs(const cv::Mat& image) {
    std::vector<cv::Point2f> corners = selectChartCorners(image);
    return calculatePatchROIs(corners);
}

std::vector<cv::Point2f> ROISelector::selectChartCorners(const cv::Mat& image) {
    // For now, return placeholder corners
    // In real implementation, use mouse callback or automatic detection
    std::vector<cv::Point2f> corners = {
        cv::Point2f(100, 100),   // top-left
        cv::Point2f(500, 100),   // top-right
        cv::Point2f(100, 400),   // bottom-left
        cv::Point2f(500, 400)    // bottom-right
    };
    
    // You would implement interactive corner selection here
    std::cout << "Please implement interactive corner selection" << std::endl;
    
    return corners;
}

std::vector<ROISelector::PatchROI> ROISelector::calculatePatchROIs(const std::vector<cv::Point2f>& corners) {
    std::vector<PatchROI> rois;
    
    if (corners.size() != 4) {
        std::cerr << "Need exactly 4 corners for chart" << std::endl;
        return rois;
    }
    
    // Calculate chart dimensions
    float width = corners[1].x - corners[0].x;
    float height = corners[2].y - corners[0].y;
    
    // 6x5 grid of patches
    int rows = 6, cols = 5;
    float patchWidth = width / cols;
    float patchHeight = height / rows;
    
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            PatchROI roi;
            roi.patchId = row * cols + col;
            
            // Calculate patch center and create ROI (20-30 pixels from center)
            float centerX = corners[0].x + col * patchWidth + patchWidth / 2;
            float centerY = corners[0].y + row * patchHeight + patchHeight / 2;
            
            int roiSize = 25; // 25x25 pixel ROI
            roi.rect = cv::Rect(centerX - roiSize/2, centerY - roiSize/2, roiSize, roiSize);
            
            rois.push_back(roi);
        }
    }
    
    return rois;
}

void ROISelector::visualizeROIs(const cv::Mat& image, const std::vector<PatchROI>& rois) {
    cv::Mat display = image.clone();
    if (display.channels() == 1) {
        cv::cvtColor(display, display, cv::COLOR_GRAY2BGR);
    }
    
    for (const auto& roi : rois) {
        cv::rectangle(display, roi.rect, cv::Scalar(0, 255, 0), 2);
        cv::putText(display, std::to_string(roi.patchId), 
                   roi.rect.tl() + cv::Point(5, 15), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
    
    cv::imshow("ROI Selection", display);
    cv::waitKey(0);
}

// MultispectralProcessor implementation
MultispectralProcessor::MultispectralProcessor() : hdrParams() {}

bool MultispectralProcessor::initialize(const std::string& calibFile, const HDRParams& params) {
    hdrParams = params;
    hdrConstructor = HDRConstructor(hdrParams);
    // return cameraCalib.loadCalibration(calibFile);
    return true; 
}

void MultispectralProcessor::processDataset(const std::string& imageFolder,
                                           const std::string& whiteRefFolder,
                                           const std::string& analyzeGeometriesFile) {
    
    // Read metadata
    std::string imageMetadataPath = imageFolder + "/metadata.txt";
    std::string whiteRefMetadataPath = whiteRefFolder + "/metadata.txt";
    
    auto imageGeometries = MetadataReader::readMetadata(imageMetadataPath);
    auto whiteRefGeometries = MetadataReader::readMetadata(whiteRefMetadataPath);
    auto geometriesToAnalyze = MetadataReader::readMetadata(analyzeGeometriesFile);
    
    // Filter geometries to analyze
    // auto geometriesToAnalyze = MetadataReader::filterGeometries(imageGeometries, analyzeGeometriesFile);
    
    std::cout << "Processing " << geometriesToAnalyze.size() << " imaging geometries..." << std::endl;
    
    for (const auto& geometry : geometriesToAnalyze) {
        processGeometry(geometry, imageFolder, whiteRefFolder);
    }
}

void MultispectralProcessor::processGeometry(const ImagingGeometry& geometry,
                                            const std::string& imageFolder,
                                            const std::string& whiteRefFolder) {
    
    std::cout << "Processing geometry: " << 
                "theta_i"<< geometry.theta_i<< 
                ", phi_i"<< geometry.phi_i<<
                ", theta_r"<< geometry.theta_r<<
                ", phi_r"<< geometry.phi_r<<
                std::endl;
    
    // Load MS images and white reference images for this geometry
    auto msImages = loadChannelImages(imageFolder, geometry);
    auto whiteRefImages = loadChannelImages(whiteRefFolder, geometry);
    
    // TODO: Load dark images based on exposure times. For now, dark noise is set to a fixed value.
    std::vector<cv::Mat> darkImages(msImages.size(),
    cv::Mat(msImages[0].size(), msImages[0].type(), cv::Scalar(hdrParams.fixedDarkNoise)));
    
    // TODO: Implement HDR construction for each channel
    // TODO: Implement image rectification
    // TODO: Implement ROI selection and patch analysis
    
    std::cout << "Completed processing for: " << geometry.filename << std::endl;
}

std::vector<cv::Mat> MultispectralProcessor::loadChannelImages(const std::string& basePath,
                                                              const ImagingGeometry& geometry) {
    std::vector<cv::Mat> images;
    
    // Placeholder - implement actual image loading for channels 2-7
    // This would involve reading 3 captures × 6 channels
    
    for (int capture = 1; capture <= 3; ++capture) {
        for (int channel = 2; channel <= 7; ++channel) {
            std::string imagePath = basePath + "/" + geometry.filename;
            cv::Mat img = cv::imread(imagePath, cv::IMREAD_UNCHANGED);
            if (!img.empty()) {
                images.push_back(img);
            }
        }
    }
    
    return images;
}