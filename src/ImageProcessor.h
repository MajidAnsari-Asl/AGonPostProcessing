#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <map>

#define NUM_ALL_MS_CHANNELS 8
#define NUM_EFFECTIVE_MS_CHANNELS 6
#define NUM_HDR_BRACKETS 3

enum class BracketStatus { FIRST_BRACKET, MIDDLE_BRACKET, LAST_BRACKET };

struct ImagingGeometry {
    double theta_i;
    double phi_i;
    double theta_r;
    double phi_r;
    double motor_x, motor_y, motor_z, motor_a;
    int hdr_bracketing;
    int filter_iterations;
    int filter_num;
    std::string timestamp;
    double exposure_time;
    std::string filename;
};

struct HDRParams {
    double upDCMargin = 200 * (1 << 16) / (1 << 12); // Scale margin for 16-bit images
    double downDCMargin = 50 * (1 << 16) / (1 << 12); // Scale margin for 16-bit images
    double fixedDarkNoise = 2080; // Fixed dark noise for all images. Equals to 130 in 12-bit scaled to 16-bit
    double minDigCount = 655; // 1% of max 16-bit pixel values (2^16/100)
    double maxDigCount = (1 << 16) - 1 - fixedDarkNoise - upDCMargin;
};

class MetadataReader {
public:
    static std::vector<ImagingGeometry> readMetadata(const std::string& filepath);
    static std::vector<ImagingGeometry> filterGeometries(
        const std::string& basePath,
        const ImagingGeometry& geometry);  
};

class CameraCalibration {
private:
    cv::Mat calibrationMatrix;
    bool isLoaded;
    
public:
    CameraCalibration();
    bool loadCalibration(const std::string& filepath);
    cv::Mat getCalibrationMatrix() const { return calibrationMatrix; }
    cv::Mat calibrateImage(const cv::Mat& image) const;
};

class HDRConstructor {
private:
    HDRParams params;
    
public:
    HDRConstructor(const HDRParams& params = HDRParams());
    cv::Mat constructHDR(const std::vector<cv::Mat>& images, 
                        const std::vector<double>& exposureTimes,
                        const std::vector<cv::Mat>& darkImages);
    cv::Mat getRadianceMap() const { return radianceMap; }
    
private:
    cv::Mat radianceMap;
    cv::Mat createMask(const cv::Mat& image) const;
};

class ImageRectifier {
public:
    bool calibrateFromImages(const std::vector<std::string>& imagePaths);
    cv::Mat rectifyImage(const cv::Mat& image);
    
private:
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    cv::Size patternSize = cv::Size(8, 6); // ChArUco board dimensions
    float squareLength = 0.04f; // in meters
    float markerLength = 0.02f; // in meters
};

class ROISelector {
public:
    struct PatchROI {
        cv::Rect rect;
        cv::Scalar averageValue;
        int patchId;
    };
    
    std::vector<PatchROI> selectROIs(const cv::Mat& image);
    void visualizeROIs(const cv::Mat& image, const std::vector<PatchROI>& rois);
    
private:
    std::vector<cv::Point2f> selectChartCorners(const cv::Mat& image);
    std::vector<PatchROI> calculatePatchROIs(const std::vector<cv::Point2f>& corners);
};

class MultispectralProcessor {
private:
    CameraCalibration cameraCalib;
    HDRConstructor hdrConstructor;
    ImageRectifier imageRectifier;
    ROISelector roiSelector;
    HDRParams hdrParams;
    
public:
    MultispectralProcessor();
    bool initialize(const std::string& calibFile, const HDRParams& params = HDRParams());
    void processDataset(const std::string& imageFolder, 
                       const std::string& whiteRefFolder,
                       const std::string& analyzeGeometriesFile);
    
private:
    void processGeometry(const ImagingGeometry& geometry,
                        const std::string& imageFolder,
                        const std::string& whiteRefFolder);
    std::vector<cv::Mat> loadChannelImages(const std::string& basePath, 
                                          const ImagingGeometry& geometry,
                                          std::vector<ImagingGeometry>& msImagesGeometries);
};

#endif