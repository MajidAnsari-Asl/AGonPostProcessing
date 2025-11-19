#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco/charuco.hpp>


#include <vector>
#include <string>
#include <map>

#include "ROISelector.h"

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

struct PatchSpectrum {
    int patchId;
    // int geometryId; 
    std::vector<double> channelValues; // [ch0, ch1, ... chN]
};

struct GeometryData {
    ImagingGeometry geometry;
    std::vector<PatchSpectrum> patches;
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
    // cv::Mat getRadianceMap() const { return radianceMap; }
    
private:
    // cv::Mat radianceMap;
    cv::Mat createMask(const cv::Mat& image) const;
};

class ImageRectifier {
public:
    bool calibrateFromImages(const std::vector<std::string>& imagePaths);
    cv::Mat rectifyImage(const cv::Mat& image, double h=0.0);

    cv::Mat rectifRefImage; // Reference image for rectification
    
private:    

    cv::Mat cameraMatrix = cv::Mat_<double>::eye(3, 3);
	cv::Mat distCoeffs   = cv::Mat_<double>::zeros(1, 5);
    std::string strCameraCalibTxtFile = "../data/My_camera_calib.txt";
    

	// ChArUco board specifications
	int squaresX = 10;
	int squaresY = 10;
    cv::Size patternSize = cv::Size(squaresX, squaresY); // ChArUco board dimensions
	float squareLength = 0.01f; // in meters
	float markerLength = 0.006f; // in meters
	int dictionaryId = 6;

    // output path
	std::string outRegPath = "../RegisteredImages/";

};

class MultispectralProcessor {
private:
    CameraCalibration cameraCalib;
    HDRConstructor hdrConstructor;
    ImageRectifier imageRectifier;
    ROISelector roiSelector;
    HDRParams hdrParams;
    std::vector<GeometryData> allData;
    
public:
    MultispectralProcessor();
    bool initialize(const std::string& calibFile, const HDRParams& params = HDRParams());
    void processDataset(const std::string& imageFolder, 
                       const std::string& whiteRefFolder,
                       const std::string& analyzeGeometriesFile);
    void writeSpectralData(const std::string& filename);
    
private:
    std::vector<PatchSpectrum> processGeometry(const ImagingGeometry& geometry,
                        const std::string& imageFolder,
                        const std::string& whiteRefFolder,
                        bool isFirstGeometry = true);
    std::vector<cv::Mat> loadChannelImages(const std::string& basePath, 
                                          const ImagingGeometry& geometry,
                                          std::vector<ImagingGeometry>& msImagesGeometries);
};

#endif