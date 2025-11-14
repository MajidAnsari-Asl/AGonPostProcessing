#include "ImageProcessor.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <sstream>

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
    const std::string& basePath,
    const ImagingGeometry& geometry) {
 
    // Read metadata
    std::string imageMetadataPath = basePath + "/metadata.txt";
    
    auto allGeometries = MetadataReader::readMetadata(imageMetadataPath);
    
    std::vector<ImagingGeometry> filtered;
    
    for (const auto& imageGeo : allGeometries) {
        if (imageGeo.theta_i == geometry.theta_i &&
            imageGeo.phi_i == geometry.phi_i &&
            imageGeo.theta_r == geometry.theta_r &&
            imageGeo.phi_r == geometry.phi_r) {
            filtered.push_back(imageGeo);
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
    cv::Mat radianceMap = cv::Mat::zeros(imageSize, CV_32FC(images[0].channels()));
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
        cv::Mat weightMask = createMask(corrected);
        
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

cv::Mat HDRConstructor::createMask(const cv::Mat& image) const {
    cv::Mat mask;
    image.convertTo(mask, CV_32F);
    
    // Simple triangular weighting function
    cv::Mat weight = cv::Mat::zeros(mask.size(), mask.type());
    
    for (int i = 0; i < mask.rows; ++i) {
        for (int j = 0; j < mask.cols; ++j) {
            float pixelValue = mask.at<float>(i, j);
            
            if (pixelValue <= params.minDigCount) {
                weight.at<float>(i, j) = 0.0f;
            } else if (pixelValue >= params.maxDigCount) {
                weight.at<float>(i, j) = 0.0f;
            } 
            else {
                // Triangular weighting - peak at middle of valid range
                float midRange = (params.minDigCount + params.maxDigCount) / 2.0f;
                weight.at<float>(i, j) = 1.0f - std::abs(pixelValue - midRange) / 
                                        ((params.maxDigCount - params.minDigCount) / 2.0f);

                // for now, the pixel values are kept without weighting
                // weight.at<float>(i, j) = 1.0f;
            }
        }
    }
    
    return weight;
}

// ImageRectifier implementation
bool ImageRectifier::calibrateFromImages(const std::vector<std::string>& imagePaths) {
    std::cout << "Image rectification calibration would be performed here" << std::endl;
    return true;
}

cv::Mat ImageRectifier::rectifyImage(const cv::Mat& image) {

	cv::aruco::Dictionary dict =
		cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_250);
    // create a Ptr from the Dictionary object
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::makePtr<cv::aruco::Dictionary>(dict);

	// create charuco board object
	cv::aruco::CharucoBoard charucoBoard =
		cv::aruco::CharucoBoard(patternSize, squareLength, markerLength, dict);
    charucoBoard.setLegacyPattern(true);
    cv::Ptr<cv::aruco::CharucoBoard> charucoBoardPtr = cv::makePtr<cv::aruco::CharucoBoard>(charucoBoard);

    // read the image of ChArUco board
	cv::Mat imageCharucoBoard;
	imageCharucoBoard = cv::imread("../data/MyChArUco_10by10_1cm_0.6cm_20230303_300dpi.tiff");

	// read camera calibration matrices
	// help from: https://docs.opencv.org/4.x/dd/d74/tutorial_file_input_output_with_xml_yml.html
	cv::FileStorage fs(strCameraCalibTxtFile, cv::FileStorage::READ);
	if (!fs.isOpened())
	{
		std::cerr << "Error opening file My_camera_calib.txt!" << std::endl;
        return cv::Mat();
	}
	fs["camera_matrix"] >> cameraMatrix;                                      // Read cv::Mat
	fs["distortion_coefficients"] >> distCoeffs;
	fs.release();                                       // explicit close

    cv::Mat undistortedImage;
    undistort(image, undistortedImage, cameraMatrix, distCoeffs);// // 4th(j=3) filter camera matrix is used for all channels. could be channelwise later

    // image registration using ChArUco board;
    // Load the images
    // cv::Mat img1 = undistortedImage;
    // cv::imshow("rectifRefImage", rectifRefImage); cv::waitKey(0);

    cv::Mat img1 = rectifRefImage;
    cv::Mat img2 = imageCharucoBoard;

    // Detect the ChArUco board corners in the first image
    std::vector<std::vector<cv::Point2f>> corners1;
    std::vector<int> ids1;
    std::vector<cv::Point2f> charucoCorners1;
    std::vector<int> charucoIds1;

    cv::Mat img1_8u;
    
    img1.convertTo(img1_8u, CV_8U, 1.0/255.0); // convert to 8-bit image

    cv::aruco::detectMarkers(img1_8u, dictionary, corners1, ids1);
    if (ids1.size() >= 4) // 4 corners should be detected at least, otherwise returns empty matrix
    {

        cv::aruco::interpolateCornersCharuco(corners1, ids1, img1_8u, charucoBoardPtr, charucoCorners1, charucoIds1);
        if (charucoIds1.size() > 0)
        {
            cv::aruco::drawDetectedCornersCharuco(img1_8u, charucoCorners1, charucoIds1);
        }
    }
    else
    {
        std::cerr << "Error detecting ChArUco corners in the image!" << std::endl;
        return cv::Mat();
    }

    // Detect the ChArUco board corners in the second image////////////////////////// probably can be done once as it is fixed in all loop itterations, CORRECT LATER
    std::vector<std::vector<cv::Point2f>> corners2;
    std::vector<int> ids2;
    std::vector<cv::Point2f> charucoCorners2;
    std::vector<int> charucoIds2;

    cv::aruco::detectMarkers(img2, dictionary, corners2, ids2);
    if (ids2.size() >= 4)
    {

        cv::aruco::interpolateCornersCharuco(corners2, ids2, img2, charucoBoardPtr, charucoCorners2, charucoIds2);
        if (charucoIds2.size() > 0)
        {
            cv::aruco::drawDetectedCornersCharuco(img2, charucoCorners2, charucoIds2);
        }
    }
    else
    {
        std::cerr << "Error detecting ChArUco corners in the ChArUco reference image!" << std::endl;
        return cv::Mat();
    }

    // Find the homography matrix
    cv::Mat homography;
    std::vector<cv::Point2f> srcPoints, dstPoints;
    for (int i = 0; i < charucoIds1.size(); i++)
    {
        for (int j = 0; j < charucoIds2.size(); j++)
        {
            if (charucoIds1[i] == charucoIds2[j])
            {
                srcPoints.push_back(charucoCorners1[i]);
                dstPoints.push_back(charucoCorners2[j]);
                break;
            }
        }
    }
    homography = findHomography(srcPoints, dstPoints);



    cv::Mat registeredImg;
    cv::Size registeredImgSize = cv::Size(img2.cols, img2.rows);
    warpPerspective(undistortedImage, registeredImg, homography, img2.size());


    //// Display the results
    //namedWindow("img1", WINDOW_NORMAL);
    //imshow("img1", img1);
    //namedWindow("img2", WINDOW_NORMAL);
    //imshow("img2", img2);
    //namedWindow("registeredImg", WINDOW_NORMAL);
    //imshow("registeredImg", registeredImg);
    //waitKey(0);

    // wrtie the registered image
    // std::string outImPath = outRegPath + strPostImageNames[i][j];
    // cv::imwrite(outImPath, registeredImg);
    

    return registeredImg.clone();
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
    
    // Read imaging geometries to be analyzed
    auto geometriesToAnalyze = MetadataReader::readMetadata(analyzeGeometriesFile);
    
    std::cout << "Processing " << geometriesToAnalyze.size() << " imaging geometries..." << std::endl;
    
    bool firstGeometry = true;

    for (const auto& geometry : geometriesToAnalyze) {
        GeometryData data;
        data.geometry = geometry;
        data.patches = processGeometry(geometry, imageFolder, whiteRefFolder, firstGeometry);
        firstGeometry = false;
        allData.push_back(data);
    }
}

std::vector<PatchSpectrum> MultispectralProcessor::processGeometry(const ImagingGeometry& geometry,
                                            const std::string& imageFolder,
                                            const std::string& whiteRefFolder,
                                            bool isFirstGeometry) {
    
    std::cout << "Processing geometry: " << 
                "theta_i="<< geometry.theta_i<< 
                ", phi_i="<< geometry.phi_i<<
                ", theta_r="<< geometry.theta_r<<
                ", phi_r="<< geometry.phi_r<<
                std::endl;
    
    // Load MS images and white reference images for this geometry
    std::vector<ImagingGeometry> msImagesGeometries;
    auto msImages = loadChannelImages(imageFolder, geometry, msImagesGeometries);
    auto whiteRefImages = loadChannelImages(whiteRefFolder, geometry, msImagesGeometries);
    
    // TODO: Load dark images based on exposure times. For now, dark noise is set to a fixed value.
    std::vector<cv::Mat> darkImages(msImages.size(),
    cv::Mat(msImages[0].size(), msImages[0].type(), cv::Scalar(hdrParams.fixedDarkNoise)));  

    imageRectifier.rectifRefImage = msImages[3].clone(); // take channel 4 in first bracket image as reference for rectification

    //------------------------------------------- HDR construction channel-wise
    std::vector<cv::Mat> hdrMSImage;
    std::vector<cv::Mat> hdrWhiteRefImage;

    for (int channel = 0; channel < NUM_EFFECTIVE_MS_CHANNELS; ++channel) {
        std::vector<cv::Mat> singleCHImages;
        std::vector<cv::Mat> singleCHWhiteRefImages;
        std::vector<cv::Mat> singleCHDarkImages;
        std::vector<double> exposureTimes;

        for (int capture = 0; capture < NUM_HDR_BRACKETS; ++capture) {
            int idx = capture * NUM_EFFECTIVE_MS_CHANNELS + channel;
            singleCHImages.push_back(msImages[idx]);
            singleCHWhiteRefImages.push_back(whiteRefImages[idx]);
            exposureTimes.push_back(msImagesGeometries[idx].exposure_time);
            singleCHDarkImages.push_back(darkImages[idx]);

        }

        //construct HDR for this channel
        cv::Mat img = hdrConstructor.constructHDR(singleCHImages, exposureTimes, singleCHDarkImages);
        if (!img.empty()) {
            hdrMSImage.push_back(img);
        }
        //construct HDR for this channel's white reference
        cv::Mat img2 = hdrConstructor.constructHDR(singleCHWhiteRefImages, exposureTimes, singleCHDarkImages);
        if (!img2.empty()) {
            hdrWhiteRefImage.push_back(img2);
        }

    }

    //------------------------------------------- Image rectification
    std::vector<cv::Mat> rectifiedHDRMSImage;
    std::vector<cv::Mat> rectifiedHDRWhiteRefImage;

    for (int channel = 0; channel < NUM_EFFECTIVE_MS_CHANNELS; ++channel) {

        auto imRec = imageRectifier.rectifyImage(hdrMSImage[channel]);
        rectifiedHDRMSImage.push_back(imRec);
        auto imRecWR = imageRectifier.rectifyImage(hdrWhiteRefImage[channel]);
        rectifiedHDRWhiteRefImage.push_back(imRecWR);


        // save images scale for visualization only
        cv::Mat displayImage;
        double minVal, maxVal;
        cv::minMaxLoc(imRec, &minVal, &maxVal);
        imRec.convertTo(displayImage, CV_8U, 255.0/(maxVal-minVal), -minVal*255.0/(maxVal-minVal));
        cv::imwrite("../results/RegImages/im_" + std::to_string(geometry.theta_i)+"_"
                                      + std::to_string(geometry.phi_i)+"_" 
                                      + std::to_string(geometry.theta_r)+"_"
                                      + std::to_string(geometry.phi_r)+"_CH"+std::to_string(channel+2)
                                      + ".png", displayImage);

    }
    
    //------------------------------------------- ROI selection and patch analysis
    // ROI selection. Done only once for the first geometry
    if (isFirstGeometry)
    {
        roiSelector.selectROICorners(rectifiedHDRMSImage[3]); // Use channel 4 (CH1-CH6) for ROI selection
        roiSelector.calculatePatchROIs(5,6); // 5 rows and 6 columns of patches for NanoTarget
    }
    
    // Patch analysis for each channel for both sample and white reference
    std::vector<std::vector<double>> msRadiance(NUM_EFFECTIVE_MS_CHANNELS);
    std::vector<std::vector<double>> msRadianceWhiteRef(NUM_EFFECTIVE_MS_CHANNELS);

    for (int channel = 0; channel < NUM_EFFECTIVE_MS_CHANNELS; ++channel) {

        auto rectifiedROI = roiSelector.rectifyROI(rectifiedHDRMSImage[channel]);
        roiSelector.calculatePatchAverages(rectifiedROI);
        msRadiance[channel] = roiSelector.getPatchAverages();

        if (channel == 3)
        {
            // Visualize and check
            cv::Mat roiViz = roiSelector.visualizeROIs(rectifiedROI);
            cv::imshow("ROI Selection Check", roiViz);
            cv::waitKey(0);
        }

        auto rectifiedROIWhiteRef = roiSelector.rectifyROI(rectifiedHDRWhiteRefImage[channel]);
        roiSelector.calculatePatchAverages(rectifiedROIWhiteRef);
        msRadianceWhiteRef[channel] = roiSelector.getPatchAverages();
    }

    //------------------------------------------- MS Reflectance calculation
    std::vector<PatchSpectrum> patches;
    for (size_t patch = 0; patch < msRadiance[0].size(); ++patch) {
        PatchSpectrum ps;
        ps.patchId = patch;
        for (size_t channel = 0; channel < msRadiance.size(); ++channel) {
            // Reflectance = Sample / WhiteReference
            double refl = msRadiance[channel][patch] / msRadianceWhiteRef[channel][patch];
            ps.channelValues.push_back(refl);
        }
        patches.push_back(ps);
    }
    
    std::cout << "Completed processing for: " << 
                "theta_i="<< geometry.theta_i<< 
                ", phi_i="<< geometry.phi_i<<
                ", theta_r="<< geometry.theta_r<<
                ", phi_r="<< geometry.phi_r<<
                std::endl;

    return patches;
}

std::vector<cv::Mat> MultispectralProcessor::loadChannelImages(const std::string& basePath,
                                                              const ImagingGeometry& geometry,
                                                              std::vector<ImagingGeometry>& msImagesGeometries) {
    std::vector<cv::Mat> images;
    std::vector<ImagingGeometry> msImGeos; 
    
    // Filter geometries to analyze
    auto geometriesToAnalyze = MetadataReader::filterGeometries(basePath, geometry);

    // Sort to load from lower to higher image name numbering
    std::vector<size_t> indices(geometriesToAnalyze.size());
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return std::stoi(geometriesToAnalyze[a].filename) < std::stoi(geometriesToAnalyze[b].filename);
    });

    for (size_t idx : indices) {
        // Skip channels 1 and 8
        if (geometriesToAnalyze[idx].filter_num == 1 || geometriesToAnalyze[idx].filter_num == 8) {
        continue;  
        }   
        // Image loading for channels 2-7
        std::string imagePath = basePath + "/" + geometriesToAnalyze[idx].filename;
        cv::Mat img = cv::imread(imagePath, cv::IMREAD_UNCHANGED);
        if (!img.empty()) {
            images.push_back(img);
            msImGeos.push_back(geometriesToAnalyze[idx]);
        }
        
        // TODO: Implement combining multiple exposures per channel if needed
    }   

    if (msImagesGeometries.empty()) {
            msImagesGeometries= msImGeos;
        }
    
    return images;
}