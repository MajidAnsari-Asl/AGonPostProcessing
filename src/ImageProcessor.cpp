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
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        return {};
    }
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

cv::Mat ImageRectifier::rectifyImage(const cv::Mat& image, double h) {

    // ---------------------------------------------------------------
    // 1. Check images
    // ---------------------------------------------------------------
    if(image.empty())
    {
        std::cerr << "Image sent to rectifyImage2() has issue. \n";
        return cv::Mat();
    }

    if(rectifRefImage.empty())
    {
        std::cerr << "Rectification reference image has issue in rectifyImage2().\n";
        return cv::Mat();
    }

    // ---------------------------------------------------------------
    // 2. Load camera intrinsics
    // ---------------------------------------------------------------
    cv::Mat cameraMatrix, distCoeffs;
    cv::FileStorage fs(strCameraCalibTxtFile, cv::FileStorage::READ);
    if (!fs.isOpened())
	{
		std::cerr << "Error opening file My_camera_calib.txt!" << std::endl;
        return cv::Mat();
	}
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    fs.release();

    // camera undistortion
    cv::Mat undistortedImage;
    undistort(image, undistortedImage, cameraMatrix, distCoeffs);

    // ---------------------------------------------------------------
    // 3. Create ChArUco board and dictionary
    // ---------------------------------------------------------------
    cv::aruco::Dictionary dict =
		cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_250);
    // create a Ptr from the Dictionary object
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::makePtr<cv::aruco::Dictionary>(dict);

	// create charuco board object
	cv::aruco::CharucoBoard charucoBoard =
		cv::aruco::CharucoBoard(patternSize, squareLength, markerLength, dict);
    charucoBoard.setLegacyPattern(true);
    cv::Ptr<cv::aruco::CharucoBoard> board = cv::makePtr<cv::aruco::CharucoBoard>(charucoBoard);


    // ---------------------------------------------------------------
    // 4. Detect markers
    // ---------------------------------------------------------------

    // convert to 8-bit image
    cv::Mat img1_8u;    
    rectifRefImage.convertTo(img1_8u, CV_8U, 1.0/255.0); 

    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners;
    cv::aruco::detectMarkers(img1_8u, dictionary, markerCorners, markerIds);

    if(markerIds.empty())
    {
        std::cerr << "No markers detected.\n";
        return cv::Mat();
    }

    // ---------------------------------------------------------------
    // 5. Refine with ChArUco corners
    // ---------------------------------------------------------------
    std::vector<cv::Point2f> charucoCorners;
    std::vector<int> charucoIds;

    cv::aruco::interpolateCornersCharuco(markerCorners, markerIds,
                                         img1_8u, board,
                                         charucoCorners, charucoIds);

    if(charucoIds.empty())
    {
        std::cerr << "No ChArUco corners detected.\n";
        return cv::Mat();
    }

    // ---------------------------------------------------------------
    // 6. Estimate pose (rvec/tvec) of the board plane
    // ---------------------------------------------------------------
    cv::Vec3d rvec_board, tvec_board;
    bool valid = cv::aruco::estimatePoseCharucoBoard(
                     charucoCorners, charucoIds,
                     board,
                     cameraMatrix, distCoeffs,
                     rvec_board, tvec_board);

    if(!valid)
    {
        std::cerr << "Pose estimation failed.\n";
        return cv::Mat();
    }

    // ---------------------------------------------------------------
    // 7. Compute the sample plane pose (offset along board normal)
    // ---------------------------------------------------------------
    cv::Mat R;
    cv::Rodrigues(rvec_board, R);   // board rotation matrix

    // Board’s local Z-axis (the board normal)
    cv::Mat boardNormalLocal = (cv::Mat_<double>(3,1) << 0, 0, 1);

    // Board normal in camera frame
    cv::Mat n_cam = R * boardNormalLocal;

    // Offset translation
    cv::Mat t_board = (cv::Mat_<double>(3,1) << tvec_board[0], tvec_board[1], tvec_board[2]);
    cv::Mat t_sample = t_board + h * n_cam;

    cv::Vec3d tvec_sample(
        t_sample.at<double>(0),
        t_sample.at<double>(1),
        t_sample.at<double>(2)
    );

    // Sample orientation is the same as board orientation
    cv::Vec3d rvec_sample = rvec_board;

    // ---------------------------------------------------------------
    // 8. Warp sample image into the board frame (top-down view)
    // ---------------------------------------------------------------
    
    // Define output resolution (in pixels per meter or similar)
    float pixelsPerMeter = 15600.0f;  // pixels per meter
    int outWidth  = int(squaresX * squareLength * pixelsPerMeter);
    int outHeight = int(squaresY * squareLength * pixelsPerMeter);

    // Prepare 4 board corners in board coordinates (Z=0)
    std::vector<cv::Point3f> boardCorners3D = {
        {0, 0, 0},
        {squaresX * squareLength, 0, 0},
        {squaresX * squareLength, squaresY * squareLength, 0},
        {0, squaresY * squareLength, 0}
    };

    // Project them using the **sample** pose (corrected for thickness)
    std::vector<cv::Point2f> projectedCorners;
    cv::projectPoints(boardCorners3D, rvec_sample, tvec_sample,
                      cameraMatrix, distCoeffs, projectedCorners);

    // Destination coordinates in the rectified (top-down) image
    std::vector<cv::Point2f> dstCorners = {
        {0, 0},
        {float(outWidth - 1), 0},
        {float(outWidth - 1), float(outHeight - 1)},
        {0, float(outHeight - 1)}
    };

    // Homography between camera image → board-aligned view
    cv::Mat H = cv::getPerspectiveTransform(projectedCorners, dstCorners);

    cv::Mat warped;
    cv::warpPerspective(undistortedImage, warped, H,
                        cv::Size(outWidth, outHeight),
                        cv::INTER_LINEAR,
                        cv::BORDER_CONSTANT);


    return warped.clone();
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

        auto imRec = imageRectifier.rectifyImage(hdrMSImage[channel], -0.0008);  // sample thickness in meters (-0.00152 for 1.52 mm)
        rectifiedHDRMSImage.push_back(imRec);
        auto imRecWR = imageRectifier.rectifyImage(hdrWhiteRefImage[channel]);
        rectifiedHDRWhiteRefImage.push_back(imRecWR);


        // save images scale for visualization only
        cv::Mat displayImage;
        double minVal, maxVal;
        cv::minMaxLoc(imRec, &minVal, &maxVal);
        imRec.convertTo(displayImage, CV_8U, 255.0/(maxVal-minVal), -minVal*255.0/(maxVal-minVal));
        // cv::imwrite("../results/RegImages/im_" + std::to_string(geometry.theta_i)+"_"
        //                               + std::to_string(geometry.phi_i)+"_" 
        //                               + std::to_string(geometry.theta_r)+"_"
        //                               + std::to_string(geometry.phi_r)+"_CH"+std::to_string(channel+2)
        //                               + ".png", displayImage);

    }
    
    //------------------------------------------- ROI selection and patch analysis
    // ROI selection. Done only once for the first geometry
    if (isFirstGeometry)
    {
        roiSelector.selectROICorners(rectifiedHDRMSImage[3]); // Use channel 4 (CH1-CH6) for ROI selection
        roiSelector.calculatePatchROIs(1,1); // 5 rows and 6 columns of patches for NanoTarget
    }
    
    // Patch analysis for each channel for both sample and white reference
    std::vector<std::vector<double>> msRadiance(NUM_EFFECTIVE_MS_CHANNELS);
    std::vector<std::vector<double>> msRadianceWhiteRef(NUM_EFFECTIVE_MS_CHANNELS);

    for (int channel = 0; channel < NUM_EFFECTIVE_MS_CHANNELS; ++channel) {

        auto rectifiedROI = roiSelector.rectifyROI(rectifiedHDRMSImage[channel]);
        roiSelector.calculatePatchAverages(rectifiedROI);
        msRadiance[channel] = roiSelector.getPatchAverages();

        auto rectifiedROIWhiteRef = roiSelector.rectifyROI(rectifiedHDRWhiteRefImage[channel]);
        roiSelector.calculatePatchAverages(rectifiedROIWhiteRef);
        msRadianceWhiteRef[channel] = roiSelector.getPatchAverages();

        if (channel == 3)
        {
            // Visualize and check
            cv::Mat roiViz = roiSelector.visualizeROIs(rectifiedROI);
            cv::imshow("ROI Selection Check", roiViz);
            cv::waitKey(0);
        }
        
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

void MultispectralProcessor::writeSpectralData(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    
    // Write header
    file << "theta_i,phi_i,theta_r,phi_r,patchID";
    for (int i = 1; i <= 6; ++i) {
        file << ",F" << i;
    }
    file << "\n";
    
    // Write data
    for (const auto& geometryData : allData) {
        const auto& geo = geometryData.geometry;
        for (const auto& patch : geometryData.patches) {
            file << geo.theta_i << "," << geo.phi_i << ","
                 << geo.theta_r << "," << geo.phi_r << ","
                 << patch.patchId;
            
            // Write MS channels F1-F6
            for (int ch = 0; ch < 6 && ch < patch.channelValues.size(); ++ch) {
                file << "," << patch.channelValues[ch];
            }
            
            // Fill remaining channels if less than 6
            for (int ch = patch.channelValues.size(); ch < 6; ++ch) {
                file << ",0";
            }
            file << "\n";
        }
    }
    
    file.close();
    std::cout << "Spectral data written to " << filename << std::endl;
}