#include "ImageProcessor.h"
#include "SpectralReconstructor.h"
#include <iostream>

int main() {
    std::cout << "Multispectral Image Processing Pipeline" << std::endl;
    
    // Configuration
    std::string imageFolder = "/Volumes/My Passport/PhDFinalData/CapturedData/Session_20251025_155918";
    std::string whiteRefFolder = "/Volumes/My Passport/PhDFinalData/CapturedData/Session_20251025_213901";
    std::string calibFile = "../data/My_camera_calib.txt";
    std::string analyzeGeometriesFile = "../data/analyzeGeometries.txt";
    std::string outputSpectralDataFile = "../results/NanoTarget/multiSpectralReflectance_Chamelion_45Degree_GON360Patch_2.txt";

    // Set HDR parameters
    HDRParams hdrParams;
    hdrParams.upDCMargin = 200 * (1 << 16) / (1 << 12); // Scale margin for 16-bit images
    hdrParams.downDCMargin = 50 * (1 << 16) / (1 << 12); // Scale margin for 16-bit images
    hdrParams.fixedDarkNoise = 2080; // Fixed dark noise for all images. Equals to 130 in 12-bit scaled to 16-bit
    hdrParams.minDigCount = 655; // 1% of max 16-bit pixel values (2^16/100)
    hdrParams.maxDigCount = (1 << 16) - 1 - hdrParams.fixedDarkNoise - hdrParams.upDCMargin;

    // Initialize processor
    MultispectralProcessor processor;
    if (!processor.initialize(calibFile, hdrParams)) {
        std::cerr << "Failed to initialize processor!" << std::endl;
        return -1;
    }
    
    // Process dataset, get MD reflectance
    try {
        processor.processDataset(imageFolder, whiteRefFolder, analyzeGeometriesFile);
        std::cout << "Processing completed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during processing: " << e.what() << std::endl;
        return -1;
    }

     
    // ---------------------------------------------------------------
    // Save or output the reflectance data
    // ---------------------------------------------------------------
    processor.writeSpectralData(outputSpectralDataFile);

    
    // ---------------------------------------------------------------
    // Spectral reconstruction from multispectral reflectance data
    // ---------------------------------------------------------------
    // Load your STUD dataset

    // // Create reconstructor
    // SpectralReconstructor reconstructor;
    // reconstructor.initialize();


    // // Reconstruct new multispectral data
    // cv::Mat myMSData = loadMyMultispectralData();
    // cv::Mat hyperspectral = reconstructor.reconstruct(myMSData);

    // // Evaluate reconstruction
    // auto results = reconstructor.evaluate(trueReflectance, hyperspectral);
    // std::cout << "RMSE: " << results.rmse << " GFC: " << results.gfc 
    //         << " ΔE00: " << results.deltaE00 << std::endl;
    
    return 0;
}