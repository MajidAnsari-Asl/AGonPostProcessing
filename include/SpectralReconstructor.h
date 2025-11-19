#ifndef SPECTRALRECONSTRUCTOR_H
#define SPECTRALRECONSTRUCTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>


class SpectralReconstructor {
private:
    cv::Mat STUD_Reflectances;    // [nSamples x nWavelengths]
    cv::Mat STUD_Responses;       // [nSamples x nChannels]
    cv::Mat Wavelengths;          // [nWavelengths x 1]
    cv::Mat D65_Illuminant;       // [nWavelengths x 1]
    cv::Mat CMF_XYZ;              // [nWavelengths x 3] - CIE XYZ CMF
    cv::Mat ReconstructionMatrix; // Second-order reconstruction matrix
    
public:

    int minWL = 400;
    int maxWL = 700;
    cv::Mat spectralSTUDData;
    cv::Mat spectralSTUDRefl;
    cv::Mat spectralonRefl;
    cv::Mat camResSpectroCam;
    cv::Mat msSpectroCamSTUDRefl;
    

    SpectralReconstructor();
    cv::Mat readSTUDSpectralData(const std::string& filename, int minWL = 400, int maxWL = 700);
    cv::Mat readSpectralonRefl(const std::string& filename);
    cv::Mat loadSpectroCamResponses(const std::string& filename);
    cv::Mat loadTruncateIlluminant(const std::string& filename, int minWL = 400, int maxWL = 700);
    cv::Mat loadCIECMF(const std::string& filename, int minWL, int maxWL);


    bool initialize();
    
    // Training
    void trainSecondOrderModel();
    
    // Reconstruction
    cv::Mat reconstruct(const cv::Mat& cameraResponses);
    
    // Evaluation metrics
    struct EvaluationResults {
        double rmse;
        double gfc;
        double deltaE00;
        double xyzRMSE;
    };
    
    EvaluationResults evaluate(const cv::Mat& trueReflectance, 
                              const cv::Mat& reconstructedReflectance);
    
    // Spectral metrics
    std::pair<double, double> spectralMetrics(const cv::Mat& trueRefl, 
                                             const cv::Mat& reconRefl);
    
    // Colorimetric metrics
    std::pair<double, double> colorimetricMetrics(const cv::Mat& trueRefl, 
                                                 const cv::Mat& reconRefl);
    
    // Utility functions
    cv::Mat reflectanceToXYZ(const cv::Mat& reflectance);
    cv::Mat xyzToLab(const cv::Mat& xyz);
    double deltaE2000(const cv::Mat& lab1, const cv::Mat& lab2);
    
    // Save/Load model
    void saveModel(const std::string& filename);
    void loadModel(const std::string& filename);
};

#endif