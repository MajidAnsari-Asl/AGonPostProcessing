#include "SpectralReconstructor.h"
#include <cmath>
#include <algorithm>

SpectralReconstructor::SpectralReconstructor() {
    
    // Load CIE XYZ color matching functions (simplified - should load from file)
    // CMF_XYZ = cv::Mat::zeros(wavelengths.rows, 3, CV_64F);
    // TODO: Load actual CMF data matching your wavelengths
    
    // trainSecondOrderModel();
}

bool SpectralReconstructor::initialize() {
    
    // ---------------------------------------------------------------
    // Load TSR spectral STUD dataset and calculate reflectances
    // ---------------------------------------------------------------
    spectralSTUDData = readSTUDSpectralData("/Users/majid/Documents/Majid NTNU PC/Publication/Recovery/Code/Codes/Resources/Majid_Munsell_SetUp_2.csv", minWL, maxWL);
    spectralonRefl = readSpectralonRefl("/Users/majid/Documents/Majid NTNU PC/Publication/Recovery/Code/Codes/Resources/C13020406.csv");
    
    // Calculate reflectance factors
    cv::Mat lastRow0 = spectralSTUDData.row(spectralSTUDData.rows - 1);
    spectralSTUDData /= lastRow0; // Divides all rows by last row to get reflectance factors
    cv::multiply(spectralSTUDData, spectralonRefl, spectralSTUDRefl); // Multiplication by spectralon reflectance

    // ---------------------------------------------------------------
    // Load camera multispectral STUD dataset and calculate reflectances
    // ---------------------------------------------------------------
    camResSpectroCam = loadSpectroCamResponses("/Users/majid/Documents/Majid NTNU PC/Publication/Recovery/Code/Codes/SpectroCam/SpectroCam_responses.csv");
    cv::Mat exposureTimes{52, 54, 28, 24, 51, 210};
    camResSpectroCam -= 130.0; // Subtract fixed dark noise

    // normalize By Exposure
    camResSpectroCam /= exposureTimes.t(); // Divides each column by corresponding exposure time
 
    // Divides all rows by last row to get reflectance factors
    cv::Mat lastRow = camResSpectroCam.row(camResSpectroCam.rows - 1);
    msSpectroCamSTUDRefl = camResSpectroCam / lastRow;

    // ---------------------------------------------------------------
    // Load TSR spectral Iluuminant D65 and color matching functions
    // ---------------------------------------------------------------
    D65_Illuminant = loadTruncateIlluminant("/Users/majid/Documents/Majid NTNU PC/Publication/Recovery/Code/Codes/Resources/CIE_std_illum_D65.csv", minWL, maxWL);


    return true;
}

cv::Mat SpectralReconstructor::readSTUDSpectralData(const std::string& filename, int minWL, int maxWL) {
    
    std::vector<std::vector<double>> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return cv::Mat();
    }

    std::string line;
    
    // Skip header
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<double> entry;
        std::string name;
        double Lv, x, y;
        
        // Read index
        std::getline(ss, token, ',');
        
        // Read data name
        std::getline(ss, name, ',');
        
        // Read Lv, x, y
        std::getline(ss, token, ','); Lv = std::stod(token);
        std::getline(ss, token, ','); x = std::stod(token);
        std::getline(ss, token, ','); y = std::stod(token);
        
        // Read spectrum from 380nm, extract 400-700nm range
        int wavelength = 380;
        while (std::getline(ss, token, ',')) {
            if (wavelength >= minWL && wavelength <= maxWL) {
                entry.push_back(std::stod(token));
            }
            wavelength++;
        }
        
        data.push_back(entry);
    }

    // Remove first row (extra Spectralon data)
    if (!data.empty()) {
        data.erase(data.begin()); 
    }

    // Convert to cv::Mat
    cv::Mat result(data.size(), data[0].size(), CV_64F);// [nSamples x nWavelengths]
    for (int i = 0; i < result.rows; ++i) {
        for (int j = 0; j < result.cols; ++j) {
            result.at<double>(i, j) = data[i][j];
        }
    }
    return result;

}

cv::Mat SpectralReconstructor::readSpectralonRefl(const std::string& filename) {
    std::vector<double> spectralonRefl;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return cv::Mat();
    }
    std::string line;
    int rowCount = 0;
    
    while (std::getline(file, line)) {
        rowCount++;
        if (rowCount >= 153 && rowCount <= 453) {
            std::stringstream ss(line);
            std::string cell;
            int colCount = 0;
            
            while (std::getline(ss, cell, ',')) {
                colCount++;
                if (colCount == 2) {
                    spectralonRefl.push_back(std::stod(cell));
                    break;
                }
            }
        }
    }
    
    // Convert to cv::Mat
    cv::Mat result(1, spectralonRefl.size(), CV_64F);// [nSamples(=1) x nWavelengths]
    for (int i = 0; i < result.rows; ++i) {
            result.at<double>(1, i) = spectralonRefl[i];
    }
    return result;
}

cv::Mat SpectralReconstructor::loadSpectroCamResponses(const std::string& filename) {
    std::vector<std::vector<double>> camResSpectroCam;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return cv::Mat();
    }
    std::string line;
    
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        int colIndex = 0;
        
        while (std::getline(ss, cell, ',')) {
            colIndex++;
            if (colIndex >= 2 && colIndex <= 7) { // Columns 2-7
                row.push_back(std::stod(cell));
            }
        }
        camResSpectroCam.push_back(row);
    }
    
    // Remove row 93 (92 in 0-based indexing)
    if (camResSpectroCam.size() > 92) {
        camResSpectroCam.erase(camResSpectroCam.begin() + 92);
    }

    // Convert to cv::Mat
    cv::Mat result(camResSpectroCam.size(), camResSpectroCam[0].size(), CV_64F);// [nSamples x nWavelengths]
    for (int i = 0; i < result.rows; ++i) {
        for (int j = 0; j < result.cols; ++j) {
            result.at<double>(i, j) = camResSpectroCam[i][j];
        }
    }
    return result;
}

cv::Mat SpectralReconstructor::loadTruncateIlluminant(const std::string& filename, int minWL, int maxWL) {
    std::vector<double> illu;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return cv::Mat();
    }
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
            
        std::getline(ss, cell, ','); // Wavelength
        double dWL = std::stod(cell);
        if (dWL >= minWL && dWL <= maxWL) {
            // entry.push_back(dWL);
            // Read X, Y, Z     
            while (std::getline(ss, cell, ','))
                illu.push_back(std::stod(cell));
        }
        else {
            continue;
        }
    } 

    // Convert to cv::Mat
    cv::Mat result(1, illu.size(), CV_64F);// [nSamples(=1) x nWavelengths]
    for (int i = 0; i < result.rows; ++i) {
            result.at<double>(1, i) = illu[i];
    }
    return result;
}

cv::Mat SpectralReconstructor::loadCIECMF(const std::string& filename, int minWL, int maxWL) {
    std::vector<std::vector<double>> cieCMF;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return cv::Mat();
    }
    std::string line;
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> entry;
            
        std::getline(ss, cell, ','); // Wavelength
        double dWL = std::stod(cell);
        if (dWL >= minWL && dWL <= maxWL) {
            // entry.push_back(dWL);
            // Read X, Y, Z     
            while (std::getline(ss, cell, ','))
                entry.push_back(std::stod(cell));
        }
        else {
            continue;
        }
        cieCMF.push_back(entry);
    }
    
    // Convert to cv::Mat
    cv::Mat result(cieCMF.size(), cieCMF[0].size(), CV_64F);// [nSamples x nWavelengths]
    for (int i = 0; i < result.rows; ++i) {
        for (int j = 0; j < result.cols; ++j) {
            result.at<double>(i, j) = cieCMF[i][j];
        }
    }
    return result;
}

void SpectralReconstructor::trainSecondOrderModel() {
    int nSamples = STUD_Responses.rows;
    int nChannels = STUD_Responses.cols;
    int nWavelengths = STUD_Reflectances.cols;
    
    // Create second-order features: [responses, responses^2]
    cv::Mat secondOrderFeatures(nSamples, 2 * nChannels, CV_64F);
    
    for (int i = 0; i < nSamples; ++i) {
        for (int j = 0; j < nChannels; ++j) {
            double val = STUD_Responses.at<double>(i, j);
            secondOrderFeatures.at<double>(i, j) = val;
            secondOrderFeatures.at<double>(i, j + nChannels) = val * val;
        }
    }
    
    // Solve: secondOrderFeatures * ReconstructionMatrix = STUD_Reflectances
    cv::solve(secondOrderFeatures, STUD_Reflectances, ReconstructionMatrix, cv::DECOMP_SVD);
}

cv::Mat SpectralReconstructor::reconstruct(const cv::Mat& cameraResponses) {
    int nSamples = cameraResponses.rows;
    int nChannels = cameraResponses.cols;
    int nWavelengths = STUD_Reflectances.cols;
    
    cv::Mat secondOrderFeatures(nSamples, 2 * nChannels, CV_64F);
    
    for (int i = 0; i < nSamples; ++i) {
        for (int j = 0; j < nChannels; ++j) {
            double val = cameraResponses.at<double>(i, j);
            secondOrderFeatures.at<double>(i, j) = val;
            secondOrderFeatures.at<double>(i, j + nChannels) = val * val;
        }
    }
    
    cv::Mat reconstructed = secondOrderFeatures * ReconstructionMatrix;
    
    // Clip to valid reflectance range [0, 1]
    cv::threshold(reconstructed, reconstructed, 0.0, 0.0, cv::THRESH_TOZERO);
    cv::threshold(reconstructed, reconstructed, 1.0, 1.0, cv::THRESH_TRUNC);
    
    return reconstructed;
}

SpectralReconstructor::EvaluationResults SpectralReconstructor::evaluate(
    const cv::Mat& trueReflectance, const cv::Mat& reconstructedReflectance) {
    
    EvaluationResults results;
    
    // Spectral metrics
    auto spectral = spectralMetrics(trueReflectance, reconstructedReflectance);
    results.rmse = spectral.first;
    results.gfc = spectral.second;
    
    // Colorimetric metrics
    auto colorimetric = colorimetricMetrics(trueReflectance, reconstructedReflectance);
    results.deltaE00 = colorimetric.first;
    results.xyzRMSE = colorimetric.second;
    
    return results;
}

std::pair<double, double> SpectralReconstructor::spectralMetrics(
    const cv::Mat& trueRefl, const cv::Mat& reconRefl) {
    
    cv::Mat diff = trueRefl - reconRefl;
    cv::Mat diffSquared;
    cv::multiply(diff, diff, diffSquared);
    
    double rmse = std::sqrt(cv::mean(diffSquared)[0]);
    
    // Goodness of Fit Coefficient (GFC)
    cv::Mat trueNorm, reconNorm, dotProduct;
    cv::multiply(trueRefl, reconRefl, dotProduct);
    double numerator = cv::sum(dotProduct)[0];
    
    double trueSum = cv::sum(trueRefl.mul(trueRefl))[0];
    double reconSum = cv::sum(reconRefl.mul(reconRefl))[0];
    double denominator = std::sqrt(trueSum * reconSum);
    
    double gfc = (denominator > 0) ? numerator / denominator : 0.0;
    
    return {rmse, gfc};
}

std::pair<double, double> SpectralReconstructor::colorimetricMetrics(
    const cv::Mat& trueRefl, const cv::Mat& reconRefl) {
    
    cv::Mat trueXYZ = reflectanceToXYZ(trueRefl);
    cv::Mat reconXYZ = reflectanceToXYZ(reconRefl);
    
    // XYZ RMSE
    cv::Mat diffXYZ = trueXYZ - reconXYZ;
    cv::Mat diffSquaredXYZ;
    cv::multiply(diffXYZ, diffXYZ, diffSquaredXYZ);
    double xyzRMSE = std::sqrt(cv::mean(diffSquaredXYZ)[0]);
    
    // Delta E 2000 (simplified - full implementation would be more complex)
    cv::Mat trueLab = xyzToLab(trueXYZ);
    cv::Mat reconLab = xyzToLab(reconXYZ);
    double deltaE00 = deltaE2000(trueLab, reconLab);
    
    return {deltaE00, xyzRMSE};
}

cv::Mat SpectralReconstructor::reflectanceToXYZ(const cv::Mat& reflectance) {
    // XYZ = reflectance * D65 * CMF_XYZ (element-wise multiplication and sum)
    cv::Mat weighted = reflectance.mul(D65_Illuminant);
    cv::Mat xyz = weighted * CMF_XYZ;
    return xyz;
}

cv::Mat SpectralReconstructor::xyzToLab(const cv::Mat& xyz) {
    // Simplified XYZ to Lab conversion (full implementation needed)
    cv::Mat lab = xyz.clone(); // Placeholder
    // TODO: Implement full CIELAB conversion
    return lab;
}

double SpectralReconstructor::deltaE2000(const cv::Mat& lab1, const cv::Mat& lab2) {
    // Simplified Delta E 2000 (full implementation would be complex)
    cv::Mat diff = lab1 - lab2;
    cv::Mat diffSquared;
    cv::multiply(diff, diff, diffSquared);
    return std::sqrt(cv::sum(diffSquared)[0]); // Euclidean distance as approximation
}

void SpectralReconstructor::saveModel(const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << "ReconstructionMatrix" << ReconstructionMatrix;
    fs << "Wavelengths" << Wavelengths;
    fs << "D65_Illuminant" << D65_Illuminant;
    fs.release();
}

void SpectralReconstructor::loadModel(const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    fs["ReconstructionMatrix"] >> ReconstructionMatrix;
    fs["Wavelengths"] >> Wavelengths;
    fs["D65_Illuminant"] >> D65_Illuminant;
    fs.release();
}