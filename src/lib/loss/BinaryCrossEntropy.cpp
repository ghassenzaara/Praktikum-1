#include <loss/BinaryCrossEntropy.h>

#include <MathUtils.h>

#include <algorithm>
#include <cmath>

namespace dnn {

std::string BinaryCrossEntropy::getName() const {
    return "binary_cross_entropy";
}

double BinaryCrossEntropy::getCostValue(const std::vector<double>& predictedOutput , const std::vector<double>& actualOutput ) const {
    std::vector<double> P;
    std::vector<double> Q;
    std::vector<double> R;
    const double n = static_cast<double>(predictedOutput.size());
    for (std::size_t i=0; i<predictedOutput.size(); ++i){
        double p = clampProbability(predictedOutput[i]);
        P.push_back(std::log(p));
        Q.push_back(std::log(1-p));
        R.push_back(1-actualOutput[i]);
    }
    double cost = -(dotProduct(actualOutput, P) + dotProduct(R, Q))/n;
    return cost;

}

double BinaryCrossEntropy::getAccuracyValue(const std::vector<double>& predictedOutput , const std::vector<double>& actualOutput ) const {    
    double count = 0.0;
    for (std::size_t i=0; i<predictedOutput.size(); ++i){
        bool a= predictedOutput[i] >0.5;
        if (a && actualOutput[i] == 1.0) {
            count++;
        } else if (!a && actualOutput[i] == 0.0) {
            count++;
        }
    }
    const double n = static_cast<double>(predictedOutput.size());
    double accuracyValue = count / n;
    return accuracyValue;
    
}

std::vector<double> BinaryCrossEntropy::getGradient(const std::vector<double>& predictedOutput , const std::vector<double>& actualOutput ) const {
    std::vector<double> gradient;
    const double n = static_cast<double>(predictedOutput.size());
    for (std::size_t i=0; i<predictedOutput.size(); ++i){
        double p=clampProbability(predictedOutput[i]);
        double grad = (p - actualOutput[i]) / (n * p * (1 - p));
        gradient.push_back(grad);        
    }
    return gradient;

}}
