#include <activation/ScaledSigmoid.h>

#include <cmath>

namespace dnn {

ScaledSigmoid::ScaledSigmoid(double min, double max) : min(min), max(max){}

double ScaledSigmoid::activation (double x)  {
    return min + (max - min) * (1 / (1 + std::exp(-x)));
}

double ScaledSigmoid::activationBackwards (double x){
    double sig= 1.0 / (1.0 + std::exp(-x));
    return (max - min) * sig * (1 - sig);
}

std::string ScaledSigmoid::getName() const {
    return "scaled_sigmoid";
}

double ScaledSigmoid::getMin() const{
    return min;
}

double ScaledSigmoid::getMax() const{
    return max;
}

}
