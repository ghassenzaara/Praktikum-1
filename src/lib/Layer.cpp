#include <Layer.h>

#include <MathUtils.h>

#include <cmath>
#include <iostream>
#include <utility>
#include <stdexcept>

namespace dnn {

Layer::Layer(uint64_t id, uint64_t seed, uint64_t layerInputSize, uint64_t layerOutputSize,
             ActivationFunction* activationFunction, std::string  name)
    : id(id), name(std::move(name)), seed(seed), layerInputSize(layerInputSize), layerOutputSize(layerOutputSize),
      activationFunction(activationFunction) {
    accumulatedDWeights.assign(layerOutputSize, std::vector<double>(layerInputSize, 0.0));
    accumulatedDBias.assign(layerOutputSize, 0.0);
}

Layer::~Layer() {
    delete activationFunction;
}

void Layer::forwardPropagation(const std::vector<double>& previousLayerOutput){
    preActivationOutput.assign(layerOutputSize, 0.0);
    activationOutput.assign(layerOutputSize, 0.0);
    for (int i=0; i<layerOutputSize; i++){
        preActivationOutput[i] = dotProduct(weights[i], previousLayerOutput) + bias[i];
        activationOutput[i]= activationFunction->activation(preActivationOutput[i]);
    }
}

void Layer::backwardPropagation(const std::vector<double>& dActivationOutput, const std::vector<double>& previousLayerOutput){
    if (activationFunction==nullptr){
        throw std::logic_error("non-supported activation");
    }
    std::vector<double> dPreActivation(layerOutputSize);
    for (int i=0; i<layerOutputSize; i++){
        dPreActivation[i] = dActivationOutput[i] * activationFunction->activationBackwards(preActivationOutput[i]);
    }
    dWeights.resize(layerOutputSize, std::vector<double>(layerInputSize));
    for (int i=0; i<layerOutputSize; i++){
        for (int j=0; j<layerInputSize; j++){
            dWeights[i][j] = dPreActivation[i] * previousLayerOutput[j];
        }
    }
    dBias = dPreActivation;
    for (int i=0; i < layerOutputSize; i++){
        for (int j=0; j<layerInputSize; j++){
            accumulatedDWeights[i][j]+= dWeights[i][j];
        }
    }
    for(int i=0; i<layerOutputSize; i++){
        accumulatedDBias[i] += dBias[i];
    }
    numProcessedSamples++;
    dPreviousLayerOutput.assign(layerInputSize, 0.0);
    for (int j=0; j<layerInputSize; j++){
        std::vector<double> weightsColumn(layerOutputSize);
        for (int i=0; i<layerOutputSize; i++){
            weightsColumn[i] = weights[i][j];
        }
        dPreviousLayerOutput[j] = dotProduct(weightsColumn, dPreActivation);
    }
}

void Layer::updateParameters(const std::vector<std::vector<double>>& dWeights, const std::vector<double>& dBias, double learningRate){
    for (int i=0; i<layerOutputSize; i++){
        bias[i] -= learningRate * dBias[i];
        for (int j=0; j<layerInputSize;j++){
            weights[i][j]-= learningRate * dWeights[i][j];
        }
    }   
}

void Layer::commitToWeights(double learningRate){
    if (numProcessedSamples==0){
        return;
    }
    for (int i=0; i<layerOutputSize; i++){
        accumulatedDBias[i] = accumulatedDBias[i]/numProcessedSamples;
        for (int j=0; j<layerInputSize; j++){
            accumulatedDWeights[i][j] = accumulatedDWeights[i][j]/numProcessedSamples;
        }
    }
    updateParameters(accumulatedDWeights, accumulatedDBias, learningRate);
    accumulatedDWeights.assign(layerOutputSize, std::vector<double>(layerInputSize, 0.0));
    accumulatedDBias.assign(accumulatedDBias.size(), 0.0);
    numProcessedSamples = 0;
}

void Layer::init() {
    std::mt19937 randomGenerator(seed);

    // Glorot/Xavier uniform initialization:
    // scales the weight range by fan-in/fan-out so pre-activation variance stays roughly O(1)
    // regardless of layer width, instead of growing with layerInputSize the way a fixed
    // [-1, 1] range does.
    const double limit = std::sqrt(6.0 / static_cast<double>(layerInputSize + layerOutputSize));
    std::uniform_real_distribution<double> distribution(-limit, limit);

    weights.assign(layerOutputSize, std::vector<double>(layerInputSize));
    for (auto& row : weights) {
        for (double& value : row) {
            value = distribution(randomGenerator);
        }
    }

    // Zero-initialized bias
    bias.assign(layerOutputSize, 0.0);
}

void Layer::setParameters(std::vector<std::vector<double>> weights, std::vector<double> bias) {
    this->weights = std::move(weights);
    this->bias = std::move(bias);
}

uint64_t Layer::getId() const {
    return id;
}

const std::string& Layer::getName() const {
    return name;
}

const std::vector<double>& Layer::getActivationOutput() const {
    return activationOutput;
}

const std::vector<double>& Layer::getPreActivationOutput() const {
    return preActivationOutput;
}

ActivationFunction* Layer::getActivationFunction() const {
    return activationFunction;
}

const std::vector<std::vector<double>>& Layer::getWeights() const {
    return weights;
}

const std::vector<double>& Layer::getBias() const {
    return bias;
}

const std::vector<double>& Layer::getDPreviousLayerOutput() const {
    return dPreviousLayerOutput;
}

const std::vector<std::vector<double>>& Layer::getDWeights() const {
    return dWeights;
}

const std::vector<double>& Layer::getDBias() const {
    return dBias;
}

}
