#include <Network.h>

#include <iostream>
#include <stdexcept>

namespace dnn {
Network::Network(std::vector<std::tuple<std::vector<std::vector<double>>, std::vector<double>, ActivationFunction*>> layerParameters, LossFunction*  lossFunction) 
: ownedLossFunction(lossFunction) {
    for (std::size_t i=0; i<layerParameters.size(); i++){
        auto weights = std::get<0>(layerParameters[i]);
        auto bias = std::get<1>(layerParameters[i]);
        auto activationFunction = std::get<2>(layerParameters[i]);
        Layer* layer = new Layer(i, nextSeed, weights[0].size(), weights.size(), activationFunction);
        layer->setParameters(weights, bias);
        layers.push_back(layer);
    }
}

Network::~Network(){
    delete ownedLossFunction;
    for (Layer* layer : layers) {
        delete layer;
    }
}

Network& Network::addLayer(uint64_t inputDim, uint64_t outputDim, ActivationFunction* activationFunction){
    auto layer = new Layer(layers.size(), nextSeed, inputDim, outputDim, activationFunction);
    layer->init();
    layers.push_back(layer);
    nextSeed++;
    return *this;
}

const std::vector<double>& Network::getLayerInput(uint64_t layerId) const{
    if (layerId==0){
        return initialInput;
    }
    return layers[layerId-1]->getActivationOutput();
}

void Network::fullForwardPropagation(const std::vector<double>& input){
    initialInput = input;
    if (layers.empty()){
        output=initialInput;
    }
    else{
        for(Layer* layer: layers){
            const auto& layerInput = getLayerInput(layer->getId());
            layer->forwardPropagation(layerInput);
        }
        output = layers.back()->getActivationOutput();
    }
}

void Network::fullBackwardPropagation(const std::vector<double>& predictedOutput, const std::vector<double>& actualOutput){
    if (ownedLossFunction==nullptr){
        throw std::logic_error("ownedLossFunction is not set");
    }
    auto dACurr = ownedLossFunction->getGradient(predictedOutput, actualOutput);
    for(std::size_t i = layers.size(); i>0; i--){
        auto& layer = layers[i-1];
        layer->backwardPropagation(dACurr, getLayerInput(i-1));
        dACurr = layer->getDPreviousLayerOutput();
    }
}

void Network::trainBatch(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& actualOutputs, std::size_t batchSize,
                     int epochs, double learningRate){
    if (ownedLossFunction==nullptr){
        throw std::logic_error("ownedLossFunction is not set");
    }
    if (inputs.empty()){
        throw std::invalid_argument("trainBatch: inputs is empty");
    }
    if (actualOutputs.empty()){
        throw std::invalid_argument("trainBatch: actualOutputs is empty");
    }
    if (inputs.size()!=actualOutputs.size()){
        throw std::invalid_argument("trainBatch: inputs and actualOutputs have different numbers of samples");
    }
    if (batchSize==0){
        throw std::invalid_argument("trainBatch: batchSize must be greater than 0");
    }
    costHistory.clear();
    accuracyHistory.clear();

    for (int i=0; i<epochs; i++){
        double epochCost=0.0;
        double epochAccuracy=0.0;
        for (std::size_t j=0; j<inputs.size(); j++){
            fullForwardPropagation(inputs[j]);
            epochCost+=ownedLossFunction->getCostValue(output, actualOutputs[j]);
            epochAccuracy+=ownedLossFunction->getAccuracyValue(output, actualOutputs[j]);
            fullBackwardPropagation(output, actualOutputs[j]);
            if (j%batchSize==0){
                for (Layer* layer: layers){
                    layer->commitToWeights(learningRate);
                }
            }
        }
        for (Layer* layer: layers){
            layer->commitToWeights(learningRate);
        }
        costHistory.push_back(epochCost/inputs.size());
        accuracyHistory.push_back(epochAccuracy/inputs.size());
    }  
}

double Network::computeAccuracy(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& actualOutputs){
    if (ownedLossFunction==nullptr){
        throw std::logic_error("ownedLossFunction is not set");
    }
    if (inputs.empty()){
        throw std::invalid_argument("computeAccuracy: inputs is empty");
    }
    if (actualOutputs.empty()){
        throw std::invalid_argument("computeAccuracy: actualOutputs is empty");
    }
    if (inputs.size()!=actualOutputs.size()){
        throw std::invalid_argument("computeAccuracy: inputs and actualOutputs have different numbers of samples");
    }
    double totalCorrect=0;
    for (std::size_t i=0; i<inputs.size(); i++){
        fullForwardPropagation(inputs[i]);
        totalCorrect+=ownedLossFunction->getAccuracyValue(output, actualOutputs[i]);
    }
    return totalCorrect/inputs.size();
}


void Network::setLossFunction(LossFunction* lossFunction) {
    delete this->ownedLossFunction;
    this->ownedLossFunction = lossFunction;
}

const LossFunction* Network::getLossFunction() const {
    return ownedLossFunction;
}

const std::vector<double>& Network::getOutput() const {
    return output;
}

const std::vector<double>& Network::getCostHistory() const {
    return costHistory;
}

const std::vector<double>& Network::getAccuracyHistory() const {
    return accuracyHistory;
}

const std::vector<Layer*>& Network::getLayers() const {
    return layers;
}

void Network::setLayers(std::vector<Layer*> newLayers) {
    for (Layer* layer : layers) {
        delete layer;
    }
    layers = std::move(newLayers);
    initialInput.clear();
    output.clear();
    costHistory.clear();
    accuracyHistory.clear();
}

}
