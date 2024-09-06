#include "weight_utils.hpp"

#include <algorithm>
#include <random>

namespace weight_utils {

NeuronWeights initializeNeuronWeights(int num_inputs, double value) {
    return NeuronWeights(num_inputs, value);
}

NeuronWeights initializeNeuronWeights(int num_inputs) {
    NeuronWeights weights(num_inputs);
    // Random number between -1 and 1
    std::generate(weights.begin(), weights.end(),
                  []() { return ((double)rand() / RAND_MAX) * 2 - 1; });
    return weights;
}

LayerWeights initializeLayerWeights(int num_inputs, int num_neurons,
                                    double value) {
    LayerWeights weights;
    for (int i = 0; i < num_neurons; i++) {
        weights.push_back(initializeNeuronWeights(num_inputs, value));
    }
    return weights;
}

LayerWeights initializeLayerWeights(int num_inputs, int num_neurons) {
    LayerWeights weights;
    for (int i = 0; i < num_neurons; i++) {
        weights.push_back(initializeNeuronWeights(num_inputs));
    }
    return weights;
}

NetworkWeights initializeNetworkWeights(
    int num_inputs, const std::vector<int>& neurons_per_layer, double value) {
    NetworkWeights weights;
    for (int i = 0; i < neurons_per_layer.size(); i++) {
        weights.push_back(
            initializeLayerWeights(num_inputs, neurons_per_layer[i], value));
        num_inputs = neurons_per_layer[i];
    }
    return weights;
}

NetworkWeights initializeNetworkWeights(
    int num_inputs, const std::vector<int>& neurons_per_layer) {
    NetworkWeights weights;
    for (int i = 0; i < neurons_per_layer.size(); i++) {
        weights.push_back(
            initializeLayerWeights(num_inputs, neurons_per_layer[i]));
        num_inputs = neurons_per_layer[i];
    }
    return weights;
}

}  // namespace weight_utils