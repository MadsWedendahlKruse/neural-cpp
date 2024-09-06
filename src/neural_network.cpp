#include "neural_network.hpp"

#include "weight_utils.hpp"

NeuralNetwork::NeuralNetwork(
    const std::vector<std::vector<std::vector<double>>>& weights,
    activation_functions::ActivationFunction* activation_function) {
    this->layers = std::vector<Layer>();
    for (int i = 0; i < weights.size(); i++) {
        layers.push_back(Layer(weights[i], activation_function));
    }
}

NeuralNetwork::NeuralNetwork(
    int num_inputs, const std::vector<int>& neurons_per_layer,
    activation_functions::ActivationFunction* activation_function)
    : NeuralNetwork(
          weight_utils::initializeNetworkWeights(num_inputs, neurons_per_layer),
          activation_function) {}

std::vector<double> NeuralNetwork::evaluate(
    const std::vector<double>& inputs) const {
    std::vector<double> outputs = inputs;
    for (int i = 0; i < layers.size(); i++) {
        outputs = layers[i].evaluate(outputs);
    }
    return outputs;
}
