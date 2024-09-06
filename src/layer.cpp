#include "layer.hpp"

#include "weight_utils.hpp"

Layer::Layer(const std::vector<std::vector<double>>& weights,
             activation_functions::ActivationFunction* activation_function) {
    this->neurons = std::vector<Neuron>();
    for (int i = 0; i < weights.size(); i++) {
        neurons.push_back(Neuron(weights[i], activation_function));
    }
}

Layer::Layer(int num_inputs, int num_neurons,
             activation_functions::ActivationFunction* activation_function)
    : Layer(weight_utils::initializeLayerWeights(num_inputs, num_neurons),
            activation_function) {}

std::vector<double> Layer::evaluate(const std::vector<double>& inputs) const {
    std::vector<double> outputs;
    for (int i = 0; i < neurons.size(); i++) {
        outputs.push_back(neurons[i].evaluate(inputs));
    }
    return outputs;
}
