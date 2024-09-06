#include "neuron.hpp"

#include <stdexcept>

#include "weight_utils.hpp"

Neuron::Neuron(const std::vector<double>& weights,
               activation_functions::ActivationFunction* activation_function) {
    if (weights.size() == 0) {
        throw std::invalid_argument("Neuron must have at least one weight");
    }
    this->weights = weights;
    bias = 0;
    this->activation_function = activation_function;
}

Neuron::Neuron(int num_inputs,
               activation_functions::ActivationFunction* activation_function)
    : Neuron(weight_utils::initializeNeuronWeights(num_inputs),
             activation_function) {}

double Neuron::evaluate(const std::vector<double>& inputs) const {
    double sum = 0;
    for (int i = 0; i < inputs.size(); i++) {
        sum += inputs[i] * weights[i];
    }
    sum += bias;
    return (*activation_function)(sum);
}
