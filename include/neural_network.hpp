#pragma once

#include "layer.hpp"

/**
 * A neural network is a collection of layers of neurons.
 */
class NeuralNetwork {
public:
    /**
     * Construct a network with a given set of weights and activation function.
     * @param weights The weights of the network. The first dimension is the
     * layer index, the second dimension is the neuron index, and the third
     * dimension is the weight index.
     * @param activation_function The activation function to use.
     */
    NeuralNetwork(
        const std::vector<std::vector<std::vector<double>>>& weights,
        activation_functions::ActivationFunction* activation_function);

    /**
     * Construct a network with a given number of layers and neurons per layer.
     * @param num_inputs The number of inputs to the network.
     * @param neurons_per_layer The number of neurons in each layer.
     * @param activation_function The activation function to use. TODO: Multiple
     * activation functions?
     */
    NeuralNetwork(
        int num_inputs, const std::vector<int>& neurons_per_layer,
        activation_functions::ActivationFunction* activation_function);

    /**
     * Evaluate the network at a given input.
     * @param inputs The inputs to the network.
     * @return The outputs of the network.
     */
    std::vector<double> evaluate(const std::vector<double>& inputs) const;

    /**
     * Get the layers of the network.
     * @return The layers of the network.
     */
    std::vector<Layer>& getLayers() { return layers; }

private:
    std::vector<Layer> layers;
};