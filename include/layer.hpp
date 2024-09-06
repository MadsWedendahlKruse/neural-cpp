#pragma once

#include "neuron.hpp"

/**
 * A layer is a collection of neurons.
 */
class Layer {
public:
    /**
     * Construct a layer with a given set of weights and activation function.
     * @param weights The weights of the neurons in the layer.
     * @param activation_function The activation function to use.
     */
    Layer(const std::vector<std::vector<double>>& weights,
          activation_functions::ActivationFunction* activation_function);

    /**
     * Construct a layer with a given number of neurons and activation function.
     * @param num_inputs The number of inputs to each neuron.
     * @param num_neurons The number of neurons in the layer.
     * @param activation_function The activation function to use.
     */
    Layer(int num_inputs, int num_neurons,
          activation_functions::ActivationFunction* activation_function);

    /**
     * Evaluate the layer at a given input.
     * @param inputs The inputs to the layer.
     * @return The outputs of the layer.
     */
    std::vector<double> evaluate(const std::vector<double>& inputs) const;

    /**
     * Get the neurons in the layer.
     * @return The neurons in the layer.
     */
    std::vector<Neuron>& getNeurons() { return neurons; }

private:
    std::vector<Neuron> neurons;
};