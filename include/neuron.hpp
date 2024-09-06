#pragma once

#include <vector>

#include "activation_functions.hpp"

/**
 * A neuron is the basic building block of a neural network.
 * It takes a vector of inputs, multiplies them by a vector of weights, adds a
 * bias, and applies an activation function.
 */
class Neuron {
public:
    /**
     * Construct a neuron with a given set of weights and activation function.
     * @param weights The weights of the neuron.
     * @param activation_function The activation function to use.
     */
    Neuron(const std::vector<double>& weights,
           activation_functions::ActivationFunction* activation_function);

    /**
     * Construct a neuron with a given activation function.
     * @param num_inputs The number of inputs to the neuron.
     * @param activation_function The activation function to use.
     */
    Neuron(int num_inputs,
           activation_functions::ActivationFunction* activation_function);

    /**
     * Evaluate the neuron at a given input.
     * @param inputs The inputs to the neuron.
     * @return The output of the neuron.
     */
    double evaluate(const std::vector<double>& inputs) const;

    /**
     * Get the weights of the neuron.
     * @return The weights of the neuron.
     */
    inline const std::vector<double>& getWeights() const { return weights; }

    /**
     * Set the weights of the neuron.
     * @param weights The new weights of the neuron.
     */
    inline void setWeights(const std::vector<double>& weights) {
        this->weights = weights;
    }

    /**
     * Get the bias of the neuron.
     * @return The bias of the neuron.
     */
    inline double getBias() const { return bias; }

    /**
     * Set the bias of the neuron.
     * @param bias The new bias of the neuron.
     */
    inline void setBias(double bias) { this->bias = bias; }

private:
    std::vector<double> weights;
    double bias;
    activation_functions::ActivationFunction* activation_function;
};