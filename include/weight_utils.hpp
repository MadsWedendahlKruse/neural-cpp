#pragma once

#include <vector>

/**
 * First dimension is the weight index.
 */
using NeuronWeights = std::vector<double>;
/**
 * First dimension is the neuron index, and the second dimension is the weight
 * index.
 */
using LayerWeights = std::vector<std::vector<double>>;
/**
 * First dimension is the layer index, second dimension is the neuron index, and
 * third dimension is the weight index.
 */
using NetworkWeights = std::vector<std::vector<std::vector<double>>>;

namespace weight_utils {

/**
 * Initialize the weights of a neuron with a given value.
 * @param num_inputs The number of inputs to the neuron.
 * @param value The value to initialize the weights to.
 * @return The weights of the neuron.
 */
NeuronWeights initializeNeuronWeights(int num_inputs, double value);

/**
 * Initialize the weights of a neuron with a random value between -1 and 1.
 * @param num_inputs The number of inputs to the neuron.
 * @return The weights of the neuron.
 */
NeuronWeights initializeNeuronWeights(int num_inputs);

/**
 * Initialize the weights of a layer of neurons with a given value.
 * @param num_inputs The number of inputs to each neuron.
 * @param num_neurons The number of neurons in the layer.
 * @param value The value to initialize the weights to.
 * @return The weights of the layer. The first dimension is the neuron index,
 * and the second dimension is the weight index.
 */
LayerWeights initializeLayerWeights(int num_inputs, int num_neurons,
                                    double value);

/**
 * Initialize the weights of a layer of neurons with a random value between -1
 * and 1.
 * @param num_inputs The number of inputs to each neuron.
 * @param num_neurons The number of neurons in the layer.
 * @return The weights of the layer. The first dimension is the neuron index,
 * and the second dimension is the weight index.
 */
LayerWeights initializeLayerWeights(int num_inputs, int num_neurons);

/**
 * Initialize the weights of a neural network with a given value.
 * @param num_inputs The number of inputs to the network.
 * @param neurons_per_layer The number of neurons in each layer.
 * @param value The value to initialize the weights to.
 * @return The weights of the network. The first dimension is the layer index,
 * the second dimension is the neuron index, and the third dimension is the
 * weight index.
 */
NetworkWeights initializeNetworkWeights(
    int num_inputs, const std::vector<int>& neurons_per_layer, double value);

/**
 * Initialize the weights of a neural network with a random value between -1 and
 * 1.
 * @param num_inputs The number of inputs to the network.
 * @param neurons_per_layer The number of neurons in each layer.
 * @return The weights of the network. The first dimension is the layer index,
 * the second dimension is the neuron index, and the third dimension is the
 * weight index.
 */
NetworkWeights initializeNetworkWeights(
    int num_inputs, const std::vector<int>& neurons_per_layer);

}  // namespace weight_utils