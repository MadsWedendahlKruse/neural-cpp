#include "neural_network.hpp"
#include "stdio.h"

int main() {
    printf("Hello World!\n");
    // NeuralNetwork network({{{1, 1}, {1, 1}}, {{1, 1}}},
    NeuralNetwork network({{{1}, {1}}, {{1, 1}}, {{1}}},
                          new activation_functions::Sigmoid);
    std::vector<double> input = {1, 1};
    printf("Input:\n");
    for (int i = 0; i < input.size(); i++) {
        printf("\t%f\n", input[i]);
    }
    std::vector<double> output = network.evaluate(input);
    printf("Output:\n");
    for (int i = 0; i < output.size(); i++) {
        printf("\t%f\n", output[i]);
    }
    return 0;
}
