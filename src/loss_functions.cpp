#include "loss_functions.hpp"

namespace loss_functions {

double MeanSquaredError::operator()(const std::vector<double>& predicted,
                                    const std::vector<double>& actual) const {
    double sum = 0;
    for (int i = 0; i < predicted.size(); i++) {
        sum += std::pow(predicted[i] - actual[i], 2);
    }
    return sum / predicted.size();
}

std::vector<double> MeanSquaredError::derivative(
    const std::vector<double>& predicted,
    const std::vector<double>& actual) const {
    std::vector<double> derivative;
    for (int i = 0; i < predicted.size(); i++) {
        derivative.push_back(2 * (predicted[i] - actual[i]));
    }
    return derivative;
}

}  // namespace loss_functions