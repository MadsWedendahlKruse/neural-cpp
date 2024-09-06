#include "activation_functions.hpp"

namespace activation_functions {

double Identity::operator()(double x) const { return x; }

double Identity::derivative(double x) const { return 1; }

double Sigmoid::operator()(double x) const {
    return 1.0 / (1.0 + std::exp(-x));
}

double Sigmoid::derivative(double x) const {
    return operator()(x) * (1 - operator()(x));
}

double Tanh::operator()(double x) const { return std::tanh(x); }

double Tanh::derivative(double x) const {
    return 1 - std::pow(std::tanh(x), 2);
}

}  // namespace activation_functions