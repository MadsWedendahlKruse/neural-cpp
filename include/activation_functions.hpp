#pragma once

#include <cmath>

/**
 * Activation functions are used to introduce non-linearity into the neural
 * network. This is important because if we only use linear functions, the whole
 * network would be equivalent to a single layer, as the composition of linear
 * functions is still a linear function.
 */
namespace activation_functions {

/**
 * Abstract base class for activation functions.
 */
class ActivationFunction {
public:
    /**
     * Evaluate the activation function at x.
     * @param x The input to the activation function.
     * @return The output of the activation function.
     */
    virtual double operator()(double x) const = 0;
    /**
     * Evaluate the derivative of the activation function at x.
     * @param x The input to the activation function.
     * @return The derivative of the activation function.
     */
    virtual double derivative(double x) const = 0;
};

class Identity : public ActivationFunction {
public:
    double operator()(double x) const override;
    double derivative(double x) const override;
};

/**
 * The sigmoid activation function.
 */
class Sigmoid : public ActivationFunction {
public:
    double operator()(double x) const override;
    double derivative(double x) const override;
};

/**
 * The hyperbolic tangent activation function.
 */
class Tanh : public ActivationFunction {
public:
    double operator()(double x) const override;
    double derivative(double x) const override;
};

}  // namespace activation_functions