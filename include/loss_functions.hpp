#pragma once

#include <cmath>
#include <vector>

namespace loss_functions {

/**
 * A loss function is a function that measures the difference between the
 * predicted and actual values.
 */
class LossFunction {
public:
    /**
     * Compute the loss between the predicted and actual values.
     * @param predicted The predicted values.
     * @param actual The actual values.
     * @return The loss between the predicted and actual values.
     */
    virtual double operator()(const std::vector<double>& predicted,
                              const std::vector<double>& actual) const = 0;

    /**
     * Compute the derivative of the loss with respect to the predicted values.
     * @param predicted The predicted values.
     * @param actual The actual values.
     * @return The derivative of the loss with respect to the predicted values.
     */
    virtual std::vector<double> derivative(
        const std::vector<double>& predicted,
        const std::vector<double>& actual) const = 0;
};

/**
 * The mean squared error loss function.
 */
class MeanSquaredError : public LossFunction {
public:
    double operator()(const std::vector<double>& predicted,
                      const std::vector<double>& actual) const override;
    std::vector<double> derivative(
        const std::vector<double>& predicted,
        const std::vector<double>& actual) const override;
};

}  // namespace loss_functions