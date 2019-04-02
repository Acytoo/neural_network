#ifndef CORE_SIMPLE_NET_H_
#define CORE_SIMPLE_NET_H_

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
using std::vector;
using cv::Mat // using mat for fast matrix operations


namespace simple_net {
  // c++11
  enum class ActivationFunction {
    ReLu,
    Sigmoid,
    Tanh
    };


  class Net {
  private:
    vector<Mat> layers_; // layers, input layer: layer[0]; output layer: layer[n-1];
    vector<int> bias_;   // bias, b, int
    vector<Mat> weights_; // weights, Mat
    vector<int> layer_neuron_numbers_; // number of neurons of each layers
    ActivationFunction activation_function_ = ActivationFunction::ReLu; // default activation function: ReLu
    float loss_;
    Mat target_;
    float learning_rate_;

    int InitWeights();
    int InitBias();
  public:
    Net(vector<int> layer_neuron_numbers);
    int Train();
    int Predict();
    int Forward();
    int Backward();


  };
}

#endif
