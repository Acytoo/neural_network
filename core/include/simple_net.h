#ifndef CORE_SIMPLE_NET_H_
#define CORE_SIMPLE_NET_H_

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
using std::vector;
using cv::Mat; // using mat for fast matrix operations
using cv::Scalar; // Mat + bias


namespace simple_net {
  // c++11
  enum class ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh
    };


  class Net {
  private:
    vector<Mat> layers_; // layers, input layer: layer[0]; output layer: layer[n-1];
    int layer_size_;
    vector<Mat> bias_;   // bias, b, Mat for Weight · inputs + bias
    vector<Mat> weights_; // weights, Mat
    vector<int> layer_neuron_numbers_; // number of neurons of each layers
    ActivationFunction activation_function_ = ActivationFunction::ReLU; // default activation function: ReLu
    double loss_;
    Mat target_;
    Mat output_error_;
    double learning_rate_;
    Mat delta_error_;

    double weight_mean_ = 0.0, weight_stddev_ = 0.1, bias_num_ = 0.05;

    int InitWeights(double mean, double stddev);
    int InitBias(const Scalar& bias);
    int Activate(const Mat& product, Mat& layer);
    int ReLU(const Mat& product, Mat& layer);   // ReLU activate function

    int CalculateLoss(); // Loss function
    int UpdateWeights();
    int CalculateDelta();

  public:
    Net(vector<int> layer_neuron_numbers,
        double learning_rate=0.3,
        double bias_num=0.05,
        double weight_mean=0.0,
        double weight_stddev=0.1);

    int Train();
    int Predict();
    int Forward();
    int Backward();
    // void set_learning_rate(double learning_rate) {learning_rate_ = learning_rate;}

  };
}

#endif
