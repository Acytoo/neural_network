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
    vector<Mat> layers_;//layers, input layer: layer[0]; output layer: layer[n-1];
    int layer_size_;
    vector<Mat> bias_;   // bias, b, Mat for Weight · inputs + bias
    vector<Mat> weights_; // weights, Mat
    vector<int> layer_neuron_numbers_; // number of neurons of each layers
    ActivationFunction activation_function_ = ActivationFunction::ReLU; // default activation function: ReLu
    float loss_;

    Mat target_;
    Mat output_error_;  // no
    Mat batch_output_error_; // accumulated error in one batch
    double learning_rate_;
    vector<Mat> delta_error_;

    int batch_size_; // batch size
    int epochs_;

    double weight_mean_ = 0.0, weight_stddev_ = 0.1, bias_num_ = 0.05;

    int InitWeights(double mean, double stddev);
    int InitBias(const Scalar& bias);
    int Activate(const Mat& product, Mat& layer);
    int ReLU(const Mat& product, Mat& layer);   // ReLU activate function

    int CalculateLoss(); // Loss function
    int UpdateWeightsAndBias();
    int CalculateDelta();
    int Derivative(const Mat& layer, Mat& delta_err);

  public:
    Net(vector<int> layer_neuron_numbers,
        double learning_rate=0.3,
        double bias_num=0.05,
        double weight_mean=0.0,
        double weight_stddev=0.1);

    int Train(const vector<Mat>& input, const vector<Mat>& target);
    int Train(const vector<Mat>& input, const vector<Mat>& target, int batch_size, int epochs);
    int Predict(Mat&, Mat&);
    int Forward();
    int Backward();
    // void set_learning_rate(double learning_rate) {learning_rate_ = learning_rate;}
    int AccumulateLoss();  // accumulate loss for a batch
    int BatchBackPropagation(); // Backward after a barch training
    int ShowAccuAndLoss();  // Show current accuracy and loss
    int set_target(Mat target) {target_ = target; return 0;} // deprecated


  };
}

#endif
