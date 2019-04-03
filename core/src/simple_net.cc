#include "simple_net.h"

#include <omp.h>
#include <ctime>

using std::cout;
using std::endl;
using std::string;
using std::vector;
using cv::Mat;
using cv::Scalar;


namespace simple_net {
  Net::Net(vector<int> layer_neuron_numbers,
           double learning_rate,
           double bias_num,
           double weight_mean,
           double weight_stddev) {

    // init layers
    layer_neuron_numbers_ = layer_neuron_numbers;
    layer_size_ = layer_neuron_numbers_.size();
    layers_.resize(layer_size_);
    for (int i=0; i != layer_size_; ++i) {
      layers_[i].create(layer_neuron_numbers_[i], 1, CV_32FC1); // row, col, type
    }

    // init weights and bias
    weights_.resize(layer_size_ - 1);
    bias_.resize(layer_size_ - 1);
    for (int i=0, stop=layer_size_-1; i != stop; ++i) {
      weights_[i].create(layers_[i+1].rows, layers_[i].rows, CV_32FC1);
      bias_[i] = Mat::zeros(layers_[i+1].rows, 1, CV_32FC1);
    }

    int err = InitWeights(weight_mean, weight_stddev);
    if (err) {
      cout << "Error init weights"<< endl;
      exit(-1);
    }
    err = InitBias(Scalar(bias_num));
    if (err) {
      cout << "Error init bias" << endl;
      exit(-2);
    }
  }

  int Net::InitWeights(double mean,  double stddev) {
    // cv2.randn(dst, mean, stddev)
    // Gaussian distribution now
    for (int i=0, stop = layer_size_-1; i != stop; ++i) {
      randn(weights_[i], mean, stddev);
    }
    return 0;
  }

  int Net::InitBias(const Scalar& bias) {
    for (int i=0, stop=layer_size_-1; i != stop; ++i) {
      bias_[i] = bias;
    }
    return 0;
  }

  int Net::Train() {
    return 0;
  }

  int Net::Predict() {
    return 0;
  }

  int Net::Forward() {
    for (int i=0, stop = layer_size_-1; i != stop ; ++i) {
      Mat product = weights_[i] * layers_[i] + bias_[i]; // sequence: weight, layer
      int err = Activate(product, layers_[i+1]);
      if (err) {
        cout << "Error activate" << endl;
        exit(-3);
      }
    }
    return 0;
  }

  int Net::Backward() {
    CalculateLoss();
    UpdateWeights();
    return 0;
  }


  int Net::Activate(const Mat& product, Mat& layer) {
    switch(activation_function_) {
    case ActivationFunction::ReLU:
      return ReLU(product, layer);
      break;
    case ActivationFunction::Sigmoid:
      cout << "Sigmoid, not support yet" << endl;
      break;
    case ActivationFunction::Tanh:
      cout << "Tanh, not support yet" << endl;
      break;
    }
    return 0;
  }

  int Net::ReLU(const Mat& product, Mat& layer) {
    // layer: n x 1 mat
    time_t start_time = std::time(0);
    int stop_r = product.rows;
#pragma omp parallel for
    for (int row=0; row < stop_r; ++row) {
      // printf("OpenMP, thread index: %d\n", omp_get_thread_num());
      float tmp = product.at<float>(row,0);
      layer.at<float>(row,0) = tmp < 0 ? 0: tmp;
    }
    cout << "time cost: " << std::time(0) - start_time << endl;
    return 0;
  }


  // void cv::pow	(	InputArray 	src,
  //                   double 	power,
  //                   OutputArray 	dst
  //                   )
  int Net::CalculateLoss() {
    output_error_ = target_ - layers_[layer_size_-1];
    Mat error_tmp;
    cv::pow(output_error_, 2.0, error_tmp);
    Scalar err_sum_tmp = cv::sum(error_tmp);
    loss_ = err_sum_tmp[0] / layers_[lauer_size_-1].rows;
    return 0;
  }

  int Net::UpdateWeights() {
    CalaulateDelda();
    return 0;
  }

  int Net::CalculateDelta() {
    delta_error_.resize(layer_size_-1);

    return 0;
  }


}
