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

    learning_rate_ = learning_rate;
    weight_mean_ = weight_mean;
    weight_stddev_ = weight_stddev;
    bias_num_ = bias_num;

    for (int i=0; i != layer_size_; ++i) {
      layers_[i].create(layer_neuron_numbers_[i], 1, CV_32FC1); // row, col, type
    }

    // init weights and bias
    weights_.resize(layer_size_ - 1);
    bias_.resize(layer_size_ - 1);
    for (int i=0, stop=layer_size_-1; i != stop; ++i) {
      weights_[i].create(layers_[i+1].rows, layers_[i].rows, CV_32FC1);
      randn(weights_[i], weight_mean_, weight_stddev_);
      bias_[i] = Mat::zeros(layers_[i+1].rows, 1, CV_32FC1);
      bias_[i] = Scalar(bias_num_);
    }

    // int err = InitWeights(weight_mean, weight_stddev);
    // if (err) {
    //   cout << "Error init weights"<< endl;
    //   exit(-1);
    // }
    // err = InitBias(Scalar(bias_num));
    // if (err) {
    //   cout << "Error init bias" << endl;
    //   exit(-2);
    // }

  }

  // Deprecated
  int Net::InitWeights(double mean,  double stddev) {
    // cv2.randn(dst, mean, stddev)
    // Gaussian distribution now
    for (int i=0, stop = layer_size_-1; i != stop; ++i) {
      randn(weights_[i], mean, stddev);
    }
    return 0;
  }

  // Deprecated
  int Net::InitBias(const Scalar& bias) {
    for (int i=0, stop=layer_size_-1; i != stop; ++i) {
      bias_[i] = bias;
    }
    return 0;
  }

  // online learning  minist test set(hello world in neural network)
  int Net::Train(const vector<Mat>& input, const vector<Mat>& target) {
    if (input[0].rows != layers_[0].rows) {
      cout << "Error input rows != layer[0] rows" << endl;
      return -1;
    }
    time_t start_time = std::time(0);
    // Mat sample;
    int train_times = 0;
    while (train_times <= 5) {
      for (int i=0, stop=input.size(); i != stop; ++i) {
        target_ = target[i];
        layers_[0] = input[i];
        Forward();
        Backward();
        ShowAccuAndLoss();
        ++train_times;
      }
      learning_rate_ *= 1.01;
    }
    cout << "train time cost: " << std::time(0) - start_time << endl;
    return 0;
  }


  // （1）iteration：表示1次迭代（也叫training step），每次迭代更新1次网络结构的参数；
  // （2）batch-size：1次迭代所使用的样本量；
  // （3）epoch：1个epoch表示过了1遍训练集中的所有样本。
  // The loss function computes the error for a single training example;
  // the cost function is the average of the loss funcitons of the entire training set.
  // ---Andrew Ng

  // training
  int Net::Train(const vector<Mat>& input, const vector<Mat>& target, int batch_size, int epochs) {

    time_t start_time = std::time(0);

    batch_output_error_ = Mat::zeros(target[0].size(), CV_32FC1); // might move to constructor
    int trainingset_size = input.size();

    if (input[0].rows != layers_[0].rows) {
      cout << "Error input rows != layer[0] rows" << endl;
      cout << input[0].size() << " !=  " << layers_[0].size() << endl;
      return -1;
    }
    if (target[0].rows != layers_[layer_size_-1].rows) {
      cout << "Error target rows != layer[n-1] rows" << endl;
      return -2;
    }
    if (trainingset_size != target.size()) {
      cout << "input.size != target.size" << endl;
      return -3;
    }
    batch_size_ = batch_size;
    epochs_ = epochs;

    // assume training dataset is larger than batch size
    int input_size = input.size();
    int total_forward_times = epochs * input_size;
    int forward_times = 0;
    int trainingset_index = 0;
    while (forward_times <= total_forward_times) {
      trainingset_index = forward_times % input_size;

      layers_[0] = input[trainingset_index];
      target_ = target[trainingset_index];
      Forward();
      AccumulateLoss();

      ++forward_times;
      if (forward_times % batch_size_ == 0) {
        // one batch
        BatchBackPropagation(); // Backward after a barch training
        batch_output_error_ = Mat::zeros(target[0].size(), CV_32FC1); //reset error
        ShowAccuAndLoss();
        learning_rate_ *= 1.01;
      }

    }

    cout << "train time cost: " << std::time(0) - start_time << endl;
    return 0;
  }



  int Net::Predict(Mat& input, Mat& res) {
    if (input.rows != layers_[0].rows) {
      cout << "Row of mat for prediction should equals to the row of layers[0]"
           << endl;
      return -1;
    }
    cout << "---------------------Predict" << endl;
    layers_[0] = input;
    Forward();
    res = layers_[layer_size_-1];
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

  int Net::Activate(const Mat& product, Mat& layer) {
    switch(activation_function_) {
    case ActivationFunction::ReLU:
      return ReLU(product, layer);
      break;
    case ActivationFunction::Sigmoid:
      cout << "Sigmoid, not supported yet" << endl;
      break;
    case ActivationFunction::Tanh:
      cout << "Tanh, not supported yet" << endl;
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
      float tmp = product.at<float>(row,0);
      layer.at<float>(row,0) = tmp < 0 ? 0: tmp;
    }
    // cout << "time cost: " << std::time(0) - start_time << endl;
    return 0;
  }

  int Net::Backward() {
    CalculateLoss();
    UpdateWeightsAndBias();
    return 0;
  }

  int Net::CalculateLoss() {
    output_error_ = target_ - layers_[layer_size_-1];
    Mat error_tmp;
    cv::pow(output_error_, 2.0, error_tmp);
    loss_ = cv::sum(error_tmp)[0] / 2; //layers_[layer_size_-1].rows;
    cout << "loss " << loss_ << endl;
    return 0;
  }

  // update weights and bias
  int Net::UpdateWeightsAndBias() {
    CalculateDelta();
    for (int i=0, stop=weights_.size(); i != stop; ++i) {
      weights_[i] += learning_rate_ * (delta_error_[i] * layers_[i].t());
      bias_[i] += learning_rate_ * delta_error_[i];
    }
    return 0;
  }

  // Calculate delta of each layer
  int Net::CalculateDelta() {
    delta_error_.resize(layer_size_-1);
    for (int i=layer_size_-2; i >= 0; --i) {
      delta_error_[i].create(layers_[i+1].size(), CV_32FC1); //layers_[i+1].type()
      Mat tmp = Mat::zeros(layers_[i+1].rows, 1, CV_32FC1); //(layers_[i+1]);
      Derivative(layers_[i+1], tmp);

      if (i == layer_size_-2) { // output layer delta error
        delta_error_[i] = tmp.mul(output_error_);
      }
      else {  // hidden layer delta error
        delta_error_[i] = tmp.mul(weights_[i+1].t() * delta_error_[i+1]);
      }
    }
    return 0;
  }

  int Net::Derivative(const Mat& layer, Mat& delta_tmp) {
    switch(activation_function_) {
    case ActivationFunction::ReLU:
      for (int row=0, stop = layer.rows; row != stop; ++row) {
        delta_tmp = 0 < layer.at<float>(row, 0) ? 1 : 0;
      }
      break;
    case ActivationFunction::Sigmoid:
      cout << "Sigmoid, not supported yet" << endl;
      break;
    case ActivationFunction::Tanh:
      cout << "Tanh, not supported yet" << endl;
      break;
    }
    return 0;
  }


  int Net::BatchBackPropagation() {
    output_error_ = batch_output_error_ / batch_size_; //might batch err x 2
    Mat error_tmp;
    cv::pow(output_error_, 2.0, error_tmp);
    loss_ = cv::sum(error_tmp)[0] / 2; // cv::sum() return a scalar
    UpdateWeightsAndBias();
    return 0;
  }

  int Net::ShowAccuAndLoss() {
    // Show current accuracy and loss
    cout << "loss from last batch: " << loss_ << endl;
    return 0;
  }

  int Net::AccumulateLoss() {
    batch_output_error_ += target_ - layers_[layer_size_-1];
    return 0;
  }

}
