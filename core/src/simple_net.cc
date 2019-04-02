#include "simple_net.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;
using cv::Mat;


namespace simple_net {
  Net::Net(vector<int> layer_neuron_numbers) {
    layer_neuron_numbers_ = layer_neuron_numbers;
    int err = InitWeight();
    if (!err) {
      cout << "Error init weights"<< endl;
      exit(-1);
    }
    err = InitBias();
    if (!err) {
      cout << "Error init bias" << endl;
      exit(-2);
    }
  }

  int Net::InitWeights() {
    return 0;
  }

  int Net::InitBias() {
    return 0;
  }

  int Net::Train() {
    return 0;
  }

  int Net::Predict() {
    return 0;
  }

  int Net::Forward() {
    return 0;
  }

  int Net::Backward() {
    return 0;
  }

}
