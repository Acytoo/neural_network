#include <iostream>

#include <simple_net.h>

using namespace std;

using namespace cv;

int main() {
  vector<int> layer_neuron_numbers = {256,64,32,128};
  double learning_rate = 0.3;
  double bias_num = 0.5;
  double weight_mean = 0.0;
  double weight_stddev = 0.1;
  simple_net::Net test_net(layer_neuron_numbers,
                           learning_rate,
                           bias_num,
                           weight_mean,
                           weight_stddev);
  test_net.Forward();
  return 0;
}
