#include <iostream>

#include <simple_net.h>

using namespace std;

using namespace cv;

int main() {
  vector<int> layer_neuron_numbers = {2,3,3,2,1};
  double learning_rate = 0.05;
  double bias_num = 0.3;
  double weight_mean = 0.0;
  double weight_stddev = 0.1;
  simple_net::Net test_net(layer_neuron_numbers,
                           learning_rate,
                           bias_num,
                           weight_mean,
                           weight_stddev);

  // Mat target(128, 1, CV_32FC1, Scalar(0)) ;

  // test_net.set_target(target);
  // test_net.Forward();
  // test_net.Backward();
  // vector<Mat> input = {Mat::ones(256, 1, CV_32FC1)};

  Mat input_0(2, 1, CV_32FC1);
  input_0.at<float>(0, 0) = 1;
  input_0.at<float>(1, 0) = 0;

  Mat target_0(1, 1, CV_32FC1);
  target_0.at<float>(0, 0) = 1;

  Mat input_1(2, 1, CV_32FC1);
  input_1.at<float>(0, 0) = 0;
  input_1.at<float>(1, 0) = 1;

  Mat target_1(1, 1, CV_32FC1);
  target_1.at<float>(0, 0) = 0;

  Mat input_2(2, 1, CV_32FC1);
  input_2.at<float>(0, 0) = 1;
  input_2.at<float>(1, 0) = 1;

  Mat target_2(1, 1, CV_32FC1);
  target_2.at<float>(0, 0) = 1;

  Mat input_3(2, 1, CV_32FC1);
  input_3.at<float>(0, 0) = 0;
  input_3.at<float>(1, 0) = 0;

  Mat target_3(1, 1, CV_32FC1);
  target_3.at<float>(0, 0) = 0;

  // test_net.Train({input_0, input_1}, {target_0, target_1});
  test_net.Train({input_0, input_1, input_2, input_3},
                 {target_0, target_1, target_2, target_3});
  return 0;
}
