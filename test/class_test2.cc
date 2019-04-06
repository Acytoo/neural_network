#include <iostream>

#include <simple_net.h>

using namespace std;

using namespace cv;


void InitMat(Mat& m,float* num) {
  for(int i=0;i<m.rows;i++)
    for(int j=0;j<m.cols;j++)
      m.at<float>(i,j)=*(num+i*m.rows+j);
}

int main() {
  vector<int> layer_neuron_numbers = {2,2, 3, 2};
  double learning_rate = 0.4;
  double bias_num = 0.3;
  double weight_mean = 0.0;
  double weight_stddev = 0.1;
  simple_net::Net test_net(layer_neuron_numbers,
                           learning_rate,
                           bias_num,
                           weight_mean,
                           weight_stddev);

  vector<Mat> input, target;
  int batch_size = 10, epochs = 5;

  Mat input_tmp(2, 1, CV_32FC1);
  input_tmp.at<float>(0,0) = 1;
  Mat target_tmp(2, 1, CV_32FC1);
  target_tmp.at<float>(0,0) = 1;

  srand(time(0));
  for (int i=0; i != 50; ++i) {
    int random_int = rand() % 100 + 1; // 1 - 100
    // cout << random_int << endl;
    input_tmp.at<float>(1,0) = (float)random_int;
    target_tmp.at<float>(1,0) = (float)random_int + rand() % 10;
    input.push_back(input_tmp);
    target.push_back(target_tmp);

  }

  test_net.Train(input,
                 target,
                 batch_size,
                 epochs);
  // float temp[] = {1,2,3,4,5,6,7,8,9};
  // Mat a(3, 3, CV_32FC1);

  // InitMat(a, temp);
  // a = a / 2;
  // cout << a << endl;
  return 0;
}
