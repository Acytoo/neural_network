#include <iostream>
#include <random>
#include <string>
// #include <algorithm>

#include <stdarg.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <simple_net.h>

using namespace std;
using namespace cv;

// for linux user only, data structor: <training/testing> / <label> / <id>.png
// data source: mnist, https://github.com/myleott/mnist_png
// input: image; target: label

// 1. get input image path and label
// 2. shuffle image path and label
// 3. read image file, get vector<Mat>
// 4. train

int ReadDataPath(vector<string>& input,
                 vector<string>& target,
                 string& folder_path) {

  if (folder_path[folder_path.size()-1] != '/')
    folder_path += "/";

  // get image label
  // cout << 1 << endl;
  struct stat s;
  const char* dir_name = folder_path.c_str();
  lstat(dir_name, &s);
  if(!S_ISDIR(s.st_mode))
    return -1; // not a folder
  struct dirent *ptr;
  DIR *dir;
  dir = opendir(dir_name);
  vector<string> sub_dirs;
  while((ptr=readdir(dir))!=NULL) {
    if(ptr->d_type == 4)
      sub_dirs.push_back(ptr->d_name);
  }

  // get image path
  // cout << 2 << endl;
  for (auto i=sub_dirs.begin(), stop=sub_dirs.end(); i!=stop; ++i) {
    struct dirent *ptr;
    DIR *dir;
    string prefix = folder_path + *i + "/";
    const char* sub_dir_name = (prefix).c_str();
    dir = opendir(sub_dir_name);
    while((ptr=readdir(dir))!=NULL) {
      if(ptr->d_type == 8) {
        input.push_back(prefix+ptr->d_name);
        target.push_back(*i);
        // cout << "type: " << *i << " name: " << prefix+ptr->d_name << endl;
        // Mat tmp = cv.imread()
        // read file, grayscale, /255.0, cvtTo CV_F32C1, flattern
      }
    }
  }
  return 0;
}

// shuffle two vectors, and these two vectors have same permutation
template <class T>
int ShuffleInputAndTarget(vector<T>& input, vector<T>& target) {
  if (input.size() != target.size()) {
    cout << "Error: Image and label not match" << endl;
    return -1;
  }
  auto g = std::default_random_engine(0);
  std::shuffle(input.begin(), input.end(), g);
  g = std::default_random_engine(0);
  std::shuffle(target.begin(), target.end(), g);
  return 0;
}


int GetTrainingMat(vector<string>& img_path,
                   vector<string>& label,
                   vector<Mat>& input,
                   vector<Mat>& target) {
  for (int i=0, stop=img_path.size(); i != stop; ++i) {
    // read image
    Mat tmp = imread(img_path[i], 0);
    tmp.convertTo(tmp, CV_32FC3, 1.0/255, 0); // convert to 0<tmp<1 float Mat
    // reshape to 784 rows Mat
    tmp = tmp.reshape(1, 784);// cn, rows; cn: number of new channel;
    input.push_back(tmp); // push back
    // store label
    int int_label = stoi(label[i]); // c++11
    Mat mat_label = Mat::zeros(10, 1, CV_32FC1);
    mat_label.at<float>(int_label, 0) = 1; // eg: label 2 -> [0;0;1;0;0;0;0;0;0;0]
    target.push_back(mat_label);
  }
  return 0;
}

int main() {
  vector<string> img_path;
  vector<string> label;
  img_path.reserve(60000); // mnist training set has 60,000 pics
  label.reserve(60000);
  string folder_path = "../data/mnist_png/training/";
  int err = ReadDataPath(img_path, label, folder_path);
  if (err)
    exit(err);
  err = ShuffleInputAndTarget(img_path, label);
  if (err)
    exit(err);
  vector<Mat> input, target;
  input.reserve(60000);
  target.reserve(60000);
  GetTrainingMat(img_path, label, input, target);

  // Now let's build THE net !
  cout<< input[0].size() << endl;
  vector<int> layer_neuron_numbers = {784, 512, 256, 10};
  double learning_rate = 0.4;
  double bias_num = 0.5;
  double weight_mean = 0.0;
  double weight_stddev = 0.1;
  simple_net::Net mnist_test(layer_neuron_numbers,
                             learning_rate,
                             bias_num,
                             weight_mean,
                             weight_stddev);

  int batch_size = 100, epochs = 1;
  cout << "start training" << endl;
  mnist_test.Train(input,
                   target,
                   batch_size,
                   epochs);


  return 0;
}
