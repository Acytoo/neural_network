/* Read data from mnist dataset (png format, from https://github.com/myleott/mnist_png)

***shuffle training data***

shuffle index for two vector(images and labels)

of use the same shuffle seed, to get same permutation

*/

#include <iostream>
#include <algorithm>
#include <random>

using namespace std;
int shuffle_test() {
  // c++11 and later
  vector<int>a = {1,2,3,4,5,6,7,8,9,0};
  vector<int>b = {1,2,3,4,5,6,7,8,9,0};
  auto rng = std::default_random_engine(0);
  std::shuffle(a.begin(), a.end(), rng);
  rng = std::default_random_engine(0);
  std::shuffle(b.begin(), b.end(), rng);

  for(int i=0, stop = a.size(); i != stop; ++i) {
    cout << a[i] << " " << b[i] << endl;

  }
  return 0;
}
