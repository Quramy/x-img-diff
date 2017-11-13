#include <opencv2/opencv.hpp>
#include <iostream>
#include "hunter.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[], char* envp[]) {
  auto img1 = imread("../hunter/img/actual.png");
  auto img2 = imread("../hunter/img/expected.png");
  auto conf = ph::DiffdetectConfig();
  // cout << conf.debug << endl;
  ph::detectDiff(img1, img2, conf);
  return 0;
}
