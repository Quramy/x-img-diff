#include <opencv2/opencv.hpp>
#include <iostream>
#include "hunter.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[], char* envp[]) {
  if (argc != 3) {
    cerr << "usage: x-img-diff <actual_image_path> <expected_image_path>" << endl;
    return -1;
  }

  auto img1 = imread(argv[1], IMREAD_COLOR);
  auto img2 = imread(argv[2], IMREAD_COLOR);

  if (!img1.data) {
    cerr << "No actual image data" << endl;
    return -1;
  }

  if (!img2.data) {
    cerr << "No expected image data" << endl;
    return -1;
  }

  auto conf = ph::DiffdetectConfig();
  // cout << conf.debug << endl;
  ph::detectDiff(img1, img2, conf);
  return 0;
}
