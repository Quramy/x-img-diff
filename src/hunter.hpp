#ifndef __HUNTER_H__
#define __HUNTER_H__

#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace ph {

  class DiffdetectConfig {
    public:
      bool debug = true;
      DiffdetectConfig() {
        this->debug = true;
      }
  };

  struct DiffResult {
    vector<Rect> diffRects = vector<Rect>();
    vector<Rect> matchedRects = vector<Rect>();
    DiffResult() {
    }
  };

  void detectDiff(const Mat &img1, const Mat &img2, const DiffdetectConfig &config);
}


#endif
