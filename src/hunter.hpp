#ifndef __HUNTER_H__
#define __HUNTER_H__

#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace ph {

  struct DiffConfig {
    bool debug = false;
    int maxMatchingPoints = 400;
    int shiftDelta = 2;
    int connectionDistance = 60;
    int thresholdPixcelNorm = 10;
    int gridSize = 32;
    DiffConfig() {
    }
  };

  struct PixelMatchingResult {
    bool isMatched;
    Point2i translate;
    Rect center1;
    Rect bounding1;
    Rect center2;
    Rect bounding2;
    vector<Rect> diffMarkers1;
    vector<Rect> diffMarkers2;
    PixelMatchingResult() {
      this->isMatched = true;
    }
    PixelMatchingResult(const Rect& center1, const Rect& bounding1, const Rect& center2,  const Rect& bounding2) {
      this->isMatched = true;
      this->center1 = center1;
      this->bounding1 = bounding1;
      this->center2 = center2;
      this->bounding2 = bounding2;
    }
    PixelMatchingResult(
        const Rect& center1, const Rect& bounding1, const vector<Rect>& diffMarkers1,
        const Rect& center2, const Rect& bounding2, const vector<Rect>& diffMarkers2) {
      this->isMatched = false;
      this->center1 = center1;
      this->bounding1 = bounding1;
      this->center2 = center2;
      this->bounding2 = bounding2;
      this->diffMarkers1 = diffMarkers1;
      this->diffMarkers2 = diffMarkers2;
    }
  };

  struct DiffResult {
    vector<PixelMatchingResult> matches;
    vector<Rect> strayingRects1;
    vector<Rect> strayingRects2;
    DiffResult() { }
    DiffResult(const vector<PixelMatchingResult>& matches, const vector<Rect>& strayingRects1, const vector<Rect>& strayingRects2) {
      this->matches = matches;
      this->strayingRects1 = strayingRects1;
      this->strayingRects2 = strayingRects2;
    }
  };

  void detectDiff(const Mat &img1, const Mat &img2, DiffResult& out, const DiffConfig &config);
}

#endif
