#ifndef __HUNTER_H__
#define __HUNTER_H__

#include <vector>
#include <opencv2/opencv.hpp>

namespace ph {

  struct DiffConfig {
    bool debug = false;
    int thresholdTransX = 400;
    int maxMatchingPoints = 800;
    bool useCanny = true;
    int shiftDelta = 2;
    int connectionDistance = 60;
    int thresholdPixelNorm = 10;
    int gridSize = 32;
    DiffConfig();
  };

  struct PixelMatchingResult {
    bool isMatched;
    cv::Point2i translate;
    cv::Rect center1;
    cv::Rect bounding1;
    cv::Rect center2;
    cv::Rect bounding2;
    std::vector<cv::Rect> diffMarkers1;
    std::vector<cv::Rect> diffMarkers2;
    PixelMatchingResult();
    PixelMatchingResult(const cv::Rect& center1, const cv::Rect& bounding1, const cv::Rect& center2,  const cv::Rect& bounding2);
    PixelMatchingResult(const cv::Rect& center1, const cv::Rect& bounding1, const std::vector<cv::Rect>& diffMarkers1,
        const cv::Rect& center2, const cv::Rect& bounding2, const std::vector<cv::Rect>& diffMarkers2);
  };

  struct DiffResult {
    std::vector<PixelMatchingResult> matches;
    std::vector<cv::Rect> strayingRects1;
    std::vector<cv::Rect> strayingRects2;
    DiffResult();
    DiffResult(const std::vector<PixelMatchingResult>& matches, const std::vector<cv::Rect>& strayingRects1, const std::vector<cv::Rect>& strayingRects2);
  };

  void detectDiff(const cv::Mat &img1, const cv::Mat &img2, DiffResult& out, const DiffConfig &config);
}

#endif
