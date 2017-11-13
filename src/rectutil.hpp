#ifndef RECTUTIL_H
#define RECTUTIL_H

#include <vector>
#include <opencv2/opencv.hpp>

namespace ph {
  namespace rectu {
    std::vector<cv::Rect> copy(const std::vector<cv::Rect>& rects);
    void shiftRects(const std::vector<cv::Rect>& rects, std::vector<cv::Rect>& out, int sx, int sy);
    void createRectsFromKeypoints(const std::vector<std::vector<cv::KeyPoint>>& categorizedPoints, std::vector<cv::Rect>& out, bool check = true);
    bool isValidRect(const cv::Rect& r);
    bool intersect(const cv::Rect& r1, const cv::Rect& r2, cv::Rect& connected, int marginThreshold = 0);
    bool inBox(const cv::Point2i& pt, const std::vector<cv::Rect>& rects, int margin = 0);
    void mergeRects(const std::vector<cv::Rect>& inputRects, std::vector<cv::Rect>& out, int threshold = 0);
    void mergeRectsIfSameCenter(const std::vector<cv::Rect>& inputRects, const std::vector<cv::Point2i>& centers, std::vector<cv::Rect>& out, std::vector<cv::Point2i>& cOut, int threshold = 0);
    int volume(const cv::Rect& rect);
    void filterIntersections(const std::vector<cv::Rect> inRects1, const std::vector<cv::Rect> inRects2, const std::vector<cv::Point2i> inCenters, 
        std::vector<cv::Rect> outRects1, std::vector<cv::Rect> outRects2, std::vector<cv::Point2i> outCenters);
    bool allClose(const cv::Mat& img1, const cv::Rect& r1, const cv::Mat& img2, const cv::Rect& r2,
        cv::Mat& roi1, cv::Mat& roi2, cv::Point2i& sv, int dr = 2);
    bool allCloseWithShift(const cv::Mat& img1, const cv::Rect& r1, const cv::Mat& img2, const cv::Rect& r2, const cv::Point2i& sv);
    void nonzeroRects(const cv::Mat& input, int dx, int dy, std::vector<cv::Rect>& out);
    bool expand(const std::vector<cv::Rect>& rects, const int targetIndex, const int rows, const int cols, cv::Rect& out);
    void drawRects(cv::Mat& inOutImg, const std::vector<cv::Rect>& rects, const cv::Scalar& color = cv::Scalar(255, 0, 0), int w = 1);
  }
}

#endif
