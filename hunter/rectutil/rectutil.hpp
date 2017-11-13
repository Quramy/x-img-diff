#ifndef RECTUTIL_H
#define RECTUTIL_H

#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace ph {
  namespace rectu {
    vector<Rect> copy(const vector<Rect>& rects);
    void shiftRects(const vector<Rect>& rects, vector<Rect>& out, int sx, int sy);
    void createRectsFromKeypoints(const vector<vector<KeyPoint>>& categorizedPoints, vector<Rect>& out, bool check = true);
    bool isValidRect(const Rect& r);
    bool intersect(const Rect& r1, const Rect& r2, Rect& connected, int marginThreshold = 0);
    bool inBox(const Point2i& pt, const vector<Rect>& rects, int margin = 0);
    void mergeRects(const vector<Rect>& inputRects, vector<Rect>& out, int threshold = 0);
    void mergeRectsIfSameCenter(const vector<Rect>& inputRects, const vector<Point2i>& centers, vector<Rect>& out, vector<Point2i>& cOut, int threshold = 0);
    int volume(const Rect& rect);
    void filterIntersections(const vector<Rect> inRects1, const vector<Rect> inRects2, const vector<Point2i> inCenters, 
        vector<Rect> outRects1, vector<Rect> outRects2, vector<Point2i> outCenters);
    bool allClose(const Mat& img1, const Rect& r1, const Mat& img2, const Rect& r2,
        Mat& roi1, Mat& roi2, Point2i& sv, int dr = 2);
    bool allCloseWithShift(const Mat& img1, const Rect& r1, const Mat& img2, const Rect& r2, const Point2i& sv);
    void nonzeroRects(const Mat& input, int dx, int dy, vector<Rect>& out);
    bool expand(const vector<Rect>& rects, const int targetIndex, const int rows, const int cols, Rect& out);
    void drawRects(Mat& inOutImg, const vector<Rect>& rects, const Scalar& color = Scalar(255, 0, 0), int w = 1);
  }
}

#endif
