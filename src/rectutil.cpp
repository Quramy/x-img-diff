#include <vector>
#include <opencv2/opencv.hpp>
#include "rectutil.hpp"

using namespace std;
using namespace cv;

namespace ph {
  namespace rectu {
    vector<Rect> copy(const vector<Rect>& rects) {
      auto ret = vector<Rect>();
      for (auto rect: rects) {
        ret.push_back(Rect(rect.x, rect.y, rect.width, rect.height));
      }
      return ret;
    }

    Rect shift(const Rect& rect, int sx, int sy) {
      return Rect(rect.x + sx, rect.y + sy, rect.width, rect.height);
    }
    
    void shiftRects(const vector<Rect>& rects, vector<Rect>& out, int sx, int sy) {
      auto ret = vector<Rect>();
      for (auto& r: rects) {
        ret.push_back(shift(r, sx, sy));
      }
      out = ret;
    }

    void createRectsFromKeypoints(const vector<vector<KeyPoint>>& categorizedPoints, vector<Rect>& out, bool check) {
      out = vector<Rect>();
      for (auto& points: categorizedPoints) {
        if (points.size() == 0) break;
        int minX = (int)points.at(0).pt.x;
        int minY = (int)points.at(0).pt.y;
        int maxX = (int)points.at(0).pt.x;
        int maxY = (int)points.at(0).pt.y;
        for (auto& kp: points) {
          const auto& pt = kp.pt;
          if (minX > pt.x) minX = (int)pt.x;
          if (maxX < pt.x) maxX = (int)pt.x;
          if (minY > pt.y) minY = (int)pt.y;
          if (maxY < pt.y) maxY = (int)pt.y;
        }
        const Rect r(Point2i(minX, minY), Point2i(maxX, maxY));
        if (!check || isValidRect(r)) {
          out.push_back(Rect(Point2i(minX, minY), Point2i(maxX, maxY)));
        }
      }
    }

    bool isValidRect(const Rect& r) {
      return r.width >= 4 && r.height >= 4 && r.width * r.height >= 80;
    }

    bool intersect(const Rect& r1, const Rect& r2, Rect& connected, int marginThreshold) {
      auto mx1 = max(r1.tl().x, r2.tl().x);
      auto mx2 = min(r1.br().x, r2.br().x);
      auto my1 = max(r1.tl().y, r2.tl().y);
      auto my2 = min(r1.br().y, r2.br().y);
      auto ret = mx2 - mx1 + marginThreshold > 0 && my2 - my1 + marginThreshold > 0;
      connected = Rect(
          Point2i(min(r1.tl().x, r2.tl().x), min(r1.tl().y, r2.tl().y)),
          Point2i(max(r1.br().x, r2.br().x), max(r1.br().y, r2.br().y))
          );
      return ret;
    }

    bool inBox(const Point2i& pt, const vector<Rect>& rects, int margin) {
      for (auto& r: rects) {
        if ((pt.x > r.x - margin) && (pt.x < r.x + r.width + margin) && (pt.y > r.y - margin) && (pt.y < r.y + r.height + margin)) {
          return true;
        }
      }
      return false;
    }

    void mergeRects(const vector<Rect>& inputRects, vector<Rect>& out, int threshold) {
      auto l = inputRects.size();
      auto connectedPairs = vector<vector<int>>(l);
      auto rects = copy(inputRects);
      for (int i = 0; i < l; ++i) {
        auto& r1 = rects.at(i);
        for (int j = i + 1; j < l; ++j) {
          auto& r2 = rects.at(j);
          Rect connected;
          if (intersect(r1, r2, connected, threshold)) {
            connectedPairs.at(i).push_back(j);
            rects.at(i) = connected;
            r1 = connected;
            rects.at(j) = connected;
          }
        }
      }
      out = vector<Rect>();
      for (int i = 0; i < l; ++i) {
        if (connectedPairs.at(i).size() == 0) {
          out.push_back(rects.at(i));
        }
      }
    }

    void mergeRectsIfSameCenter(const vector<Rect>& inputRects, const vector<Point2i>& centers, vector<Rect>& out, vector<Point2i>& cOut, int threshold) {
      auto l = inputRects.size();
      auto connectedPairs = vector<vector<int>>(l);
      auto rects = copy(inputRects);
      for (int i = 0; i < l; ++i) {
        auto& r1 = rects.at(i);
        auto& c1 = centers.at(i);
        for (int j = i + 1; j < l; ++j) {
          auto& r2 = rects.at(j);
          auto& c2 = centers.at(j);
          if (c1.x != c2.x || c1.y != c2.y) continue;
          Rect connected;
          if (intersect(r1, r2, connected, threshold)) {
            connectedPairs.at(i).push_back(j);
            rects.at(i) = connected;
            r1 = connected;
            rects.at(j) = connected;
          }
        }
      }
      out = vector<Rect>();
      cOut = vector<Point2i>();
      for (int i = 0; i < l; ++i) {
        if (connectedPairs.at(i).size() == 0) {
          out.push_back(rects.at(i));
          cOut.push_back(centers.at(i));
        }
      }
    }

    void filterIntersections(const vector<Rect>& inRects1, const vector<Rect>& inRects2, const vector<Point2i>& inCenters, 
        vector<Rect>& outRects1, vector<Rect>& outRects2, vector<Point2i>& outCenters) {
      int size = inCenters.size();
      auto marked = vector<bool>(size, false);
      Rect c;
      for (int i = 0; i < size; ++i) {
        if (marked.at(i)) continue;
        auto& r11 = inRects1.at(i);
        auto& r21 = inRects2.at(i);
        for (int j = i + 1; j < size; ++j) {
          if (marked.at(j)) continue;
          auto& r12 = inRects1.at(j);
          auto& r22 = inRects2.at(j);
          if (intersect(r11, r12, c) || intersect(r21, r22, c)) {
            auto v1 = r11.area();
            auto v2 = r12.area();
            if (v1 > v2) {
              marked.at(j) = true;
            } else {
              marked.at(i) = true;
            }
          }
        }
      }
      auto outR1 = vector<Rect>();
      auto outR2 = vector<Rect>();
      auto outC = vector<Point2i>();
      for (int i = 0; i < size; ++i) {
        if (marked.at(i)) continue;
        outR1.push_back(inRects1.at(i));
        outR2.push_back(inRects2.at(i));
        outC.push_back(inCenters.at(i));
      }
      outRects1 = outR1;
      outRects2 = outR2;
      outCenters = outC;
    }

    bool allClose(const Mat& img1, const Rect& r1, const Mat& img2, const Rect& r2,
        Mat& roi1, Mat& roi2, Point2i& sv, int dr) {
      sv = Point2i(0, 0);
      if (r1.width != r2.width || r1.height != r2.height) {
        return false;
      }
      for (int sx = -dr; sx <= dr; ++sx) {
        for (int sy = -dr; sy <= dr; ++sy) {
          auto imgr1 = img1(r1);
          auto imgr2 = img2(shift(r2, sx, sy));
          Mat diff;
          bitwise_xor(imgr1, imgr2, diff);
          cvtColor(diff, diff, COLOR_RGB2GRAY);
          if (countNonZero(diff) == 0) {
            roi1 = imgr1;
            roi2 = imgr2;
            sv = Point2i(sx, sy);
            return true;
          }
        }
      }
      return false;
    }

    bool allCloseWithShift(const Mat& img1, const Rect& r1, const Mat& img2, const Rect& r2, const Point2i& sv) {
      if (r1.width != r2.width || r1.height != r2.height) {
        return false;
      }
      auto imgr1 = img1(r1);
      auto imgr2 = img2(shift(r2, sv.x, sv.y));
      auto diff = imgr1 != imgr2;
      return countNonZero(diff) == 0;
    }

    void nonzeroRects(const Mat& input, int dx, int dy, vector<Rect>& out) {
      int h = input.rows, w = input.cols;
      out = vector<Rect>();
      int nx = w / dx + (w % dx == 0 ? 0 : 1);
      int ny = h / dy + (h % dy == 0 ? 0 : 1);
      for (int i = 0; i < ny; ++i) {
        int y1 = i * dy, y2 = (i + 1) * dy;
        if (y2 >= h) y2 = h - 1;
        for (int j = 0; j < nx; ++j) {
          int x1 = j * dx, x2 = (j + 1) * dx;
          if (x2 >= w) x2 = w - 1;
          // cout << x1 << ", " << y1 << ", " << x2 << ", " << y2 << endl;
          auto roi = input(Rect(Point2i(x1, y1), Point2i(x2, y2)));
          auto boundingR = boundingRect(roi);
          if (boundingR.width > 0 && boundingR.height >0) {
            out.push_back(shift(boundingR, x1, y1));
          }
        }
      }
    }

    bool expand(const vector<Rect>& rects, const int targetIndex, const int rows, const int cols, Rect& out) {
      const auto& target = rects.at(targetIndex);
      const int x1 = target.tl().x, x2 = target.br().x, y1 = target.tl().y, y2 = target.br().y;
      int ex1 = -1, ex2 = cols, ey1 = -1, ey2 = rows;
      const auto x1Bounds = Rect(Point2i(0, y1), Point2i(x2, y2));
      const auto x2Bounds = Rect(Point2i(x1, y1), Point2i(cols - 1, y2));
      const auto y1Bounds = Rect(Point2i(x1, 0), Point2i(x2, y2));
      const auto y2Bounds = Rect(Point2i(x1, y1), Point2i(x2, rows - 1));
      Rect connected;
      for (auto& r: rects) {
        if (&r == &target) continue;
        if (intersect(r, x1Bounds, connected, 0)) ex1 = max(ex1, r.br().x);
        if (intersect(r, x2Bounds, connected, 0)) ex2 = min(ex2, r.tl().x);
        if (intersect(r, y1Bounds, connected, 0)) ey1 = max(ey1, r.br().y);
        if (intersect(r, y2Bounds, connected, 0)) ey2 = min(ey2, r.tl().y);
      }
      auto outCandidate = Rect(Point2i(ex1 + 1, ey1 + 1), Point2i(ex2 -1, ey2 - 1));
      if (ex1 < 0 || ey1 < 0 || ex2 >= cols || ey2 >= rows) {
        out = outCandidate;
        return false;
      }
      const auto upperBound = Rect(Point2i(ex1 + 1, ey1 + 1), Point2i(ex2 - 1, y2));
      const auto lowerBound = Rect(Point2i(ex1 + 1, y1), Point2i(ex2 - 1, ey2 - 1));
      for (auto& r: rects) {
        if (&r == &target) continue;
        if (intersect(r, upperBound, connected, 0)) ey1 = max(ey1, r.br().y);
        if (intersect(r, lowerBound, connected, 0)) ey2 = min(ey2, r.tl().y);
      }
      outCandidate = Rect(Point2i(ex1 + 1, ey1 + 1), Point2i(ex2 -1, ey2 - 1));
      out = outCandidate;
      return true;
    }

    void drawRects(Mat& inOutImg, const vector<Rect>& rects, const Scalar& color, int w) {
      for (auto& r: rects) {
        rectangle(inOutImg, r, color, w);
      }
    }
  }
}
