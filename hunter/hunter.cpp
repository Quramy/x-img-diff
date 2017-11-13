#include <cmath>
#include <iostream>
#include <vector>
#include <set>
#include <opencv2/opencv.hpp>
#include "rectutil/rectutil.hpp"
#include "hunter.hpp"

using namespace std;
using namespace cv;

namespace ph {

  void debugRect(const Rect& r) {
    cout << r.x << ", " << r.y << ", " << r.x + r.width << ", " << r.y + r.height << endl;
  }

  struct PixelMatchingResult {
    bool isMatched;
    Point2i translate;
    vector<Rect> boundingMarkers1;
    vector<Rect> boundingMarkers2;
    PixelMatchingResult() {
      this->isMatched = true;
    }
    PixelMatchingResult(const vector<Rect>& boundingMarkers1, const vector<Rect>& boundingMarkers2) {
      this->isMatched = false;
      this->boundingMarkers1 = boundingMarkers1;
      this->boundingMarkers2 = boundingMarkers2;
    }
  };

  struct Tension {
    int x1;
    int y1;
    int x2;
    int y2;
    Tension() { }
    Tension(int x1, int y1, int x2, int y2) {
      this->x1 = x1;
      this->y1 = y1;
      this->x2 = x2;
      this->y2 = y2;
    }
  };

  void debugTension(const Tension& t) {
    cout << t.x1 << ", " << t.y1 << ", " << t.x2 << "," << t.y2 << endl;
  }

  vector<vector<DMatch>> matchKps(const Mat& des1, const Mat& des2) {
    auto bf = BFMatcher(2);
    auto ret = vector<vector<DMatch>>();
    auto mm = vector<vector<DMatch>>();
    bf.knnMatch(des1, des2, mm, 2);
    for (auto& matches: mm) {
      if (matches.at(0).distance < 0.3 * matches.at(1).distance) {
        ret.push_back(vector<DMatch>(1, matches.at(0)));
      }
    }
    return ret;
  }

  void clusterMatches(const vector<KeyPoint>& kp1, const Mat& des1, const vector<KeyPoint> kp2, const Mat& des2, vector<vector<DMatch>>& matches, int n,
      vector<vector<DMatch>>& cgg, vector<Point2i>& qdc
      ) {
    int k = min((int)(matches.size() / 3), n);
    int dxw = 100, dyw = 1000, wy = 100;
    Mat v(matches.size(), 6, CV_32FC1);
    for (int i = 0; i < matches.size(); ++i) {
      auto& p1 = kp1[matches.at(i).at(0).queryIdx].pt;
      auto& p2 = kp2[matches.at(i).at(0).trainIdx].pt;
      v.at<float>(i, 0) = (float) p1.x;
      v.at<float>(i, 1) = (float) p2.x;
      v.at<float>(i, 2) = (float) p1.y * wy;
      v.at<float>(i, 3) = (float) p2.y * wy;
      v.at<float>(i, 4) = (float) (round(p2.x - p1.x) * dxw);
      v.at<float>(i, 5) = (float) (round(p2.y - p1.y) * dyw);
    }
    auto criteria = cvTermCriteria(CV_TERMCRIT_EPS|CV_TERMCRIT_ITER, 10, 1.0);
    Mat centers;
    Mat clusters = Mat::zeros(v.rows, 1, CV_32SC1);
    kmeans(v, k, clusters, criteria, 10, KMEANS_PP_CENTERS, centers);
    cgg = vector<vector<DMatch>>(k);
    for (int i = 0; i < clusters.rows; ++i) {
      int l = clusters.at<int>(i);
      cgg.at(l).push_back(matches.at(i).at(0));
    }
    qdc = vector<Point2i>();
    for (int i = 0; i < centers.rows; ++i) {
      Point2i p((int)(round(centers.at<float>(i, 4) / dxw)), (int)round((centers.at<float>(i, 5) / dyw)));
      qdc.push_back(p);
    }
  }

  void binaryDiff(const Mat& roi1, const Mat& roi2, Mat& out, int tpn) {
    Mat ad, adx;
    absdiff(roi1, roi2, ad);
    cvtColor(ad, adx, COLOR_RGB2GRAY);
    threshold(adx, out, tpn, 255, THRESH_BINARY); 
  }

  void arroundDiffMatch(const Mat& img1, const Rect& r1, const Mat& img2, const Rect& r2, const Tension& db, 
      const Point2i& sv, vector<Rect>& outRects, Tension& updatedDb) {
    int grid = 32; // TODO
    int norm = 10; // TODO 
    auto arroundDiffRects = vector<Rect>();
    int gix1 = 0, gix2 = 0, giy1 = 0, giy2 = 0;
    int gox1 = 0, gox2 = 0, goy1 = 0, goy2 = 0;
    const int r1x1 = r1.x, r1x2 = r1.x + r1.width, r1y1 = r1.y, r1y2 = r1.y + r1.height;
    const int r2x1 = r2.x, r2x2 = r2.x + r2.width, r2y1 = r2.y, r2y2 = r2.y + r2.height;
    int dbx1 = db.x1, dby1 = db.y1, dbx2 = db.x2, dby2 = db.y2;
    bool adx1 = true, adx2 = true, ady1 = true, ady2 = true;
    while (adx1 || adx2 || ady1 || ady2) {
      // left
      if (adx1) {
        gox1 = min(gix1 + grid, dbx1);
        auto t1 = Rect(Point2i(r1x1 - gox1, r1y1 - giy1), Point2i(r1x1 - gix1, r1y2 + giy2));
        auto t2 = Rect(Point2i(r2x1 - gox1, r2y1 - giy1), Point2i(r2x1 - gix1, r2y2 + giy2));
        // cout << "left" << endl;
        // debugRect(t1);
        // debugRect(t2);
        if (!rectu::allCloseWithShift(img1, t1, img2, t2, sv)) {
          Mat diff;
          vector<Rect> rects;
          binaryDiff(img1(t1), img2(t2), diff, norm);
          rectu::nonzeroRects(diff, grid, grid, rects);
          rectu::mergeRects(rects, rects, grid);
          int maxr = 0, minr = t1.height;
          for (auto& rr: rects) {
            minr = min(minr, rr.tl().y);
            maxr = max(maxr, rr.br().y);
          }
          float wd = (float)(maxr - minr), wr = (float)(r1.height + giy1 + giy2);
          if (wd / wr >= 0.6) {
            adx1 = false;
          } else {
            rectu::shiftRects(rects, rects, -gox1, -giy1);
            arroundDiffRects.insert(arroundDiffRects.end(), rects.begin(), rects.end());
          }
        }
      }
      if (adx1) {
        gix1 = gox1;
        adx1 = gix1 < dbx1;
      }

      // top
      if (ady1) {
        goy1 = min(giy1 + grid, dby1);
        auto t1 = Rect(Point2i(r1x1 - gix1, r1y1 - goy1), Point2i(r1x2 + gix2, r1y1 - giy1));
        auto t2 = Rect(Point2i(r2x1 - gix1, r2y1 - goy1), Point2i(r2x2 + gix2, r2y1 - giy1));
        // cout << "top" << endl;
        // debugRect(t1);
        // debugRect(t2);
        if (!rectu::allCloseWithShift(img1, t1, img2, t2, sv)) {
          Mat diff;
          vector<Rect> rects;
          binaryDiff(img1(t1), img2(t2), diff, norm);
          rectu::nonzeroRects(diff, grid, grid, rects);
          rectu::mergeRects(rects, rects, grid);
          int maxr = 0, minr = t1.width;
          for (auto& rr: rects) {
            minr = min(minr, rr.tl().x);
            maxr = max(maxr, rr.br().x);
          }
          float wd = (float)(maxr - minr), wr = (float)(r1.width + gox1 + gox2);
          if (wd / wr >= 0.6) {
            ady1 = false;
          } else {
            rectu::shiftRects(rects, rects, -gix1, -goy1);
            arroundDiffRects.insert(arroundDiffRects.end(), rects.begin(), rects.end());
          }
        }
      }
      if (ady1) {
        giy1 = goy1;
        ady1 = giy1 < dby1;
      }

      // right
      if (adx2) {
        gox2 = min(gix2 + grid, dbx2);
        auto t1 = Rect(Point2i(r1x2 + gix2, r1y1 - giy1), Point2i(r1x2 + gox2, r1y2 + giy2));
        auto t2 = Rect(Point2i(r2x2 + gix2, r2y1 - giy1), Point2i(r2x2 + gox2, r2y2 + giy2));
        // cout << "right" << endl;
        // debugRect(t1);
        // debugRect(t2);
        if (!rectu::allCloseWithShift(img1, t1, img2, t2, sv)) {
          Mat diff;
          vector<Rect> rects;
          binaryDiff(img1(t1), img2(t2), diff, norm);
          rectu::nonzeroRects(diff, grid, grid, rects);
          rectu::mergeRects(rects, rects, grid);
          int maxr = 0, minr = t1.height;
          for (auto& rr: rects) {
            minr = min(minr, rr.tl().y);
            maxr = max(maxr, rr.br().y);
          }
          float wd = (float)(maxr - minr), wr = (float)(r1.height + giy1 + giy2);
          if (wd / wr >= 0.6) {
            adx2 = false;
          } else {
            rectu::shiftRects(rects, rects, r1.width + gix2, -giy1);
            arroundDiffRects.insert(arroundDiffRects.end(), rects.begin(), rects.end());
          }
        }
      }
      if (adx2) {
        gix2 = gox2;
        adx2 = gix2 < dbx2;
      }

      // bottom
      if (ady2) {
        goy2 = min(giy2 + grid, dby2);
        auto t1 = Rect(Point2i(r1x1 - gix1, r1y2 + giy2), Point2i(r1x2 + gix2, r1y2 + goy2));
        auto t2 = Rect(Point2i(r2x1 - gix1, r2y2 + giy2), Point2i(r2x2 + gix2, r2y2 + goy2));
        // cout << "bottom" << endl;
        // debugRect(t1);
        // debugRect(t2);
        if (!rectu::allCloseWithShift(img1, t1, img2, t2, sv)) {
          Mat diff;
          vector<Rect> rects;
          binaryDiff(img1(t1), img2(t2), diff, norm);
          rectu::nonzeroRects(diff, grid, grid, rects);
          rectu::mergeRects(rects, rects, grid);
          int maxr = 0, minr = t1.width;
          for (auto& rr: rects) {
            minr = min(minr, rr.tl().x);
            maxr = max(maxr, rr.br().x);
          }
          float wd = (float)(maxr - minr), wr = (float)(r1.width + gox1 + gox2);
          if (wd / wr >= 0.6) {
            ady2 = false;
          } else {
            rectu::shiftRects(rects, rects, -gix1, r1.height + giy2);
            copy(rects.begin(), rects.end(), back_inserter(arroundDiffRects));
          }
        }
      }
      if (ady2) {
        giy2 = goy2;
        ady2 = giy2 < dby2;
      }
    }
    updatedDb = Tension(gix1, giy1, gix2, giy2);
    rectu::mergeRects(arroundDiffRects, arroundDiffRects, grid);
    outRects = arroundDiffRects;
  }

  void pixelMatch(const Mat& img1, const vector<Rect>& matchedRects1, const Mat& img2, const vector<Rect>& matchedRects2, const vector<Point2i>& cv,
      vector<PixelMatchingResult>& out, vector<Rect>& updatedRects1, vector<Rect>& updatedRects2) {
    auto rects1 = rectu::copy(matchedRects1);
    auto rects2 = rectu::copy(matchedRects2);
    auto result = vector<PixelMatchingResult>();
    for (int i = 0; i < cv.size(); ++i) {
      // cout << "pixelMatch iterate: " << i << endl;
      auto& r1 = rects1.at(i);
      auto& r2 = rects2.at(i);
      auto& center = cv.at(i);
      vector<Rect> innerRects;
      Mat imgr1, imgr2;
      Point2i sv;
      int shiftDelta = 2; //TODO
      if (!rectu::allClose(img1, r1, img2, r2, imgr1, imgr2, sv, shiftDelta)) {
        Mat diffImg;
        int thresholdPixelNorm = 10;
        // cout << "r1: " << r1.width << "x" << r1.height << endl;
        // cout << "r2: " << r2.width << "x" << r2.height << endl;
        binaryDiff(img1(r1), img2(r2), diffImg, thresholdPixelNorm);
        vector<Rect> x;
        rectu::nonzeroRects(diffImg, 32, 32, x);
        rectu::mergeRects(x, innerRects, 32);
      } else {
        innerRects = vector<Rect>(0);
      }

      Rect eb1, eb2;
      rectu::expand(rects1, i, img1.rows, img1.cols, eb1);
      rectu::expand(rects2, i, img2.rows, img2.cols, eb2);
      auto db = Tension(
          min(r1.tl().x - eb1.tl().x, r2.tl().x - eb2.tl().x),
          min(r1.tl().y - eb1.tl().y, r2.tl().y - eb2.tl().y),
          min(eb1.br().x - r1.br().x, eb2.br().x - r2.br().x),
          min(eb1.br().y - r1.br().y, eb2.br().y - r2.br().y)
          );
      Tension dbx;
      vector<Rect> outerRects;
      cout << "r1: ";
      debugRect(r1);
      cout << "eb1: ";
      debugRect(eb1);
      cout << "r2: ";
      debugRect(r2);
      cout << "eb2: ";
      debugRect(eb2);
      cout << "db: ";
      debugTension(db);
      arroundDiffMatch(img1, r1, img2, r2, db, sv, outerRects, dbx);
      // cout << "dbx: ";
      // debugTension(dbx);

      if (innerRects.size() > 0 && outerRects.size() == 0) {
        vector<Rect> s1, s2;
        cout << "inner diff" << endl;
        rectu::shiftRects(innerRects, s1, r1.x, r1.y);
        rectu::shiftRects(innerRects, s2, r2.x, r2.y);
        result.push_back(PixelMatchingResult(s1, s2));
      } else if (innerRects.size() == 0 && outerRects.size() > 0) {
        vector<Rect> s1, s2;
        cout << "outer diff" << endl;
        rectu::shiftRects(outerRects, s1, r1.x, r1.y);
        rectu::shiftRects(outerRects, s2, r2.x, r2.y);
        result.push_back(PixelMatchingResult(s1, s2));
      } else if (innerRects.size() > 0 && outerRects.size() > 0) {
        vector<Rect> s1, s2;
        cout << "inner and outer diff" << endl;
        copy(outerRects.begin(), outerRects.end(), back_inserter(innerRects));
        rectu::shiftRects(innerRects, s1, r1.x, r1.y);
        rectu::shiftRects(innerRects, s2, r2.x, r2.y);
        result.push_back(PixelMatchingResult(s1, s2));
      } else {
        cout << "not diff" << endl;
        result.push_back(PixelMatchingResult());
      }
      rects1.at(i) = Rect(Point2i(r1.tl().x - dbx.x1, r1.tl().y - dbx.y1), Point2i(r1.br().x + dbx.x2, r1.br().y + dbx.y2));
      rects2.at(i) = Rect(Point2i(r2.tl().x - dbx.x1, r2.tl().y - dbx.y1), Point2i(r2.br().x + dbx.x2, r2.br().y + dbx.y2));
      // cout << "updated r1: ";
      // debugRect(rects1.at(i));
      // cout << "updated r2: ";
      // debugRect(rects2.at(i));
    }
    out = result;
    updatedRects1 = rects1;
    updatedRects2 = rects2;
  }

  void detectDiff(const Mat &img1, const Mat &img2, const DiffdetectConfig &config) {
    auto imgIn1 = Mat(), imgIn2 = Mat();
    Canny(img1, imgIn1, 10, 40);
    Canny(img2, imgIn2, 10, 40);

    auto akaze = AKAZE::create();
    vector<KeyPoint> kp1, kp2;
    auto mask = Mat();
    Mat des1, des2;
    akaze->detectAndCompute(imgIn1, mask, kp1, des1);
    akaze->detectAndCompute(imgIn2, mask, kp2, des2);

    if (config.debug) {
      cout << "Input image1 size: " << img1.cols << "x" << img1.rows << endl;
      cout << "Input image2 size: " << img2.cols << "x" << img2.rows << endl;
      cout << "The num of keypoints(img1): " << kp1.size() << endl;
      cout << "The num of keypoints(img2): " << kp2.size() << endl;
    }

    auto matches = matchKps(des1, des2);
    const int MAX_X_DIST = 400; // TODO
    const int fmax = 400; // TODO
    auto filteredMatches = vector<vector<DMatch>>();
    for (auto& mv: matches) {
      if (abs(kp1.at(mv.at(0).queryIdx).pt.x - kp2.at(mv.at(0).trainIdx).pt.x) < MAX_X_DIST) {
        filteredMatches.push_back(mv);
      }
    }
    if (config.debug) {
      cout << "The num of matched: " << matches.size() << ", " << filteredMatches.size() << endl;
    }
    if (filteredMatches.size() > fmax) {
      if (config.debug) {
        cout << "Sampling to:" << fmax << endl;
      }
      auto xxx = vector<vector<DMatch>>();
      auto l = filteredMatches.size();
      for (int i = 0; i < fmax; ++i) {
        int ni = i * l / fmax;
        xxx.push_back(filteredMatches[ni]);
      }
      filteredMatches = xxx;
    }

    auto qdVec = vector<pair<int, int>>();
    set<int> s{};
    for (auto& m: filteredMatches) {
      auto& kpt1 = kp1.at(m.at(0).queryIdx);
      auto& kpt2 = kp2.at(m.at(0).trainIdx);
      auto dx = (int)(kpt2.pt.x - kpt1.pt.x);
      auto dy = (int)(kpt2.pt.y - kpt1.pt.y);
      s.insert(dx);
      s.insert(dy);
      auto p = make_pair(dx, dy);
      qdVec.push_back(p);
    }
    int n = s.size() / 2 + 1;
    if (config.debug) {
      cout << "Initial cluster size: " << n << endl;
    }
    vector<vector<DMatch>> categorizedMatches;
    vector<Point2i> qdc;
    clusterMatches(kp1, des1, kp2, des2, filteredMatches, n, categorizedMatches, qdc);

    auto categorizedKp1 = vector<vector<KeyPoint>>(categorizedMatches.size());
    auto categorizedKp2 = vector<vector<KeyPoint>>(categorizedMatches.size());
    auto notCategorizedKp1 = vector<KeyPoint>();
    auto notCategorizedKp2 = vector<KeyPoint>();
    int i = 0;
    for (auto& mv: categorizedMatches) {
      for (auto& m: mv) {
          categorizedKp1.at(i).push_back(kp1.at(m.queryIdx));
          categorizedKp2.at(i).push_back(kp2.at(m.trainIdx));
      }
      ++i;
    }
    for (auto& kp: kp1) {
      for (auto& m: filteredMatches) {
        if (&kp1.at(m.at(0).queryIdx) == &kp) {
          break;
        }
      }
      notCategorizedKp1.push_back(kp);
    }
    for (auto& kp: kp2) {
      for (auto& m: filteredMatches) {
        if (&kp2.at(m.at(0).trainIdx) == &kp) {
          break;
        }
      }
      notCategorizedKp2.push_back(kp);
    }

    int distance = 60; // TODO
    auto matchedRects1 = vector<Rect>(), matchedRects2 = vector<Rect>();
    vector<Point2i> cv;
    rectu::createRectsFromKeypoints(categorizedKp1, matchedRects1);
    rectu::mergeRectsIfSameCenter(matchedRects1, qdc, matchedRects1, cv, distance);
    rectu::createRectsFromKeypoints(categorizedKp2, matchedRects2);
    rectu::mergeRectsIfSameCenter(matchedRects2, qdc, matchedRects2, cv, distance);
    rectu::filterIntersections(matchedRects1, matchedRects2, cv, matchedRects1, matchedRects2, cv);
    if (config.debug) {
      cout << "Quantized translation vectors: ";
      for (auto& p: cv) {
        cout << "(" << p.x << ", " << p.y << "), ";
      }
      cout << endl;
    }

    vector<PixelMatchingResult> matchingResults;
    vector<Rect> urects1, urects2;
    pixelMatch(img1, matchedRects1, img2, matchedRects2, cv, matchingResults, urects1, urects2);

    auto outImg1 = img1.clone();
    auto outImg2 = img2.clone();

    rectu::drawRects(outImg1, matchedRects1, Scalar(0, 255, 0), 1);
    rectu::drawRects(outImg1, urects1, Scalar(200, 200, 100), 1);
    rectu::drawRects(outImg2, matchedRects2, Scalar(0, 255, 0), 1);
    rectu::drawRects(outImg2, urects2, Scalar(200, 200, 100), 1);

    for (auto& pr: matchingResults) {
      if (!pr.isMatched) {
        rectu::drawRects(outImg1, pr.boundingMarkers1, Scalar(0, 0, 255), 1);
        rectu::drawRects(outImg2, pr.boundingMarkers2, Scalar(0, 0, 255), 1);
      }
    }

    imshow("rects1", outImg1);
    imshow("rects2", outImg2);
    waitKey(0);
  }
}

