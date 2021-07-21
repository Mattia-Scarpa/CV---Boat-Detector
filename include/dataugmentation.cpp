#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>

using namespace std;

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

#include <dataugmentation.h>


// Utility functions

void dataugmentation::allignAnnotation(vector<vector<Point2f>>& boxCorners) {
  for (size_t i = 0; i < bbox.size(); i++) {
    vector<float> x = {bbox[i][0].x, bbox[i][1].x, bbox[i][2].x, bbox[i][3].x};
    vector<float> y = {bbox[i][0].y, bbox[i][1].y, bbox[i][2].y, bbox[i][3].y};

    float xmin = *min_element(x.begin(), x.end());
    float xmax = *max_element(x.begin(), x.end());
    float ymin = *min_element(y.begin(), y.end());
    float ymax = *max_element(y.begin(), y.end());

    vector<Point2f> boxCornersTemp = {Point2f(xmin, ymin), Point2f(xmin, ymax), Point2f(xmax,ymax), Point2f(xmax,ymin)};
    boxCorners.push_back(boxCornersTemp);
  }
}

void dataugmentation::saveAndWritetxt(Mat img, string imgPath, vector<int> classNumber, vector<vector<Point2f>> boxCorners, string augType, int count) {

  stringstream ss;
  string c;
  ss << count;
  ss >> c;

  if (imwrite(imgPath.substr(0, imgPath.length()-4)+"_"+augType+"-"+c+".jpg", img)) {
    augmentedImagePath = imgPath.substr(0, imgPath.length()-4)+"_"+augType+"-"+c+".jpg";

    ofstream objLabelsFiles;
    objLabelsFiles.open(imgPath.substr(0, imgPath.length()-4)+"_"+augType+"-"+c+".txt");

    if (!classNumber.empty()) {
      for (size_t i = 0; i < classNumber.size(); i++) {
        float xmin = boxCorners[i][0].x;
        float ymin = boxCorners[i][0].y;
        float xmax = boxCorners[i][2].x;
        float ymax = boxCorners[i][2].y;

        Point2f center(xmin+(xmax-xmin)/2, ymin+(ymax-ymin)/2);
        Point2f absoluteCenter(center.x/img.cols, center.y/img.rows);
        objLabelsFiles << classNumber[i] << " " << absoluteCenter.x << " " << absoluteCenter.y << " " << (xmax-xmin)/img.cols << " " << (ymax-ymin)/img.rows << endl;
      }
    }
    objLabelsFiles.close();
  }
  else {
    imshow("saveImg", img);
    waitKey();
    augmentedImagePath.clear();
  }

}

// Constructors

dataugmentation::dataugmentation() {}


// -----------------------------------------------------------------------------
// Public functions

void dataugmentation::setImageInfo(Mat src, string imgPath, vector<vector<Point2f>> boxCorners, vector<int> classN, bool save) {
  img = src.clone();
  path = imgPath;
  bbox.clear();
  bbox = boxCorners;
  classNumber = classN;
  annotate = save;
}

void dataugmentation::equalize(Mat& dst, int count) {
  vector<Mat> img_channels;
  split(img, img_channels);

  equalizeHist(img_channels[2], img_channels[2]);
  equalizeHist(img_channels[1], img_channels[1]);
  equalizeHist(img_channels[0], img_channels[0]);

  merge(img_channels, dst);

  if (annotate) {
    saveAndWritetxt(dst, path, classNumber, bbox, "equalizationAug", count);
  }
}

void dataugmentation::equalize(int count) {
  vector<Mat> img_channels;
  split(img, img_channels);

  equalizeHist(img_channels[2], img_channels[2]);
  equalizeHist(img_channels[1], img_channels[1]);
  equalizeHist(img_channels[0], img_channels[0]);
  Mat dst;
  merge(img_channels, dst);

  if (annotate) {
    saveAndWritetxt(dst, path, classNumber, bbox, "equalizationAug", count);
  }
}

void dataugmentation::changeContrast(Mat& dst, int count) {
  srand(time(0));
  double contrastPercentage = (rand()%500)/10.0;
  dst = img.clone();
  dst.convertTo(dst, CV_64F);
  dst = dst*(1+contrastPercentage/100.0);
  dst.convertTo(dst, CV_8U);

  if (annotate) {
    saveAndWritetxt(dst, path, classNumber, bbox, "constrastAug", count);
  }
}

void dataugmentation::changeContrast(int count) {
  srand(time(0));
  double contrastPercentage = (rand()%500)/10.0;
  Mat dst = img.clone();
  dst.convertTo(dst, CV_64F);
  cout << 1+contrastPercentage/100.0 << endl;
  dst = dst*(1+contrastPercentage/100.0);
  dst.convertTo(dst, CV_8U);

  if (annotate) {
    saveAndWritetxt(dst, path, classNumber, bbox, "constrastAug", count);
  }
}

void dataugmentation::changeBrightness(Mat& dst, int count) {
  srand(time(0));
  double brightnessoffset = 50-((rand()%1000)/10.0);
  dst = img.clone();
  Mat dstCh[3];
  split(dst, dstCh);

  for (size_t i = 0; i < 3; i++) {
    add(dstCh[i], brightnessoffset, dstCh[i]);
  }
  merge(dstCh,3,dst);

  if (annotate) {
    saveAndWritetxt(dst, path, classNumber, bbox, "brightnessAug", count);
  }
}

void dataugmentation::changeBrightness(int count) {
  srand(time(0));
  double brightnessoffset = 50-((rand()%1000)/10.0);
  Mat dst = img.clone();
  Mat dstCh[3];
  split(dst, dstCh);

  for (size_t i = 0; i < 3; i++) {
    add(dstCh[i], brightnessoffset, dstCh[i]);
  }
  merge(dstCh,3,dst);

  if (annotate) {
    saveAndWritetxt(dst, path, classNumber, bbox, "brightnessAug", count);
  }
}

void dataugmentation::gaussianSmooth(Mat& dst, double sigma, int count) {
  GaussianBlur(img, dst, Size(0,0), sigma);
  if (annotate) {
    saveAndWritetxt(dst, path, classNumber, bbox, "gaussianAug", count);
  }
}

void dataugmentation::gaussianSmooth(double sigma, int count) {
  Mat dst;
  GaussianBlur(img, dst, Size(0,0), sigma);
  if (annotate) {
    saveAndWritetxt(dst, path, classNumber, bbox, "gaussianAug", count);
  }
}

void dataugmentation::changePerspective(Mat& dst, float sigma, int count) {

  int H = img.rows-1;
  int W = img.cols-1;
  vector<Point2f> imgPts = {Point2f(0, 0), Point2f(0, H), Point2f(W, H), Point2f(W, 0)};
  srand(time(0));
  float hyp = 0.3*sigma;          // hyperparameter definition

  // chosing randomly destination points
  vector<Point2f> dstPts = {Point2f(rand()%int(W*hyp), rand()%int(H*hyp)), Point2f(rand()%int(W*hyp), H-rand()%int(H*hyp)), Point2f(W-rand()%int(W*hyp), H-rand()%int(H*hyp)), Point2f(W-rand()%int(W*hyp), rand()%int(H*hyp))};

  Mat warp = findHomography(imgPts, dstPts);    // Transformation matrix estimation
  dst = Mat::zeros( img.rows, img.cols, img.type() );
  warpPerspective(img, dst, warp, dst.size());  // Image transformation

  if (!bbox.empty()) {
    for (size_t i = 0; i < bbox.size(); i++) {
      perspectiveTransform(bbox[i], bbox[i], warp);
    }
    // annotations allignment
    vector<vector<Point2f>> allignedBox;
    allignAnnotation(allignedBox);
  // save image and write label
    if (annotate) {
      saveAndWritetxt(dst, path, classNumber, allignedBox, "perspectiveAug", count);
    }
  }
  else {
    // save image and write label
    if (annotate) {
      saveAndWritetxt(dst, path, classNumber, bbox, "perspectiveAug", count);
    }
  }
}

void dataugmentation::changePerspective(float sigma, int count) {
  // Point2f imgPts[4], dstPts[4];

  int H = img.rows-1;
  int W = img.cols-1;
  vector<Point2f> imgPts = {Point2f(0, 0), Point2f(0, H), Point2f(W, H), Point2f(W, 0)}; // initial points coordinates (image vertex)
  srand(time(0));
  float hyp = 0.3*sigma;          // hyperparameter definition

  // chosing randomly destination points
  vector<Point2f> dstPts = {Point2f(rand()%int(W*hyp), rand()%int(H*hyp)), Point2f(rand()%int(W*hyp), H-rand()%int(H*hyp)), Point2f(W-rand()%int(W*hyp), H-rand()%int(H*hyp)), Point2f(W-rand()%int(W*hyp), rand()%int(H*hyp))};

  Mat warp = findHomography(imgPts, dstPts);    // Transformation matrix estimation
  Mat dst = Mat::zeros( img.rows, img.cols, img.type());
  warpPerspective(img, dst, warp, dst.size());  // Image transformation

  if (!bbox.empty()) {
    for (size_t i = 0; i < bbox.size(); i++) {
      vector<Point2f> bboxTemp;
      perspectiveTransform(bbox, bboxTemp, warp);
    }
    // annotations allignment
    vector<vector<Point2f>> allignedBox;
    allignAnnotation(allignedBox);
  // save image and write label
  if (annotate) {
      saveAndWritetxt(dst, path, classNumber, allignedBox, "perspectiveAug", count);
    }
  }
  else {
    // save image and write label
    if (annotate) {
      saveAndWritetxt(dst, path, classNumber, bbox, "perspectiveAug", count);
    }
  }
}
