#ifndef DATAUGMENTETION_H
#define DATAUGMENTETION_H

#include <opencv2/core.hpp>

class dataugmentation {

public:

  // Constructor

  dataugmentation();

  //----------------------------------------------------------------------------
  // Public functions

  void setImageInfo(cv::Mat src, string imgName,
    std::vector<std::vector<cv::Point2f>> boxCorners,
    std::vector<int> classN, bool save = true);

  // equalization
  void equalize(cv::Mat& dst, int count = 0);

  void equalize(int count = 0);

  // contrast change
  void changeContrast(cv::Mat& dst, int count = 0);

  void changeContrast(int count = 0);

  // brightness change
  void changeBrightness(cv::Mat& dst, int count = 0);

  void changeBrightness(int count = 0);

  // gaussian filtering change
  void gaussianSmooth(cv::Mat& dst, double sigma = 3,
    int count = 0);

  void gaussianSmooth(double sigma = 3, int count = 0);

  // Perspective transformation
  void changePerspective(cv::Mat& dst, float sigma = 0.5,
    int count = 0);

  void changePerspective(float sigma = 0.5, int count =0);

  string augmentedImagePath;

private:

  void allignAnnotation(std::vector<std::vector<cv::Point2f>>& boxCorners);

  void saveAndWritetxt(
    cv::Mat img, std::string imgPath, std::vector<int> classNumber,
    std::vector<std::vector<cv::Point2f>> boxCorners,
    std::string augType, int count);


  cv::Mat img;
  string path;
  std::vector<std::vector<cv::Point2f>> bbox;
  std::vector<int> classNumber;
  bool annotate;
};
#endif
