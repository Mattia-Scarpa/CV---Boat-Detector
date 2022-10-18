#ifndef LABELTXT_H
#define LABELTXT_H

#include <opencv2/core.hpp>
#include <json.hpp>

class labeltxt {

public:

  // Constructors

  labeltxt();

  //----------------------------------------------------------------------------
  // Public functions

  void setJsonPath(std::string path); // Specify the path of the JSON annotation file

  void setImg(std::string path);      // Specify the path of the image

  std::string getJsonRootPath();      // Return the JSON path

  cv::Mat getImage();                 // Return Image

  std::string getImageName();         // Return Image name (found in the JSON annotation)

  void extractLabelsCoordinates(      // Extract and convert annotation label and box
    std::string emptyClass = "water", bool wtxt = true,
    bool classify = false, std::string obj = "boat");

  std::vector<std::string> getLabelsList(); // Return all the used label

  std::vector<int> getClassNumber();        // Return all the used label mapped as int


  // Public variables

  std::vector<std::vector<cv::Point2f>> corners;


private:

  // Utility functions

  int labelToInt(std::string label);

  // labels characteristics structuring elements
  struct labels {
    std::vector<int> objectClass;
    std::vector<cv::Point2f> absCenter;
    std::vector<float> width;
    std::vector<float> height;
  };

  // clear the label structuring elements
  void clearObjClass(struct labels& lab) {
    lab.objectClass.clear();
    lab.absCenter.clear();
    lab.width.clear();
    lab.height.clear();
  }

  // variables
  std::string rootPath;
  std::string imgName;
  nlohmann::json rawLabels;
  std::vector<std::string> usedLabels;
  cv::Mat img;
  struct labels label;

};
#endif
