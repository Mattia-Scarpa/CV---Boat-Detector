#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

#include <json.hpp>

#include <labeltxt.h>

using namespace std;
using json = nlohmann::json;

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;


// Utility functions

int labeltxt::labelToInt(string label) {
  auto idx = find(usedLabels.begin(), usedLabels.end(), label);

  if (idx != usedLabels.end()) {
    return distance(usedLabels.begin(), idx);
  } else {
    usedLabels.push_back(label);
    idx = find(usedLabels.begin(), usedLabels.end(), label);
    return distance(usedLabels.begin(), idx);
  }
}


// Constructors

labeltxt::labeltxt() {}


// -----------------------------------------------------------------------------
// Public functions

void labeltxt::setJsonPath(string path) {
  ifstream uLabel(path);
  uLabel >> rawLabels;
  // extracting JSON dictionary
  imgName = string(rawLabels["name"]);
  rootPath = path.substr(0, path.length() - imgName.length()-1);
  imgName = imgName.substr(0, imgName.length()-4);
}

void labeltxt::setImg(string path) {
  img = imread(path);
}

string labeltxt::getImageName() {
  return imgName;
}

string labeltxt::getJsonRootPath() {
  return rootPath;
}

Mat labeltxt::getImage() {
  return img;
}

void labeltxt::extractLabelsCoordinates(string emptyClass, bool wtxt, bool classify, string obj) {
  clearObjClass(label);
  corners.clear();

  // Check if there is labels or if the image has been classified as negative image (water class)
  if (int(rawLabels["annotationsCount"]) != 0 && string(rawLabels["annotations"][0]["label"]) != emptyClass) {
    for (size_t i(0); i < rawLabels["annotations"].size(); i++) {

      // Extracting coordinates of each label in the image
      if (rawLabels["annotations"][i]["type"] == "box") {
        float xmin = rawLabels["annotations"][i]["coordinates"][0]["x"];
        float ymin = rawLabels["annotations"][i]["coordinates"][0]["y"];
        float xmax = rawLabels["annotations"][i]["coordinates"][1]["x"];
        float ymax = rawLabels["annotations"][i]["coordinates"][1]["y"];

        vector<Point2f> cornerstemp = {Point2f(xmin, ymin), Point2f(xmin,ymax), Point2f(xmax,ymax), Point2f(xmax, ymin)};
        corners.push_back(cornerstemp);

        // che the center of each bounding box
        Point2f center(xmin+(xmax-xmin)/2, ymin+(ymax-ymin)/2);
        Point2f absoluteCenter(center.x/img.cols, center.y/img.rows);

        int numClass;
        if (classify) {
        numClass = labelToInt(rawLabels["annotations"][i]["label"]);
        }
        else {
          numClass = 0;
          usedLabels = {obj};
        }

        label.objectClass.push_back(numClass);
        label.absCenter.push_back(absoluteCenter);
        label.width.push_back((xmax-xmin)/img.cols);
        label.height.push_back((ymax-ymin)/img.rows);
      }
    }

    if (wtxt) {
      ofstream objLabelsFiles;

      objLabelsFiles.open(rootPath+imgName+".txt", ios::out | ios::app);

      for(size_t j(0); j < label.objectClass.size(); j++) {
        objLabelsFiles << label.objectClass[j] << " " << label.absCenter[j].x << " " << label.absCenter[j].y << " " << label.width[j] << " " << label.height[j] << endl;
      }
      objLabelsFiles.close();
      //clearObjClass(label);
    }
    else {
      //clearObjClass(label);
    }
  }
  else {
    if (wtxt) {
      ofstream objLabelsFiles;
      objLabelsFiles.open(rootPath+imgName+".txt", ios::out | ios::app);
      objLabelsFiles.close();
    }
  }
}

vector<string> labeltxt::getLabelsList() {
  return usedLabels;
}

vector<int> labeltxt::getClassNumber() {
  return label.objectClass;
}
