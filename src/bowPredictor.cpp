#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

using namespace cv;

/*
// structuring elements

struct bbox {
  vector<string> labels;
  vector<Point> minP;
  vector<Point> maxP;
};

void addBox(struct bbox& lab, string class, Point min, Point max) {
    labels.push_back(class);
    minP.push_back(min);
    maxP.push_back(max);
}

void clearBox(struct bbox& lab) {
  labels.clear();
  minP.clear();
  maxP.clear();
}

    addBox(predictedBoxes, classes[classIds[index]], Point(box.x, box.y), Point(box.x + box.width, box.y + box.height));
*/

// Setting up the command line parsers
const string keys = {
  "{help h usage ?                |                 |   Boat detection using YOLO algorithm}"
  "{@trainImgList train t         |data/train.txt   |   -t --train     \n\t\tDefine the full path to the train images list file\n}"
};


int main(int argc, char const *argv[]) {

  CommandLineParser parser(argc, argv, keys);
  parser.about("Boat detector pre-process");
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }

  vector<string> trainImgPath;
  string trainListFile = "test";




  return 0;
}
