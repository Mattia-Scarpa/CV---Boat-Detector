#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;


// Initialize the parameters
float objectnessThreshold = 0.5; // Objectness threshold
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;  // Width of network's input image
int inpHeight = 416; // Height of network's input image
vector<string> classes;

// Command line parser function, keys accepted by command line parser
const string keys = {
  "{help h usage ?                |                 |   Label formatting from a standard JSON annotation to a YOLO-Darknet txt files &  data augmentation preprocess}"
  "{@modelPath nnPath p           |darknet_cfg/     |   -p --nnPath     \n\t\tSet up the model configuration files path}"
  "{@ConfigName cfg c             |yolo-obj.cfg     |   -c --cfg        \n\t\tDefine the configuration file name}"
  "{@WeightsName weights w        |yolo-obj.weights |   -w --weights    \n\t\tDefine the weights file name}"
  "{@imagePath image i            |img/             |   -i --images     \n\t\tDefine the path to the test images}"
};


// Utilities functions


// Last layer identification
auto getOutputsNames(const Net& net) {
  static vector<string> names;

  if (names.empty()) {
    //Get the indices of the output layers, i.e. the layers with unconnected outputs
    vector<int> outLayers = net.getUnconnectedOutLayers();
    // Extract all layers names in the network
    vector<String> layersNames = net.getLayerNames();

    // Get the outputLayers names
    names.resize(outLayers.size());
    for (size_t i = 0; i < outLayers.size(); i++) {
      names[i] = layersNames[outLayers[i]-1];
    }
  }
  return names;
}


// Draw the predicted bounding box
void drawBox(int classId, float confidence, int xmin, int ymin, int xmax, int ymax, Mat& img) {

  rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(0,50,255), 2);

  string labelConfidence = format("%.2f", confidence);

  if (!classes.empty() && classId < (int)classes.size()) {
    labelConfidence = classes[classId] + ": " + labelConfidence;
  }

  // Displaying the label at the top of its bounding box
  int baseLine;
  Size labelSize = getTextSize(labelConfidence, FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, 1, &baseLine);
  ymin = max(ymin, labelSize.height);
  rectangle(img, Point(xmin, ymin - round(1.5*labelSize.height)), Point(xmin + round(1.5*labelSize.width), ymin + baseLine), Scalar(255, 255, 255), FILLED);
    putText(img, labelConfidence, Point(xmin, ymin), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);
}


// Remove low confidence labels

void postPocess(Mat& img, const vector<Mat>& outs) {
  vector<int> classIds;
  vector<float> confidences;
  vector<Rect> boxes;

  for (size_t i = 0; i < outs.size(); i++) {
    // Scan through all the bounding boxes output from the network and keep only the
    // ones with high confidence scores. Assign the box's class label as the class
    // with the highest score for the box.

    float* data = (float*)outs[i].data;
    for (size_t j = 0; j < outs[i].rows; j++, data += outs[i].cols) {

      Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
      Point classIdPoint;
      double confidence;

      minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

      if (confidence > confThreshold) {
        int centerX = (int)(data[0] * img.cols);
        int centerY = (int)(data[1] * img.rows);
        int width = (int)(data[2] * img.cols);
        int height = (int)(data[3] * img.rows);
        int xmin = centerX - width / 2;
        int ymin = centerY - height / 2;

        classIds.push_back(classIdPoint.x);
        confidences.push_back(confidence);
        boxes.push_back(Rect(xmin, ymin, width, height));

      }
    }
  }

  // Non Maxima Suppression (NMS)

  vector<int> indices;
  NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

  for (size_t i = 0; i < indices.size(); i++) {
    int index = indices[i];
    Rect box = boxes[index];
    drawBox(classIds[index], confidences[index], box.x, box.y, box.x + box.width, box.y + box.height, img);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------


int main(int argc, char const *argv[]) {

  CommandLineParser parser(argc, argv, keys);
  parser.about("Boat detector pre-process");
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }


  string MODEL_PATH = "../" + parser.get<string>("@modelPath");

  String nnConfiguration = MODEL_PATH + parser.get<string>("@ConfigName");
  String nnWeights = MODEL_PATH + parser.get<string>("@WeightsName");


  cout << "Loading darknet models configuration file and trained weights..." << endl;
  Net net = readNetFromDarknet(nnConfiguration, nnWeights);










  return 0;
}










































//------------------------------------------------------------------------------
