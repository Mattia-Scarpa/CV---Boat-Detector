#include <fstream>
#include <sstream>
#include <iostream>
#include <numeric>

using namespace std;

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

using namespace cv;
using namespace dnn;



// Auxiliary structure
//Bounding box structuring element
struct bbox {
  Point center;       // boundingBox center Point
  float IoU;          // intersection over union
};

// Element defined by its Row and column in the boundingBos-IoU matrix
struct element {
  int i;
  int j;
};


// Parameters initialization
float objectnessThreshold = 0.5; // Objectness threshold
float confThreshold = .5f; // Confidence threshold
float nmsThreshold = .4f;  // Non-maximum suppression threshold
int inpWidth = 416;  // Width of network's input image
int inpHeight = 416; // Height of network's input image
vector<string> classes;
vector<Rect> predBoxes;
vector<Rect> trueBoxes;

// Command line parser function, keys accepted by command line parser
const string keys = {
  "{help h usage ?                |                   |   Boat detection using YOLO algorithm}"
  "{@modelPath nnPath p           |cfg/               |   -p --nnPath     \n\t\tSet up the model configuration files path\n}"
  "{@WeightsPath weights w        |cfg/               |   -w --weights    \n\t\tDefine the weights files paths\n}"
  "{@ConfigName cfg c             |yolo-obj.cfg       |   -c --cfg        \n\t\tDefine the configuration file name\n}"
  "{@ObjClasses obj o             |obj.names          |   -o --obj        \n\t\tDefine the file path containing classes names\n}"
  "{@imagePath image i            |test_result/kaggle/|   -i --images     \n\t\tDefine the path to the test images\n}"
  "{@falseMatchCount match m      |false              |   -m --match      \n\t\tDisplay the False Negative and False Positive count}"
};


// Utility function

// Remove the row and column of a vector of vector matrix
template <typename T>
void remove_element(vector<vector<T>>&v, int i, int j) {
  v.erase(v.begin()+i);
  for (size_t i = 0; i < v.size(); i++) {
    v[i].erase(v[i].begin()+j);
  }
}

// struct bbox function
void setBox(struct bbox& b, Point pts, float iou) {
  b.center = pts;
  b.IoU = iou;
}

// return the boundingBox with the highiest IoU over all the matrix and remove the corresponding row and column
bbox findMaxIoU(vector<vector<bbox>>& b) {
  float maxIoU = 0;
  bbox maxIoUBox;
  element el;
  for (size_t i = 0; i < b.size(); i++) {
    for (size_t j = 0; j < b[i].size(); j++) {
      if (maxIoU < b[i][j].IoU) {
        maxIoU = b[i][j].IoU;
        setBox(maxIoUBox, b[i][j].center, b[i][j].IoU);
        el.i = i;
        el.j = j;
      }
    }
  }
  if (0 < maxIoU) {
    remove_element(b, el.i, el.j);
  }
  else {
    setBox(maxIoUBox, Point(-1,-1), 0);
  }
  return maxIoUBox;
}

//Find the False negative quantinty depending on the null intersection remained
int countFalseNegative(vector<vector<bbox>> m) {
  int count = 0;
  if (0 < m.size()) {
    vector<float> cumulative(m[0].size(), 0.0f);

    for (size_t i = 0; i < m.size(); i++) {
      for (size_t j = 0; j < m[i].size(); j++) {
        cumulative[j] += m[i][j].IoU;
      }
    }
    for (size_t i = 0; i < cumulative.size(); i++) {
      if (cumulative[i] == 0) {
        count++;
      }
    }
  }
  return count;
}


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
void drawBox(int classId, float confidence, int xmin, int ymin, int xmax, int ymax, Mat& img, Scalar colorBox = Scalar(0,0,255)) {

  rectangle(img, Point(xmin, ymin), Point(xmax, ymax), colorBox, 2);

  string labelConfidence = format("%.2f", confidence*100);

  if (!classes.empty() && classId < (int)classes.size()) {
    labelConfidence = classes[classId] + ": " + labelConfidence + (char)37;
  }

  // Displaying the label at the top of its bounding box
  int baseLine;
  Size labelSize = getTextSize(labelConfidence, FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, 1, &baseLine);
  ymin = max(ymin, labelSize.height);
  putText(img, labelConfidence, Point(xmin, ymin), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,255),1);
}


// Remove low confidence labels

auto postPocess(Mat& img, const vector<Mat>& outs) {
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

      if (confidence > confThreshold) { // Filtering the detection below a certain confidence
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

    // Filter all the boxes that are outside the image and that has null area
    if (0 < box.width && 0 < box.height) {
      drawBox(classIds[index], confidences[index], box.x, box.y, box.x + box.width, box.y + box.height, img);
      // Save all the remaining drawn boxes
      predBoxes.push_back(box);
    }
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

// Main function
int main(int argc, char const *argv[]) {

  CommandLineParser parser(argc, argv, keys);
  parser.about("Boat detector with YOLO algorithm");
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }

  // Extraing command line informations and specifications
  string MODEL_PATH = "../" + parser.get<string>("@modelPath");

  String nnConfiguration = MODEL_PATH + parser.get<string>("@ConfigName");
  String nnWeightsPath = "../" + parser.get<string>("@WeightsPath");

  string classesFile = MODEL_PATH + parser.get<string>("@ObjClasses");
  ifstream ifs(classesFile.c_str());
  string line;
  while (getline(ifs, line)) classes.push_back(line);

  string imgPath = "../" + parser.get<string>("@imagePath");

  vector<string> imagesPath;
  vector<string> dataFormat = {imgPath+"*.jpg", imgPath+"*.png"};
  for (size_t i = 0; i < dataFormat.size(); i++) {
    vector<string> temp;
    glob(dataFormat[i], temp);
    for (size_t j(0); j < temp.size(); ++j) {
      imagesPath.push_back(temp[j]);
    }
  }
  cout << "A total of " << imagesPath.size() << " images to test has been found!" << endl;
  if (imagesPath.size() == 0) {
    cout << "No Images found, please check the folder: " << (char)126 << imgPath.substr(2, imgPath.length()) << endl;
    return 0;
  }

  vector<String> nnWeights;
  glob(nnWeightsPath+"*.weights", nnWeights);
  cout << "A total of " << nnWeights.size() << " weights files has been found!" << endl;
  if (nnWeights.size() == 0) {
    cout << "No weights found, please check the folder: " << (char)126 << nnWeightsPath.substr(2, nnWeightsPath.length()) << endl;
    cout << "Download the weights from: \nhttps://drive.google.com/drive/folders/1l9XJYxJBKEy6aWem1EERb68zj5qSdiZU?usp=sharing" << endl;
    return 0;
  }
  for (size_t i = 0; i < nnWeights.size(); i++) {
    cout << i << ": " << nnWeights[i] << endl;
  }
  cout << "Please select the wieghts file to use (type a number): ";
  int selection;
  cin >> selection;


  cout << "Loading darknet model configuration files and trained weights..." << endl;
  Net net = readNetFromDarknet(nnConfiguration, nnWeights[selection]);

  bool USE_GRADIENT;
  char grad;
  cout << "Do you want to perform detection from the gradient magnitude image? (y/n) ";
  cin >> grad;
  switch (grad) {
    case 'y':
      USE_GRADIENT = true;
    break;
    case 'Y':
      USE_GRADIENT = true;
    break;
    case 'n':
      USE_GRADIENT = false;
    break;
    case 'N':
      USE_GRADIENT = false;
    break;
    default:
      USE_GRADIENT = false;
    break;
  }

  for (size_t i = 0; i < imagesPath.size(); i++) {
    // clearing the vector of prediction and true boxes
    predBoxes.clear();
    trueBoxes.clear();
    // reading the image
    Mat img = imread(imagesPath[i]);

    Mat blob, src;

    if (USE_GRADIENT) {
      src = img.clone();
      src.convertTo(src, CV_32F, 1/255.0);
      Mat gx, gy;
      Sobel(src, gx, CV_32F, 1, 0, 1);
      Sobel(src, gy, CV_32F, 0, 1, 1);

      magnitude(gx, gy, src);
      src.convertTo(src, CV_8UC3, 255);
    }
    else {
      src = img.clone();
    }

    blobFromImage(src, blob, 1/255.0, Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);

    //Sets the input to the network
    net.setInput(blob);

    vector<Mat> outs;
    // Detection
    net.forward(outs, getOutputsNames(net));

    // Filter below a certain confidence and merge the overlapping box, then draw the rectangle
    postPocess(img, outs);

    // Look for ground truth txt file
    ifstream gTruth;
    gTruth.open(imagesPath[i].substr(0, imagesPath[i].length()-4)+".txt");
    if(!gTruth) {
      cout << "No labels files found for image: " << imagesPath[i] << endl;
    }

    float w;
    vector<float> absoluteCoordinates;
    // Extracting the ground truth bounding boxes
    while (gTruth >> w) { //Ignoring the label informations
      int i = 0;
      absoluteCoordinates.clear();
      while (i < 4) {
        gTruth >> w;
        absoluteCoordinates.push_back(w);
        i++;
      }
      // extracting the boxes coordinates
      int centerX = (int)(absoluteCoordinates[0] * img.cols);
      int centerY = (int)(absoluteCoordinates[1] * img.rows);
      int width = (int)(absoluteCoordinates[2] * img.cols);
      int height = (int)(absoluteCoordinates[3] * img.rows);
      int xmin = centerX - width / 2;
      int ymin = centerY - height / 2;
      trueBoxes.push_back(Rect(xmin, ymin, width, height));
    }

    // Calculate Intersection over Union
    vector<vector<bbox>> boxInfo;
    vector<vector<float>> IoUvalue;
    vector<bbox> boxInfoTemp;

    for (size_t i = 0; i < predBoxes.size(); i++) {
      boxInfoTemp.clear();
      bbox boxIoU;
      for (size_t j = 0; j < trueBoxes.size(); j++) {
        // calculating Intersection between true and predicted boxes
        Rect intersect = predBoxes[i] & trueBoxes[j];
        float intersectArea = intersect.width * intersect.height;
        float unionArea = (predBoxes[i].width * predBoxes[i].height)
         + (trueBoxes[j].width * trueBoxes[j].height) - intersectArea;
        float IoU = intersectArea / unionArea;
        setBox(boxIoU, Point(predBoxes[i].x+(predBoxes[i].width/2),
         predBoxes[i].y+(predBoxes[i].height/2)), IoU);
        boxInfoTemp.push_back(boxIoU);
      }
      boxInfo.push_back(boxInfoTemp);
    }

    // count false Negative
    int falseNegative = countFalseNegative(boxInfo);
    if (predBoxes.size() == 0) {
      falseNegative = trueBoxes.size(); // All the true boxes are false negative (if nothing has been detected)
    }
    int falsePositive = 0;
    int ln = 0;

    // count False positive (extra functions not complete)
    if (0 < boxInfo.size()) {
      falsePositive = boxInfo.size() - boxInfo[0].size() + falseNegative;
      if (falsePositive < 0) {  // reset if false positive is Negative
        falsePositive= 0;
      }
      ln = min(boxInfo.size(), boxInfo[0].size());
    }

    cout << "\n-----------------------------------------------------------------" << endl;
    cout << "Image: " << imagesPath[i].substr(imgPath.length(), imagesPath[i].length()) << endl;
    for (size_t k = 0; k < ln; k++) {
      bbox maxBox = findMaxIoU(boxInfo);
      if (0 < maxBox.IoU) {
        std::cout << "Boat found in position at: " << maxBox.center << "\t IoU = " << maxBox.IoU << '\n';
      }
    }
    cout << "False Negative: " << falseNegative << endl;
    cout << "False Positive: " << falsePositive << endl;

    namedWindow("Detection", WINDOW_NORMAL);
    imshow("Detection", img);
    waitKey();
  }
  return 0;
}
