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

string MODEL_PATH = 


int main(int argc, char const *argv[]) {




  return 0;
}
