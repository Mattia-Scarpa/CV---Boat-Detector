#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>

using namespace std;

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

#include <labeltxt.h>
#include <dataugmentation.h>

/*
#include <json.hpp>
#include <iomanip>

using json = nlohmann::json;*/

// Command line parser function, keys accepted by command line parser
const string keys = {
  "{help h usage ?                |           |   Implementation of the label formatting from a standard JSON annotation to a YOLO-Darknet txt files & implementation of a data augmentation pre process}"
  "{@useValidation useval v       |false      |   -u --useval   \n\t\tSpecify if the validation images has to be taken form a different folder than that of the train set\n}"
  "{@validationRatio vratio r     |.25        |   -r --vratio   \n\t\tSet up the train-validation images ratio\n}"
  "{@trainSubdir tdir t           |train/     |   -t --tdir     \n\t\tSet up the train subdirectory from data/ (e.g., 'train/')\n}"
  "{@valSubdir vdir v             |test/      |   -t --tdir     \n\t\tSet up the train subdirectory from data/ (e.g., 'train/')\n}"
  "{@formatLabel label l          |true       |   -l --label    \n\t\tExtract YOLO-Darknet txt files from standard JSON boats annotation\n}"
  "{@setEmptyClass empty e        |water      |   -e --empty    \n\t\tSet label class name for image without objects of interest\n}"
  "{@doAugmentation aug a         |true       |   -a --aug      \n\t\tSet true to perform data augmentation on the train dataset\n}"
  "{@Classification class c       |false      |   -c --class    \n\t\tSet true also to classify the images as well as to identify the generic objects (e.g., boats)\n}"
  "{@Object obj o                 |boat       |   -o --obj      \n\t\tDefine the generic object name (needed if Classification=false)\n}"
  "{@Perspective ptransf p        |3          |   -p --ptransf  \n\t\tDefine the number of increasing perspective transformation\n}"
};


template <typename T>
T remove_at(std::vector<T>&v, typename std::vector<T>::size_type n) {
    T ans = std::move_if_noexcept(v[n]);
    v[n] = std::move_if_noexcept(v.back());
    v.pop_back();
    return ans;
}

float VAL_RATIO;
int VAL_SIZE;


int main(int argc, char const *argv[]) {

  CommandLineParser parser(argc, argv, keys);
  parser.about("Boat detector pre-process");
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }

  bool useVal = parser.get<bool>("@useValidation");
  if (!useVal) {
     VAL_RATIO = parser.get<float>("@validationRatio");
  }

  // Get labeling requirement
  string emptyClass = parser.get<string>("@setEmptyClass");
  bool DO_LABEL = parser.get<bool>("@formatLabel");
  bool DO_AUGMENTATION = parser.get<bool>("@doAugmentation");
  bool DO_CLASSIFICATION = parser.get<bool>("@Classification");
  string OBJ_NAME = parser.get<string>("@Object");
  int PERSPECTIVE_COUNT = parser.get<int>("@Perspective");
  string DATAPATH_TRAIN = "../data/"+parser.get<string>("@trainSubdir");
  string DATAPATH_TEST = "../data/"+parser.get<string>("@valSubdir");


  // ----------------------------------------------------------------------------


  vector<string> imagesPath, labelsPath;
  // Reading images pathtest
  glob(DATAPATH_TRAIN+"*.jpg", imagesPath);
  cout << "A total of " << imagesPath.size() << " images has been found" << endl;
  // Reading labels path
  glob(DATAPATH_TRAIN+"*.json", labelsPath);
  cout << "A total of " << labelsPath.size() << " labels has been found" << endl;

/*for (size_t i = 0; i < labelsPath.size(); i++) {
    ifstream uLabel(labelsPath[i]);
    json raw;
    uLabel >> raw;

    for (size_t j = 0; j < raw["annotations"].size(); j++) {
      cout << (raw["annotations"][j]["label"] == "lanciafino" || raw["annotations"][j]["label"] == "lanciafino marrone" || raw["annotations"][j]["label"] == "lanciamaggiore") << endl;
      if (raw["annotations"][j]["label"] == "lanciafino" || raw["annotations"][j]["label"] == "lanciafino marrone" || raw["annotations"][j]["label"] == "lanciamaggiore") {
        json labelmodified = R"({"label":"lancia"})"_json;
        raw["annotations"][j].update(labelmodified);
        cout << raw["annotations"][j]["label"] << endl;
        std::ofstream out(labelsPath[i]);
        out << std::setw(4) << raw << endl;
      }
    }
  }*/



  // Set up the validation size if dataset split is required otherwise get path infos
  vector<string> imagesPathVal, labelsPathVal;

  if (!useVal) {
    VAL_SIZE = imagesPath.size() * VAL_RATIO;
  } else {
    // Reading images path
    glob(DATAPATH_TEST+"*.jpg", imagesPathVal);
    cout << "A total of " << imagesPathVal.size() << " images has been found" << endl;
    // Reading labels path
    glob(DATAPATH_TEST+"*.json", labelsPathVal);
    cout << "A total of " << labelsPathVal.size() << " labels has been found" << endl;
  }


  // Manipulating boats labels annotation from JSON format to YOLO-darknet txt file (if required)

  labeltxt* labelParser = new labeltxt();
  dataugmentation* augment = new dataugmentation();

  vector<string> augmentedPath, validationPath;

  unsigned int microsecond = 1000;


  if (!useVal) {
    cout << "Extracting validation images: " << VAL_SIZE << " images..." << endl;
    for (size_t i = 0; i < VAL_SIZE; i++) {
      srand(time(0));
      int index = rand() % imagesPath.size() + 1;
      string valImgPath = remove_at(imagesPath, index);
      string valLabPath = remove_at(labelsPath, index);

      validationPath.push_back(valImgPath);

      labelParser->setJsonPath(valLabPath);
      labelParser->setImg(valImgPath);
      labelParser->extractLabelsCoordinates(emptyClass, DO_LABEL, DO_CLASSIFICATION, OBJ_NAME);
    }
  }
  else {
    cout << "getting validation images: " << labelsPathVal.size() << "images..." << endl;
    for (size_t i = 0; i < labelsPathVal.size(); i++) {
      labelParser->setJsonPath(labelsPathVal[i]);
      labelParser->setImg(imagesPathVal[i]);
      labelParser->extractLabelsCoordinates(emptyClass, DO_LABEL, DO_CLASSIFICATION, OBJ_NAME);
    }
  }



  cout << "Performing labeling formatting and data augmentation..." << endl;

  for (size_t i(0); i < labelsPath.size(); i++) {
    labelParser->setJsonPath(labelsPath[i]);
    labelParser->setImg(imagesPath[i]);
    labelParser->extractLabelsCoordinates(emptyClass, DO_LABEL, DO_CLASSIFICATION, OBJ_NAME);

    if (DO_AUGMENTATION) {
      Mat srcImg = labelParser->getImage();

      augment->setImageInfo(srcImg, imagesPath[i], labelParser->corners, labelParser->getClassNumber());

      augment->equalize(srcImg);
      if (!augment->augmentedImagePath.empty()) {
        augmentedPath.push_back(augment->augmentedImagePath);
      }

      augment->changeContrast(srcImg);
      if (!augment->augmentedImagePath.empty()) {
        augmentedPath.push_back(augment->augmentedImagePath);
      }

      augment->changeBrightness(srcImg);
      if (!augment->augmentedImagePath.empty()) {
        augmentedPath.push_back(augment->augmentedImagePath);
      }

      augment->gaussianSmooth(srcImg);
      if (!augment->augmentedImagePath.empty()) {
        augmentedPath.push_back(augment->augmentedImagePath);
      }

      for (size_t i = 0; i < PERSPECTIVE_COUNT; i++) {
        float sigma = 1.0f/(PERSPECTIVE_COUNT-i);
        augment->changePerspective(srcImg, sigma, i);
        if (!augment->augmentedImagePath.empty()) {
          augmentedPath.push_back(augment->augmentedImagePath);
        }
      }
    }


  }



  // ----------------------------------------------------------------------------
  // ----------------------------------------------------------------------------
  // ----------------------------------------------------------------------------


  // Creating the validation dataset (1000 items as default)

  cout << "Writing conf datafiles" << endl;

  ofstream valFile, testFile, objNamesFile, objDataFile;
  int j = imagesPath[0].find("data/");

  remove((imagesPath[0].substr(0, j+5)+"test.txt").c_str());
  remove((imagesPath[0].substr(0, j+5)+"train.txt").c_str());
  remove((imagesPath[0].substr(0, j+5)+"obj.names").c_str());
  remove((imagesPath[0].substr(0, j+5)+"obj.data").c_str());

  valFile.open(imagesPath[0].substr(0, j+5)+"test.txt", ios::out | ios::app);
  testFile.open(imagesPath[0].substr(0, j+5)+"train.txt", ios::out | ios::app);
  objNamesFile.open(imagesPath[0].substr(0, j+5)+"obj.names", ios::out | ios::app);
  objDataFile.open(imagesPath[0].substr(0, j+5)+"obj.data", ios::out | ios::app);

  if (!useVal) {
    cout << "Exporting validation path: " << VAL_SIZE << " images..." << endl;
    for (size_t i = 0; i < validationPath.size(); i++) {
      valFile << validationPath[i].substr(j, validationPath[i].length()) << endl;
    }
  }
  else {
    cout << "Exporting validation images path: " << imagesPathVal.size() << " images..." << endl;
    for (size_t i = 0; i < labelsPathVal.size(); i++) {
      valFile << imagesPathVal[i].substr(j, imagesPathVal[i].length()) << endl;
    }
  }

  cout << "Exporting train images path: " << imagesPath.size() << " images..." << endl;
  for (size_t i = 0; i < imagesPath.size(); i++) {
    testFile << imagesPath[i].substr(j, imagesPath[i].length()) << endl;
  }
  cout << "Exporting augmented images path: " << augmentedPath.size() << " images..." << endl;
  for (size_t i = 0; i < augmentedPath.size(); i++) {
    //cout << augmentedPath[i].substr(j, imagesPath[i].length()) << endl;
    testFile << augmentedPath[i].substr(j, augmentedPath[i].length()) << endl;
    //cout << augmentedPath[i].substr(j, augmentedPath[i].length()) << endl;
    usleep(5 * microsecond);//sleeps for 5 milliseconds
  }

  valFile.close();
  testFile.close();

  vector<string> labelList = labelParser->getLabelsList();

  for (size_t i = 0; i < labelList.size(); i++) {
    objNamesFile << labelList[i] << endl;
  }

  cout << "Generating configuration files ..." << endl;
  objDataFile << "classes = " << labelList.size() << endl;
  objDataFile << "train  = data/train.txt" << endl;
  objDataFile << "valid  = data/test.txt" << endl;
  objDataFile << "names = data/obj.names" << endl;
  objDataFile << "backup = backup/" << endl;

  objNamesFile.close();
  objDataFile.close();

  cout << "DONE!" << endl;


  return 0;
}
