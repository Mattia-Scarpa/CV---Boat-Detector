#include <opencv2/core.hpp>
#include <json.hpp>

class labeltxt {

public:

  // Constructors

  labeltxt();

  //----------------------------------------------------------------------------
  // Public functions

  void setJsonPath(std::string path);

  void setImg(std::string path);

  std::string getJsonRootPath();

  cv::Mat getImage();

  std::string getImageName();

  void extractLabelsCoordinates(std::string emptyClass = "water", bool wtxt = true, bool classify = false, std::string obj = "boat"); // write txt

  std::vector<std::string> getLabelsList();

  void annotationAllignement();

  std::vector<int> getClassNumber();


  // Public variables

  std::vector<std::vector<cv::Point2f>> corners;


private:

  // Utility functions

  int labelToInt(std::string label);

  struct labels {
    std::vector<int> objectClass;
    std::vector<cv::Point2f> absCenter;
    std::vector<float> width;
    std::vector<float> height;
  };

  void clearObjClass(struct labels& lab) {
    lab.objectClass.clear();
    lab.absCenter.clear();
    lab.width.clear();
    lab.height.clear();
  }

  std::string rootPath;
  std::string imgName;
  nlohmann::json rawLabels;
  std::vector<std::string> usedLabels;
  cv::Mat img;
  struct labels label;

};
