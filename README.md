# Computer Vision - Boat Detector

## Introduction

In this project, it was required to perform boat detection from single frames that can be useful in automated analysis of boat traffic in the sea.
Different techniques is used in Computer Vision in order to perform object detection of any kind (e.g., resorting to HOG descriptors combined with Machine Learning algorithm, such SVM, or Bag of Words image classification), however, for this project it has been chosen a Convolutional Neural Network (CNN) in order to deal with the the high variability with which a generic boat can present itself. In particualr the YOLO algorithm has been exploited to perform this task. 
YOLO (You Only Look Once) algorithm employs CNN to detect object in real-time; the CNN is used to predict various class probabilities and bounding box simultaneously. The YOLO object detector has been trained using the darknet open-source neural network framework with a dataset of boats images found from the kaggle platform at the link: https://www.kaggle.com/clorichel/boat-types-recognition/.
All the images have been manually annotated and from the annotation json files obtained it has been extracted the labels in txt format used by the darknet for training. Three approaches have been adopted to modify the images that have fed the darknet neural networks, then, the result it has been tested with the YOLO algorithm on a set of test images.

## Data Annotation

In the kaggle dataset all the images was divided in class depending on the specific boat (e.g., sail boat, ferry boat etc.) but, for the purpose of the project, every boats has been grouped in a single class. In each image a bounding box , defined by the coordinates <img src="https://render.githubusercontent.com/render/math?math=\color{white}(x_{min}, y_{min}), (x_{max}, y_{max})">, has been used to highlight each boat present on the scene.
The images has been annotated using the website https://dataloop.ai/ which return the annotations on a dataset of JSON files, one for each image, containing all the labels information. To parse all the necessary information contained in the JSON files a dedicated class in C++, which makes use of the library json.hpp _https://github.com/nlohmann/json_, named `labeltxt.cpp` & `labeltxt.h`, has been created to convert the annotations provided in a txt files in the format `classNumber x y width height` where the `classNumber` is nothing but the object class mapped to an integer, therefore for the current project specification it will be just <img src="https://render.githubusercontent.com/render/math?math=\color{white}0">, while `x y width height` are the coordinates of the box center, its width and its height, normalized from <img src="https://render.githubusercontent.com/render/math?math=\color{white}0"> to <img src="https://render.githubusercontent.com/render/math?math=\color{white}1"> with respect to the image size.
This process is mainly performed by two functions in the class, specifically: 
```c++
void setJsonPath(std::string path); // Specify the path of the JSON annotation file
```
which load the JSON file, specified by the provided path, and 
```c++
  void extractLabelsCoordinates(      // Extract and convert annotation label and box
    std::string emptyClass = "water", bool wtxt = true,
    bool classify = false, std::string obj = "boat");
```
that firstly checks if there are annotations in the image, or if the images has no label or labeled as without boats, then for each annotation found extract the bounding box coordinates converting them in the correct format and, by default, write the `.txt` file with all the information required.

## Data Preprocessing

As previously mentioned, different processes have been performed on the images that have been used to train the network.
Firstly, in all the case, the image set has been split in a train and validation set with a, default, ratio of <img src="https://render.githubusercontent.com/render/math?math=\color{white}3:1">.

### Naive Approach

A first attempt is done by training the darknet leaving the images as they are in the original dataset, which consists of a total of 1449 images of different boats, taken in an high variety scenarios from many different point of view. Thanks to this characteristics, the result was considered as a good baseline to evaluate the performance change based on the next approaches adopted.

### Data Augmentation Approach

The neural network used is already configured to do some data augmentation on the training dataset by changing randomly the image, according to the configuration specific given, for instance rotating it of a random angle or changing the hue of the image. In this situation however it has been decided to increase images changes resorting to illumination change, contrast change, equalization of the images histograms (in the RGB color space) and blurring the images with a fixed size Gaussian filter. In addition it has been also performed perspective transformation (https://www.researchgate.net/publication/338184137_Perspective_Transformation_Data_Augmentation_for_Object_Detection) in order to enrich the data augmentation.
The class `dataugmentation.cpp` & `dataugmentation.h` allows to easily perform the steps previously introduced. 
The functions, one for each transformation, are
```c++
  void equalize(cv::Mat& dst, int count = 0);
```
```c++
  void changeContrast(cv::Mat& dst, int count = 0);
```
```c++
  void changeBrightness(cv::Mat& dst, int count = 0);
```
```c++
  void gaussianSmooth(cv::Mat& dst, double sigma = 3, int count = 0);
```
```c++
  void changePerspective(cv::Mat& dst, float sigma = 0.5, int count = 0);
```
These functions return the edited images, starting from the original provided by the function
```c++
  void allignAnnotation(std::vector<std::vector<cv::Point2f>>& boxCorners);
```
For saving the annotations automatically in the new images it is also required to specify the bounding box coordinates and the classes in the image, corresponding to the object in the specified boxes. The variable `save` is used to determine whether to generate the text annotations file or not. 
For this approach a particular emphasis has been given to the perspective transformation. Its mechanism is divided into two part, firstly, new images with different viewpoints were created resorting to the perspective transformation, then, annotation alignment is used to generate corresponding annotation files.
The perspective transformation itself is

<img src="https://render.githubusercontent.com/render/math?math=\color{white}%5Cbegin%7Bpmatrix%7Dx_s%20%5C%5C%20y_s%20%5C%5C%20w_s%5Cend%7Bpmatrix%7D"><img src="https://render.githubusercontent.com/render/math?math=\color{white}%5Cbegin%7Bmatrix%7D%20%20%3D%20%20%5C%5C%20.%5C%5C%20%5Cend%7Bmatrix%7D"><img src="https://render.githubusercontent.com/render/math?math=\color{white}%5Cbegin%7Bbmatrix%7D%20p_%7B11%7D%20%26%20p_%7B12%7D%20%26%20p_%7B13%7D%20%20%5C%5C%20%20p_%7B21%7D%20%26%20p_%7B22%7D%20%26%20p_%7B23%7D%20%20%5C%5C%20%20p_%7B31%7D%20%26%20p_%7B32%7D%20%26%201%5Cend%7Bbmatrix%7D"><img src="https://render.githubusercontent.com/render/math?math=\color{white}%5Cbegin%7Bpmatrix%7Dx_t%5C%5Cy_t%5C%5C1%5Cend%7Bpmatrix%7D"><img src="https://render.githubusercontent.com/render/math?math=\color{white}%5Cbegin%7Bmatrix%7D%20%20%3D%20%20%5C%5C%20.%5C%5C%20%5Cend%7Bmatrix%7D"><img src="https://render.githubusercontent.com/render/math?math=\color{white}P_%7B%5Ctheta%7D%5Cbegin%7Bpmatrix%7Dx_t%20%5C%5C%20y_t%20%5C%5C%201%5Cend%7Bpmatrix%7D">

where <img src="https://render.githubusercontent.com/render/math?math=\color{white}%5Cleft(%5Cfrac%7Bx_s%7D%7Bw_s%7D%2C%20%5Cfrac%7By_s%7D%7Bw_s%7D%5Cright)%5ET"> is the source coordinates of the input pixel and <img src="https://render.githubusercontent.com/render/math?math=\color{white}%5Cleft(x_t%2C%20y_t%5Cright)%5ET"> is the coordinates of the pixel in the output image. The perspective transformation matrix, which can be considered as a planar homography, is found starting from the four vertices of the image 














































