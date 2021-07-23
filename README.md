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

<img src="https://render.githubusercontent.com/render/math?math=\color{white}A%3D%5Cleft(0%2C0%5Cright)%2C%20B%3D%5Cleft(0%2CH%5Cright)%2C%20C%3D%5Cleft(W%2CH%5Cright)%2C%20D%3D%5Cleft(W%2C0%5Cright)%2C">

where <img src="https://render.githubusercontent.com/render/math?math=\color{white}W"> is the width of the image and <img src="https://render.githubusercontent.com/render/math?math=\color{white}H"> is the height, and the four random points 

<img src="https://render.githubusercontent.com/render/math?math=\color{white}A_%7B%5Ctheta%7D%3D%5Cleft(x_%7Btl%7D%2Cy_%7Btl%7D%5Cright)%2C%20B_%7B%5Ctheta%7D%3D%5Cleft(x_%7Bbl%7D%2CH-y_%7Bbl%7D%5Cright)%2C%20C_%7B%5Ctheta%7D%3D%5Cleft(W-x_%7Bbr%7D%2CH-y_%7Bbr%7D%5Cright)%2C%20D_%7B%5Ctheta%7D%3D%5Cleft(W-x_%7Btr%7D%2Cy_%7Btr%7D%5Cright)%2C">

with

<img src="https://render.githubusercontent.com/render/math?math=\color{white}x_%7Btl%7D%2Cx_%7Bbl%7D%2Cx_%7Bbr%7D%2Cx_%7Btr%7D%20%5Cin%20%5Cleft(0%2C%20W(%5Clambda%5Ctheta)%5Cright)">
<img src="https://render.githubusercontent.com/render/math?math=\color{white}y_%7Btl%7D%2Cy_%7Bbl%7D%2Cy_%7Bbr%7D%2Cy_%7Btr%7D%20%5Cin%20%5Cleft(0%2C%20H(%5Clambda%5Ctheta)%5Cright)">

The choice of the four points taken in this way ensure that any three points in <img src="https://render.githubusercontent.com/render/math?math=\color{white}A_%7B%5Ctheta%7D%2C%20B_%7B%5Ctheta%7D%2C%20C_%7B%5Ctheta%7D"> and <img src="https://render.githubusercontent.com/render/math?math=\color{white}D_%7B%5Ctheta%7D"> are non-collinear. The hyperparameter <img src="https://render.githubusercontent.com/render/math?math=\color{white}%5Ctheta"> is the perspective parameter; the grater of value <img src="https://render.githubusercontent.com/render/math?math=\color{white}%5Ctheta">, the more obvious the perspective transformation.
In the perspective transformation function the 8 points just introduced are chosen randomly in the interval previously defined.
In the new image also the corresponding bounding box is modified becoming, usually, trapezoidal. Unfortunately CNN can not deal with boxes that are not rectangular, therefore, an auxiliary function is created in order to align the corresponding bounding box.
In particular, assuming <img src="https://render.githubusercontent.com/render/math?math=\color{white}a%5E*%3D(x_1%2Cy_1)%2C%20b%5E*%3D(x_2%2Cy_2)%2C%20c%5E*%3D(x_3%2Cy_3)%2C%20d%5E*%3D(x_4%2Cy_4)"> as the four vertex of the transformed bounding box, the new coordinates are chosen as follows:

<img src="https://render.githubusercontent.com/render/math?math=\color{white}x%5E*_%7Bmin%7D%3Dmin%5Cleft(x_1%2C%20x_2%2C%20x_3%2C%20x_4%5Cright)%2C">
<img src="https://render.githubusercontent.com/render/math?math=\color{white}y%5E*_%7Bmin%7D%3Dmin%5Cleft(y_1%2C%20y_2%2C%20y_3%2C%20y_4%5Cright)%2C">
<img src="https://render.githubusercontent.com/render/math?math=\color{white}x%5E*_%7Bmax%7D%3Dmax%5Cleft(x_1%2C%20x_2%2C%20x_3%2C%20x_4%5Cright)%2C">
<img src="https://render.githubusercontent.com/render/math?math=\color{white}y%5E*_%7Bmax%7D%3Dmax%5Cleft(y_1%2C%20y_2%2C%20y_3%2C%20y_4%5Cright).">

This method allows to automatically generate trainable annotated images, without additional manual labeling.
Finally, if the variable `save` is set to `true`, for all the transformation the private function 
```c++
  void saveAndWritetxt(
    cv::Mat img, std::string imgPath, std::vector<int> classNumber,
    std::vector<std::vector<cv::Point2f>> boxCorners,
    std::string augType, int count);
```
will generate the txt file with the corresponding bounding boxes annotations.

### Gradient Approach

Gradient often plays a key role in Computer Vision for object detection or pattern recognition based on the object appearance. A perfect example is the Histogram of Oriented Gradient (HOG) descriptor, used for image classification. The boats, despite the high variety with which they show, have some characteristic shape that might be exploited resorting to the gradient of the image. In particular it has been chosen to simply use the magnitude of the gradient in both the <img src="https://render.githubusercontent.com/render/math?math=\color{white}x"> and <img src="https://render.githubusercontent.com/render/math?math=\color{white}y"> directions. It has been decided to not filter the image (as for instance is done for the Canny edges) since also noise can be learned, and, moreover, the thickness of the edges might carry important information. This transformation can also be combined with data augmentation. However, the result did not bring any improvement, and therefore this approach has been discarded.

## Darknet Training

The Neural Network used is the [Alexey's darknet](https://github.com/AlexeyAB/darknet/) which is constantly maintained and contains many improvement with respect the official [Darknet repo](https://github.com/pjreddie/darknet), where the last commit was in September 2018.
The network was trained for YOLOv4, using an RTX3090, modifying the configuration file according to the instruction provided by the creator and the task requirements. For all the three approaches the same resolution and the same number of iteration (always greater than the minimum recommended quantity) have been used. Also, the pre-trained files provided in the instruction have been used as starting [weights](https://drive.google.com/open?id=1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp).

## Result
All the three trainings went well, as shown in Figure 1, but unfortunately, due to a CUDA 11.3 bug it was not possible to see also the mean Average Precision. For this reason, the source code has been modified to keep track of the progress every 1000 iterations and then manually evaluate the metrics on the validation set.
| ![](chart/chart_naive.png) | ![](chart/chart_augmented.png) | ![](chart/chart_gradient.png) |
|:--:| 
| *Training loss: a) Naive approach; b) Data Augmentation approach; c) Gradient approach* |





<img src="https://render.githubusercontent.com/render/math?math=\color{white}">








































