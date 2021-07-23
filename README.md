# Computer Vision - Boat Detector

## Introduction

In this project, it was required to perform boat detection from single frames that can be useful in automated analysis of boat traffic in the sea.
Different techniques is used in Computer Vision in order to perform object detection of any kind (e.g., resorting to HOG descriptors combined with Machine Learning algorithm, such SVM, or Bag of Words image classification), however, for this project it has been chosen a Convolutional Neural Network (CNN) in order to deal with the the high variability with which a generic boat can present itself. In particualr the YOLO algorithm has been exploited to perform this task. 
YOLO (You Only Look Once) algorithm employs CNN to detect object in real-time; the CNN is used to predict various class probabilities and bounding box simultaneously. The YOLO object detector has been trained using the darknet open-source neural network framework with a dataset of boats images found from the kaggle platform at the link: https://www.kaggle.com/clorichel/boat-types-recognition/.
All the images have been manually annotated and from the annotation json files obtained it has been extracted the labels in txt format used by the darknet for training. Three approaches have been adopted to modify the images that have fed the darknet neural networks, then, the result it has been tested with the YOLO algorithm on a set of test images.

## Data Annotation

In the kaggle dataset all the images was divided in class depending on the specific boat (e.g., sail boat, ferry boat etc.) but, for the purpose of the project, every boats has been grouped in a single class. In each image a bounding box , defined by the coordinates $(x_{min}, y_{min}), (x_{max}, y_{max})$, has been used to highlight each boat present on the scene.
The images has been annotated using the website https://dataloop.ai/ which return the annotations on a dataset of JSON files, one for each image, containing all the labels information. To parse all the necessary information contained in the JSON files a dedicated class in C++, which makes use of the library json.hpp _https://github.com/nlohmann/json_, named ~~labeltxt.cpp~~ & ~~labeltxt.h~~, has been created to convert the annotations provided in a txt files in the format ~~classNumber x y width height~~ where the ~~classNumber~~ is nothing but the object class mapped to an integer, therefore for the current project specification it will be just $0$, while ~~x y width height~~ are the coordinates of the box center, its width and its height, normalized from $0$ to $1$ with respect to the image size.\newline
This process is mainly performed by two functions in the class, specifically 
\lstinputlisting[language=C++, firstline=15, lastline=15, basicstyle=\scriptsize, tabsize=1]{include/labeltxt.h}
which load the JSON file, specified by the provided path, and 
\lstinputlisting[language=C++, firstline=25, lastline=27, basicstyle=\scriptsize]{include/labeltxt.h} 
that firstly checks if there are annotations in the image, or if the images has no label or labeled as without boats, then for each annotation found extract the bounding box coordinates converting them in the correct format and, by default, write the {\fontfamily{qcr}\selectfont .txt} file with all the information required.
