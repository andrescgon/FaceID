/*PCA Process for tranning data*/
#ifndef MY_PCA_H
#define MY_PCA_H

#include <iostream>
#include <vector>
#include <float.h>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

class MyPCA {

public:
	MyPCA(vector<Mat>& _facesMat);
	void init(vector<Mat>& _facesMat);
    void getImgSize(vector<Mat>& _facesMat);
    void mergeMatrix(vector<Mat>& _facesMat);
    void getAverageVector();
    void subtractMatrix();
    void getBestEigenVectors(Mat _covarMatrix);
    Mat getFacesMatrix();
	Mat getAverage();
	Mat getEigenvectors();
    ~MyPCA();

private:
    int imgSize = -1;//Dimension of features
    int imgRows = -1;//row# of image
    Mat allFacesMatrix;
    Mat avgVector;
    Mat subFacesMatrix;
    Mat eigenVector;
};

#endif
