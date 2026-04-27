#include "WriteTrainData.h"

WriteTrainData::WriteTrainData(MyPCA _trainPCA, vector<string>& _trainFacesID)
{
    numberOfFaces = _trainPCA.getFacesMatrix().cols;
    if(numberOfFaces > 0) {
        trainFacesInEigen.create(numberOfFaces, numberOfFaces, CV_32FC1);
        project(_trainPCA);
    }
}

void WriteTrainData::project(MyPCA _trainPCA)
{
    //cout << "Write Class"<<_trainPCA.getFacesMatrix().size() << endl;
    Mat facesMatrix = _trainPCA.getFacesMatrix();
    Mat avg = _trainPCA.getAverage();
    Mat eigenVec = _trainPCA.getEigenvectors();
    
    for (int i = 0; i < numberOfFaces; i++) {
        Mat temp;
        Mat projectFace = trainFacesInEigen.col(i);
        subtract(facesMatrix.col(i), avg, temp);
        projectFace = eigenVec * temp;
    }
    //cout << trainFacesInEigen.col(0).size() <<endl;
}

void WriteTrainData::writeTrainFacesData(vector<string>& _trainFacesID) {}
void WriteTrainData::writeMean(Mat avg) {}
void WriteTrainData::writeEigen(Mat eigen) {}

Mat WriteTrainData::getFacesInEigen()
{
    return trainFacesInEigen;
}

WriteTrainData::~WriteTrainData() {}
