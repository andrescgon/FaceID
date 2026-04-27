#include <vector>

#include "FaceDetector.h"

FaceDetector::FaceDetector(string face_cascadePath, string eyes_cascadePath)
{
    if ( !face_cascade.load(face_cascadePath) )
        cout << "ERROR:***Can not load face cascade***" << endl;
    
    if ( !eye_cascade.load(eyes_cascadePath) )
        cout << "ERROR:***Can not load eye cascade***" << endl;
}
void FaceDetector::findFacesInImage(Mat &frameRGB, Mat &toTest) {
    Mat frameGray;
    faceFlag = 0; // Reset flag
    
    // Copy input safely
    toTest = frameRGB.clone();
    
    // We resize image down for faster processing and to have consistent proportions
    // Android cameras send huge images, e.g., 1920x1080.
    double scale = 480.0 / std::max(toTest.cols, toTest.rows);
    if(scale < 1.0) resize(toTest, toTest, Size(), scale, scale);
    
    cvtColor(toTest, frameGray, COLOR_BGR2GRAY);
    Mat toReturn = toTest.clone();
    equalizeHist(frameGray, frameGray);
    
    vector<Rect> facesRec;
    
    //detect faces:
    face_cascade.detectMultiScale( frameGray, facesRec, 1.1, 5, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
    //cout << "faces: " << facesRec.size() << endl;
    
    if (facesRec.size() >= 1){
        rectangle(toTest, facesRec[0], Scalar( 255, 0, 255 ), 4);
        
        Mat faceROI = toReturn(facesRec[0]);
        //cout << "ROI SIZE " << faceROI.size() << endl;
        faceROI.copyTo(faceToTest);
        resize(faceToTest, faceToTest, Size(100,100));
        cvtColor(faceToTest, faceToTest, COLOR_BGR2GRAY); // Crucial for PCA
        faceFlag = 1;
        
        vector<Rect> eyes;
        //detect eyes
        eye_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );
        //cout << "eyes: " << eyes.size() << endl;
        
        for ( size_t j = 0; j < eyes.size(); j++ )
        {
            Point eye_center( facesRec[0].x + eyes[j].x + eyes[j].width/2, facesRec[0].y + eyes[j].y + eyes[j].height/2 );
            circle(toTest, eye_center, 2, Scalar( 255, 0, 0 ), 4, 8, 0);
        }
        
        eyes.clear();
    }
    
    facesRec.clear();
}

bool FaceDetector::goodFace()
{
    return faceFlag;
}

Mat FaceDetector::getFaceToTest()
{
    return faceToTest;
}

FaceDetector::~FaceDetector() {}
