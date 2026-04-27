#include "FaceDetector.h"
#include "FaceRecognizer.h"
#include "MyPCA.h"
#include "WriteTrainData.h"
#include <jni.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

int currentMode = 0; // 0: Prepare, 1: Train, 2: Recognize
FaceDetector *faceDetector = nullptr;
MyPCA *myPCA = nullptr;
WriteTrainData *trainData = nullptr;

std::vector<Mat> trainingFaces;
std::vector<string> trainingLabels;
bool isModelTrained = false;

extern "C" JNIEXPORT void JNICALL
Java_com_example_faceidapp_MainActivity_initNative(JNIEnv *env, jobject thiz,
                                                   jstring faceCascade,
                                                   jstring eyesCascade) {
  const char *face = env->GetStringUTFChars(faceCascade, nullptr);
  const char *eyes = env->GetStringUTFChars(eyesCascade, nullptr);

  if (faceDetector != nullptr)
    delete faceDetector;
  faceDetector = new FaceDetector(string(face), string(eyes));

  env->ReleaseStringUTFChars(faceCascade, face);
  env->ReleaseStringUTFChars(eyesCascade, eyes);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_faceidapp_MainActivity_setMode(JNIEnv *env, jobject thiz,
                                                jint mode) {
  currentMode = mode;
  if (mode == 1) {
    // Reset training when Train is pressed again
    trainingFaces.clear();
    trainingLabels.clear();
    isModelTrained = false;
    if (myPCA != nullptr) {
      delete myPCA;
      myPCA = nullptr;
    }
    if (trainData != nullptr) {
      delete trainData;
      trainData = nullptr;
    }
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_faceidapp_MainActivity_processFrame(JNIEnv *env, jobject thiz,
                                                     jlong matAddr) {
  Mat &frame = *(Mat *)matAddr;

  // Flip horizontal in-place at the very beginning to mirror the camera but NOT
  // the text
  cv::flip(frame, frame, 1);

  if (faceDetector != nullptr) {
    Mat toTest;
    Mat processed;
    cvtColor(frame, processed, COLOR_RGBA2RGB);

    // This calculates if face exists and draws it on toTest
    faceDetector->findFacesInImage(processed, toTest);

    // Copy drawn rectangles back to frame
    resize(toTest, toTest, frame.size());
    cvtColor(toTest, frame, COLOR_RGB2RGBA);

    bool goodFace = faceDetector->goodFace();
    Mat faceMat = faceDetector->getFaceToTest();

    // Remove cv::flip here. Java already sent a flipped, user-facing frame. If
    // we flip before putText and then after, text becomes mirrored!

    if (currentMode == 0) {
      putText(frame, "Modo: Preparar", Point(50, 80), FONT_HERSHEY_SIMPLEX, 1.5,
              Scalar(0, 255, 0, 255), 3);
    } else if (currentMode == 1) {
      if (goodFace && !faceMat.empty() && trainingFaces.size() < 30) {
        trainingFaces.push_back(faceMat.clone());
        trainingLabels.push_back("Propietario");

        // Add augmented data (rotation +- 10 degrees)
        Mat rotMat = getRotationMatrix2D(
            Point2f(faceMat.cols / 2, faceMat.rows / 2), 10, 1.0);
        Mat rotatedFace;
        warpAffine(faceMat, rotatedFace, rotMat, faceMat.size());
        trainingFaces.push_back(rotatedFace);
        trainingLabels.push_back("Propietario");

        rotMat = getRotationMatrix2D(
            Point2f(faceMat.cols / 2, faceMat.rows / 2), -10, 1.0);
        warpAffine(faceMat, rotatedFace, rotMat, faceMat.size());
        trainingFaces.push_back(rotatedFace);
        trainingLabels.push_back("Propietario");
      }

      putText(frame,
              "Entrenando: " + to_string(trainingFaces.size() / 3) + "/10 imgs",
              Point(50, 80), FONT_HERSHEY_SIMPLEX, 1.5,
              Scalar(0, 255, 255, 255), 3);

      if (trainingFaces.size() >= 30 && !isModelTrained) {
        myPCA = new MyPCA(trainingFaces);
        trainData = new WriteTrainData(*myPCA, trainingLabels);
        isModelTrained = true;
      }

      if (isModelTrained) {
        putText(frame, "Entrenamiento Completado!", Point(50, 150),
                FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 255, 0, 255), 3);
      }
    } else if (currentMode == 2) {
      putText(frame, "Modo: Reconocer", Point(50, 80), FONT_HERSHEY_SIMPLEX,
              1.5, Scalar(0, 0, 255, 255), 3);
      if (isModelTrained && goodFace && !faceMat.empty()) {
        FaceRecognizer fr(faceMat, myPCA->getAverage(),
                          myPCA->getEigenvectors(),
                          trainData->getFacesInEigen(), trainingLabels);
        string id = fr.getClosetFaceID();
        double dist = fr.getClosetDist();

        // If distance is reasonable
        if (dist < 2000) { // Un umbral más estricto
          putText(frame, id + " D:" + to_string((int)dist), Point(50, 150),
                  FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 255, 0, 255), 3);
        } else {
          putText(frame, "Desc. D:" + to_string((int)dist), Point(50, 150),
                  FONT_HERSHEY_SIMPLEX, 1.5, Scalar(255, 0, 0, 255), 3);
        }
      } else if (!isModelTrained) {
        putText(frame, "Debes Entrenar Primero", Point(50, 150),
                FONT_HERSHEY_SIMPLEX, 1.5, Scalar(255, 0, 0, 255), 3);
      }
    }

    // Remove cv::flip here.
  }
}