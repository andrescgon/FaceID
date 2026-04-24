#include <jni.h>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "FaceDetector.h"

using namespace cv;

int currentMode = 0; // 0: Prepare, 1: Train, 2: Recognize
FaceDetector* faceDetector = nullptr;

extern "C" JNIEXPORT void JNICALL
Java_com_example_faceidapp_MainActivity_initNative(JNIEnv *env, jobject thiz, jstring cascadePath) {
    // Convert jstring to std::string
    const char *path = env->GetStringUTFChars(cascadePath, nullptr);
    // Initialize detector here if needed with the path
    // faceDetector = new FaceDetector(path); // Modificar FaceDetector para aceptar ruta
    env->ReleaseStringUTFChars(cascadePath, path);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_faceidapp_MainActivity_setMode(JNIEnv *env, jobject thiz, jint mode) {
    currentMode = mode;
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_faceidapp_MainActivity_processFrame(JNIEnv *env, jobject thiz, jlong matAddr) {
    Mat& frame = *(Mat*)matAddr;
    
    // Basic OpenCV processing to show it works
    Mat processed;
    cvtColor(frame, processed, COLOR_RGBA2RGB);
    
    // Here we would call:
    // faceDetector->findFacesInImage(processed, processed);
    // And draw text/rectangles depending on currentMode.
    
    // For now, let's just draw a circle to prove OpenCV runs
    circle(frame, Point(frame.cols/2, frame.rows/2), 100, Scalar(0, 255, 0, 255), 5);
    putText(frame, "C++ OpenCV Activo", Point(50, 100), FONT_HERSHEY_SIMPLEX, 2.0, Scalar(255, 0, 0, 255), 3);
}