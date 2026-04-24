package com.example.faceidapp

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.OpenCVLoader
import org.opencv.core.Mat

class MainActivity : AppCompatActivity(), CameraBridgeViewBase.CvCameraViewListener2 {

    private lateinit var cameraView: CameraBridgeViewBase

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Request Camera Permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 1)
        }

        cameraView = findViewById(R.id.camera_view)
        cameraView.setCameraPermissionGranted()
        cameraView.setCvCameraViewListener(this)

        if (OpenCVLoader.initLocal()) {
            Log.d("OpenCV", "OpenCV loaded successfully")
            cameraView.enableView()
        } else {
            Log.e("OpenCV", "OpenCV failed to load")
            Toast.makeText(this, "Error cargando OpenCV", Toast.LENGTH_LONG).show()
        }

        findViewById<Button>(R.id.btn_prepare).setOnClickListener {
            setMode(0)
            Toast.makeText(this, "Modo: Preparar", Toast.LENGTH_SHORT).show()
        }
        findViewById<Button>(R.id.btn_train).setOnClickListener {
            setMode(1)
            Toast.makeText(this, "Modo: Entrenar", Toast.LENGTH_SHORT).show()
        }
        findViewById<Button>(R.id.btn_recognize).setOnClickListener {
            setMode(2)
            Toast.makeText(this, "Modo: Reconocer", Toast.LENGTH_SHORT).show()
        }

        // Initialize Native logic
        initNative("dummy_path")
    }

    override fun onCameraViewStarted(width: Int, height: Int) {}
    override fun onCameraViewStopped() {}

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame): Mat {
        val mat = inputFrame.rgba()
        // Pass frame to C++
        processFrame(mat.nativeObjAddr)
        return mat
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraView.disableView()
    }

    /**
     * A native method that is implemented by the 'faceidapp' native library,
     * which is packaged with this application.
     */
    external fun processFrame(matAddr: Long)
    external fun setMode(mode: Int)
    external fun initNative(cascadePath: String)

    companion object {
        init {
            System.loadLibrary("faceidapp")
        }
    }
}