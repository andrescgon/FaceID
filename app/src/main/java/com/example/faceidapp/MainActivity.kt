package com.example.faceidapp

import android.Manifest
import android.content.Context
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
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream

class MainActivity : AppCompatActivity(), CameraBridgeViewBase.CvCameraViewListener2 {

    private lateinit var cameraView: CameraBridgeViewBase

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        cameraView = findViewById(R.id.camera_view)
        cameraView.setCvCameraViewListener(this)
        cameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT)

        // Request Camera Permission safely
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 1)
        } else {
            activateOpenCVCamera()
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

        // Load cascades and pass to native
        val facePath = loadCascadeFile(R.raw.haarcascade_frontalface_default, "haarcascade_frontalface_default.xml")
        val eyePath = loadCascadeFile(R.raw.haarcascade_eye, "haarcascade_eye.xml")
        initNative(facePath, eyePath)
    }

    private fun loadCascadeFile(resourceId: Int, cascadeName: String): String {
        try {
            val isStream: InputStream = resources.openRawResource(resourceId)
            val cascadeDir: File = getDir("cascade", Context.MODE_PRIVATE)
            val cascadeFile = File(cascadeDir, cascadeName)
            
            if (!cascadeFile.exists()) {
                val os = FileOutputStream(cascadeFile)
                val buffer = ByteArray(4096)
                var bytesRead: Int
                while (isStream.read(buffer).also { bytesRead = it } != -1) {
                    os.write(buffer, 0, bytesRead)
                }
                isStream.close()
                os.close()
            }
            return cascadeFile.absolutePath
        } catch (e: Exception) {
            e.printStackTrace()
            return ""
        }
    }

    private fun activateOpenCVCamera() {
        cameraView.setCameraPermissionGranted()
        if (OpenCVLoader.initLocal()) {
            Log.d("OpenCV", "OpenCV loaded successfully")
            cameraView.enableView()
        } else {
            Log.e("OpenCV", "OpenCV failed to load")
            Toast.makeText(this, "Error cargando OpenCV", Toast.LENGTH_LONG).show()
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 1 && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            activateOpenCVCamera()
        } else {
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCameraViewStarted(width: Int, height: Int) {}
    override fun onCameraViewStopped() {}

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame): Mat {
        val mat = inputFrame.rgba()
        processFrame(mat.nativeObjAddr)
        return mat
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraView.disableView()
    }

    external fun processFrame(matAddr: Long)
    external fun setMode(mode: Int)
    external fun initNative(faceCascade: String, eyesCascade: String)

    companion object {
        init {
            System.loadLibrary("faceidapp")
        }
    }
}