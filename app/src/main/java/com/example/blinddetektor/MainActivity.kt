package com.example.blinddetektor

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Button
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.example.blinddetektor.camera.CameraController
import com.example.blinddetektor.ml.YoloV8OnnxDetector
import com.example.blinddetektor.speech.RelevancePolicy
import com.example.blinddetektor.speech.SpeechManager
import com.example.blinddetektor.ui.OverlayView

class MainActivity : ComponentActivity() {

  private lateinit var previewView: PreviewView
  private lateinit var overlay: OverlayView
  private lateinit var btnSpeakNow: Button
  private lateinit var btnToggleAuto: Button

  private lateinit var detector: YoloV8OnnxDetector
  private lateinit var speech: SpeechManager
  private lateinit var policy: RelevancePolicy
  private lateinit var cameraController: CameraController

  private var autoEnabled = false

  private val requestCamera = registerForActivityResult(
    ActivityResultContracts.RequestPermission()
  ) { granted ->
    if (granted) startCamera()
    else {
      speech.speak("Bez kamery to nepůjde. Povol prosím přístup ke kameře.")
    }
  }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_main)

    previewView = findViewById(R.id.previewView)
    overlay = findViewById(R.id.overlay)
    btnSpeakNow = findViewById(R.id.btnSpeakNow)
    btnToggleAuto = findViewById(R.id.btnToggleAuto)

    speech = SpeechManager(this)
    policy = RelevancePolicy()

    detector = YoloV8OnnxDetector(
      context = this,
      modelAssetName = "yolov8n.onnx",
      labelsAssetName = "labels_cs.json"
    )

    cameraController = CameraController(
      activity = this,
      previewView = previewView,
      onFrame = { image, rotationDegrees ->
        detector.detect(image, rotationDegrees)
      }
    )

    detector.onDetections = { detections, frameW, frameH ->
      overlay.updateDetections(detections, frameW, frameH)
      if (autoEnabled) {
        val toSpeak = policy.pickForAutoSpeech(detections)
        if (toSpeak.isNotEmpty()) speech.speak(policy.formatForSpeech(toSpeak))
      }
    }

    btnSpeakNow.setOnClickListener {
      val toSpeak = policy.pickForManualSpeech(overlay.getLastDetections())
      if (toSpeak.isEmpty()) speech.speak("Nic jistého teď nevidím.")
      else speech.speak(policy.formatForSpeech(toSpeak, withPositions = true))
    }

    btnToggleAuto.setOnClickListener {
      autoEnabled = !autoEnabled
      btnToggleAuto.text = getString(if (autoEnabled) R.string.auto_on else R.string.auto_off)
      speech.speak(if (autoEnabled) "Průběžné hlášení zapnuto." else "Průběžné hlášení vypnuto.")
      policy.resetAutoState()
    }

    ensureCameraPermission()
  }

  private fun ensureCameraPermission() {
    val granted = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
      PackageManager.PERMISSION_GRANTED
    if (granted) startCamera() else requestCamera.launch(Manifest.permission.CAMERA)
  }

  private fun startCamera() {
    cameraController.start()
  }

  override fun onDestroy() {
    super.onDestroy()
    cameraController.stop()
    detector.close()
    speech.close()
  }
}
