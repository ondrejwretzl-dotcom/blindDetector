package com.example.blinddetektor

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Button
import android.widget.Toast
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

  private var detector: YoloV8OnnxDetector? = null
  private lateinit var speech: SpeechManager
  private lateinit var policy: RelevancePolicy
  private var cameraController: CameraController? = null

  private var autoEnabled = false

  private val requestCamera = registerForActivityResult(
    ActivityResultContracts.RequestPermission()
  ) { granted ->
    if (granted) startCamera()
    else {
      toast("Povol prosím přístup ke kameře.")
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

    // ✅ FIX: bezpečná inicializace detektoru (app nespadne, i když chybí model nebo selže ORT)
    initDetectorSafely()

    btnSpeakNow.setOnClickListener {
      val dets = overlay.getLastDetections()
      val toSpeak = policy.pickForManualSpeech(dets)
      if (toSpeak.isEmpty()) {
        speech.speak(
          if (detector == null) "Detekce není dostupná. Zkontroluj prosím model v assets."
          else "Nic jistého teď nevidím."
        )
      } else {
        speech.speak(policy.formatForSpeech(toSpeak, withPositions = true))
      }
    }

    btnToggleAuto.setOnClickListener {
      autoEnabled = !autoEnabled
      btnToggleAuto.text = getString(if (autoEnabled) R.string.auto_on else R.string.auto_off)
      speech.speak(if (autoEnabled) "Průběžné hlášení zapnuto." else "Průběžné hlášení vypnuto.")
      policy.resetAutoState()
    }

    ensureCameraPermission()
  }

  private fun initDetectorSafely() {
    val modelName = "yolov8n.onnx"
    val labelsName = "labels_cs.json"
    try {
      val rootAssets = assets.list("")?.toSet() ?: emptySet()
      if (!rootAssets.contains(modelName)) {
        detector = null
        toast("Chybí $modelName v assets. Detekce vypnuta.")
        return
      }
      if (!rootAssets.contains(labelsName)) {
        detector = null
        toast("Chybí $labelsName v assets. Detekce vypnuta.")
        return
      }

      val d = YoloV8OnnxDetector(
        context = this,
        modelAssetName = modelName,
        labelsAssetName = labelsName
      )

      d.onDetections = { detections, frameW, frameH ->
        overlay.updateDetections(detections, frameW, frameH)
        if (autoEnabled) {
          val toSpeak = policy.pickForAutoSpeech(detections)
          if (toSpeak.isNotEmpty()) speech.speak(policy.formatForSpeech(toSpeak))
        }
      }

      detector = d
    } catch (t: Throwable) {
      detector = null
      toast("Detekce se nespustila: ${t.javaClass.simpleName}")
      // Volitelně: speech.speak("Detekce se nespustila. Zkontroluj model a kompatibilitu zařízení.")
    }
  }

  private fun ensureCameraPermission() {
    val granted = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
      PackageManager.PERMISSION_GRANTED
    if (granted) startCamera() else requestCamera.launch(Manifest.permission.CAMERA)
  }

  private fun startCamera() {
    if (cameraController != null) return

    cameraController = CameraController(
      activity = this,
      previewView = previewView,
      onFrame = { image, rotationDegrees ->
        // Kamera běží i když detektor není dostupný
        detector?.detect(image, rotationDegrees)
      }
    )
    cameraController?.start()
  }

  private fun toast(msg: String) {
    Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
  }

  override fun onDestroy() {
    super.onDestroy()
    cameraController?.stop()
    cameraController = null
    detector?.close()
    detector = null
    speech.close()
  }
}

