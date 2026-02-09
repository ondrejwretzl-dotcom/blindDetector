package com.example.blinddetektor

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
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
import com.example.blinddetektor.util.BDLogger

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

  private lateinit var logger: BDLogger

  private val requestCamera = registerForActivityResult(
    ActivityResultContracts.RequestPermission()
  ) { granted ->
    logger.log("camera_permission_result granted=$granted")
    if (granted) startCamera()
    else {
      toast("Povol prosím přístup ke kameře.")
      speech.speak("Bez kamery to nepůjde. Povol prosím přístup ke kameře.")
    }
  }

  private val requestStorageLegacy = registerForActivityResult(
    ActivityResultContracts.RequestPermission()
  ) { granted ->
    logger.log("write_external_storage_result granted=$granted api=${Build.VERSION.SDK_INT}")
    // nic dalšího: logger init běží i bez toho, ale na <29 se bez permission nemusí zapsat
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

    logger = BDLogger(this)
    val initRes = logger.init()
    if (initRes.isSuccess) {
      logger.log("app_start logFile=${initRes.getOrNull()}")
      toast("Log: ${initRes.getOrNull()}")
    } else {
      toast("Nepodařilo se vytvořit log soubor.")
    }

    if (Build.VERSION.SDK_INT <= 28) {
      val perm = android.Manifest.permission.WRITE_EXTERNAL_STORAGE
      val granted = ContextCompat.checkSelfPermission(this, perm) == PackageManager.PERMISSION_GRANTED
      if (!granted) requestStorageLegacy.launch(perm)
    }

    initDetectorSafely()

    btnSpeakNow.setOnClickListener {
      val dets = overlay.getLastDetections()
      logger.log("manual_speak pressed dets=${dets.size}")
      val toSpeak = policy.pickForManualSpeech(dets)
      if (toSpeak.isEmpty()) {
        speech.speak(if (detector == null) "Detekce není dostupná. Zkontroluj prosím model v assets." else "Nic jistého teď nevidím.")
      } else {
        speech.speak(policy.formatForSpeech(toSpeak, withPositions = true))
      }
    }

    btnToggleAuto.setOnClickListener {
      autoEnabled = !autoEnabled
      btnToggleAuto.text = getString(if (autoEnabled) R.string.auto_on else R.string.auto_off)
      speech.speak(if (autoEnabled) "Průběžné hlášení zapnuto." else "Průběžné hlášení vypnuto.")
      policy.resetAutoState()
      logger.log("auto_toggle enabled=$autoEnabled")
    }

    ensureCameraPermission()
  }

  private fun initDetectorSafely() {
    val modelName = "yolov8n.onnx"
    val labelsName = "labels_cs.json"

    try {
      val rootAssets = assets.list("")?.toSet() ?: emptySet()
      logger.log("assets_root=${rootAssets.sorted().joinToString(",")}")

      if (!rootAssets.contains(modelName)) {
        detector = null
        toast("Chybí $modelName v assets. Detekce vypnuta.")
        logger.logE("missing_model_asset $modelName")
        return
      }
      if (!rootAssets.contains(labelsName)) {
        detector = null
        toast("Chybí $labelsName v assets. Detekce vypnuta.")
        logger.logE("missing_labels_asset $labelsName")
        return
      }

      val d = YoloV8OnnxDetector(
        context = this,
        modelAssetName = modelName,
        labelsAssetName = labelsName,
        logger = logger
      )

      d.onDetections = { detections, frameW, frameH, inferenceMs ->
        overlay.updateDetections(detections, frameW, frameH)

        logger.log("detections n=${detections.size} frame=${frameW}x${frameH} infMs=$inferenceMs top=${detections.take(3).joinToString { it.labelCs + ":" + String.format("%.2f", it.score) }}")

        if (autoEnabled) {
          val toSpeak = policy.pickForAutoSpeech(detections)
          if (toSpeak.isNotEmpty()) speech.speak(policy.formatForSpeech(toSpeak))
        }
      }

      detector = d
      logger.log("detector_init OK")
    } catch (t: Throwable) {
      detector = null
      toast("Detekce se nespustila: ${t.javaClass.simpleName}")
      logger.logE("detector_init FAILED", t)
    }
  }

  private fun ensureCameraPermission() {
    val granted = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
    logger.log("camera_permission granted=$granted")
    if (granted) startCamera() else requestCamera.launch(Manifest.permission.CAMERA)
  }

  private fun startCamera() {
    if (cameraController != null) return
    logger.log("camera_start")

    cameraController = CameraController(
      activity = this,
      previewView = previewView,
      logger = logger,
      onFrame = { image, rotationDegrees ->
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
    logger.log("app_destroy")
    cameraController?.stop()
    cameraController = null
    detector?.close()
    detector = null
    speech.close()
  }
}
