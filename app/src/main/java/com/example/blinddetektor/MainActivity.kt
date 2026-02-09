package com.example.blinddetektor

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.speech.RecognizerIntent
import android.widget.Button
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.widget.SwitchCompat
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.example.blinddetektor.camera.CameraController
import com.example.blinddetektor.ml.Detection
import com.example.blinddetektor.ml.YoloV8OnnxDetector
import com.example.blinddetektor.speech.GuidanceController
import com.example.blinddetektor.speech.RelevancePolicy
import com.example.blinddetektor.speech.SpeechManager
import com.example.blinddetektor.ui.OverlayView
import com.example.blinddetektor.util.BDLogger
import com.example.blinddetektor.util.normalizeCs

class MainActivity : ComponentActivity() {

  private lateinit var previewView: PreviewView
  private lateinit var overlay: OverlayView
  private lateinit var btnSpeakNow: Button
  private lateinit var btnToggleAuto: Button
  private lateinit var btnFindObject: Button
  private lateinit var switchUnknown: SwitchCompat

  private var detector: YoloV8OnnxDetector? = null
  private var cameraController: CameraController? = null

  private lateinit var speech: SpeechManager
  private lateinit var policy: RelevancePolicy
  private lateinit var guidance: GuidanceController
  private lateinit var logger: BDLogger

  private var autoEnabled = false
  private var hideUnknown = false

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

  private val requestAudio = registerForActivityResult(
    ActivityResultContracts.RequestPermission()
  ) { granted ->
    logger.log("audio_permission_result granted=$granted")
    if (!granted) speech.speak("Bez mikrofonu nepůjde hlasové hledání.")
  }

  private val requestStorageLegacy = registerForActivityResult(
    ActivityResultContracts.RequestPermission()
  ) { granted ->
    logger.log("write_external_storage_result granted=$granted api=${Build.VERSION.SDK_INT}")
  }

  private val voiceFindLauncher = registerForActivityResult(
    ActivityResultContracts.StartActivityForResult()
  ) { res ->
    val results = res.data?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)
    val spoken = results?.firstOrNull()?.trim().orEmpty()
    logger.log("voice_find_result raw='$spoken'")
    if (spoken.isBlank()) {
      speech.speak("Nerozuměl jsem. Zkus to znovu.")
      return@registerForActivityResult
    }

    val target = resolveTargetLabel(spoken)
    if (target == null) {
      speech.speak("Neznám objekt $spoken. Zkus třeba: hrnek, židle, osoba, mobil.")
    } else {
      guidance.start(target)
    }
  }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_main)

    previewView = findViewById(R.id.previewView)
    overlay = findViewById(R.id.overlay)
    btnSpeakNow = findViewById(R.id.btnSpeakNow)
    btnToggleAuto = findViewById(R.id.btnToggleAuto)
    btnFindObject = findViewById(R.id.btnFindObject)
    switchUnknown = findViewById(R.id.switchUnknown)

    speech = SpeechManager(this)
    policy = RelevancePolicy()
    guidance = GuidanceController(speech)

    logger = BDLogger(this)
    val initRes = logger.init()
    if (initRes.isSuccess) {
      logger.log("app_start logFile=${initRes.getOrNull()}")
      toast("Log: ${initRes.getOrNull()}")
    } else {
      toast("Nepodařilo se vytvořit log soubor.")
    }

    if (Build.VERSION.SDK_INT <= 28) {
      val perm = Manifest.permission.WRITE_EXTERNAL_STORAGE
      val granted = ContextCompat.checkSelfPermission(this, perm) == PackageManager.PERMISSION_GRANTED
      if (!granted) requestStorageLegacy.launch(perm)
    }

    // audio permission pro voice-find
    run {
      val granted = ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
      if (!granted) requestAudio.launch(Manifest.permission.RECORD_AUDIO)
    }

    switchUnknown.setOnCheckedChangeListener { _, isChecked ->
      hideUnknown = isChecked
      switchUnknown.text = getString(if (hideUnknown) R.string.hide_unknown_on else R.string.hide_unknown_off)
      logger.log("hide_unknown=$hideUnknown")
    }
    switchUnknown.text = getString(R.string.hide_unknown_off)

    initDetectorSafely()

    btnFindObject.setOnClickListener { startVoiceFind() }

    btnSpeakNow.setOnClickListener {
      val dets = filterUnknownIfNeeded(overlay.getLastDetections())
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

  private fun startVoiceFind() {
    // druhým klikem ukonči navigaci
    if (guidance.isActive()) {
      guidance.stop()
      return
    }

    val granted = ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
    if (!granted) {
      requestAudio.launch(Manifest.permission.RECORD_AUDIO)
      return
    }

    val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
      putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
      putExtra(RecognizerIntent.EXTRA_LANGUAGE, "cs-CZ")
      putExtra(RecognizerIntent.EXTRA_PROMPT, "Řekni název objektu, který chceš najít.")
      putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 3)
    }
    voiceFindLauncher.launch(intent)
    speech.speak("Řekni, co mám najít.")
  }

  private fun resolveTargetLabel(spoken: String): String? {
    val s = normalizeCs(spoken)
    val synonyms = mapOf(
      "stul" to "jídelní stůl",
      "jidelni stul" to "jídelní stůl",
      "telefon" to "mobil",
      "mobilni telefon" to "mobil",
      "laptop" to "notebook",
      "clovek" to "osoba",
      "zidle" to "židle",
      "gauc" to "gauč",
      "lahev" to "láhev"
    )
    return synonyms[s] ?: spoken.trim().takeIf { it.isNotBlank() }
  }

  private fun filterUnknownIfNeeded(dets: List<Detection>): List<Detection> =
    if (!hideUnknown) dets else dets.filter { it.hasLabel }

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

      val d = YoloV8OnnxDetector(this, modelName, labelsName)

      d.onDetections = { detections, frameW, frameH ->
        val filtered = filterUnknownIfNeeded(detections)
        overlay.updateDetections(filtered, frameW, frameH)

        // navigace má prioritu (nepřekřikovat se s auto čtením)
        guidance.onDetections(filtered)

        if (autoEnabled && !guidance.isActive()) {
          val toSpeak = policy.pickForAutoSpeech(filtered)
          if (toSpeak.isNotEmpty()) speech.speak(policy.formatForSpeech(toSpeak, withPositions = true))
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
