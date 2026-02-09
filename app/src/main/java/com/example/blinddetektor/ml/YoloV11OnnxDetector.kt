package com.example.blinddetektor.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import ai.onnxruntime.*
import androidx.camera.core.ImageProxy
import com.example.blinddetektor.util.BDLogger
import kotlinx.coroutines.*
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.nio.FloatBuffer
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.*

class YoloV11OnnxDetector(
  private val context: Context,
  modelAssetName: String,
  labelsAssetName: String,
  private val logger: BDLogger
) : AutoCloseable {

  private val inputSize = 640
  private val confThreshold = 0.25f
  private val iouThreshold = 0.45f

  private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
  private val session: OrtSession
  private val labelsCs: Map<Int, String>

  private val minInferenceIntervalMs = 250L
  private val lastInferenceAt = AtomicLong(0L)
  private val busy = AtomicBoolean(false)

  var onDetections: ((List<Detection>, Int, Int, Long) -> Unit)? = null
  private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

  // hrubý odhad vertikálního FOV (doladíš podle telefonu)
  private val assumedVerticalFovDeg = 60.0

  // typické výšky objektů v metrech (pro odhad vzdálenosti)
  private val classHeightsM = mapOf(
    "osoba" to 1.70f,
    "židle" to 0.90f,
    "gauč" to 0.90f,
    "postel" to 0.55f,
    "jídelní stůl" to 0.75f,
    "televize" to 0.60f,
    "notebook" to 0.25f,
    "mobil" to 0.15f,
    "láhev" to 0.28f,
    // podle tebe: hrnek ~8cm
    "hrnek" to 0.08f,
    "kniha" to 0.24f
  )

  init {
    val modelBytes = context.assets.open(modelAssetName).readBytes()

    val opts = OrtSession.SessionOptions().apply {
      setIntraOpNumThreads(2)
      setInterOpNumThreads(1)
      try { addNnapi(); logger.log("ort_session NNAPI enabled") }
      catch (t: Throwable) { logger.log("ort_session NNAPI not available: ${t.javaClass.simpleName}: ${t.message}") }
    }

    session = env.createSession(modelBytes, opts)

    logger.log("ort_session created inputs=${session.inputNames.joinToString()} outputs=${session.outputNames.joinToString()}")
    session.inputInfo.forEach { (name, info) -> logger.log("ort_input name=$name info=$info") }
    session.outputInfo.forEach { (name, info) -> logger.log("ort_output name=$name info=$info") }

    labelsCs = loadLabels(labelsAssetName)
    logger.log("labels_loaded n=${labelsCs.size}")
  }

  fun detect(image: ImageProxy, rotationDegrees: Int) {
    val now = System.currentTimeMillis()
    val last = lastInferenceAt.get()
    if (now - last < minInferenceIntervalMs) return
    if (!lastInferenceAt.compareAndSet(last, now)) return
    if (!busy.compareAndSet(false, true)) return

    val t0 = SystemClock.elapsedRealtime()
    val bitmap = image.toBitmap()
    val rotated = bitmap.rotate(rotationDegrees.toFloat())
    val preMs = SystemClock.elapsedRealtime() - t0

    scope.launch {
      var inferenceMs = -1L
      try {
        val (inputTensor, scaleX, scaleY) = preprocess(rotated)
        val inputName = session.inputNames.first()

        val tInf0 = SystemClock.elapsedRealtime()
        val outputs = session.run(mapOf(inputName to inputTensor))
        inferenceMs = SystemClock.elapsedRealtime() - tInf0
        inputTensor.close()

        val dets = postprocess(outputs, scaleX, scaleY, rotated.width, rotated.height)
        outputs.close()

        withContext(Dispatchers.Main) { onDetections?.invoke(dets, rotated.width, rotated.height, inferenceMs) }
        logger.log("frame_done preMs=$preMs infMs=$inferenceMs dets=${dets.size}")
      } catch (t: Throwable) {
        logger.logE("inference_failed preMs=$preMs infMs=$inferenceMs", t)
      } finally {
        busy.set(false)
      }
    }
  }

  private fun preprocess(src: Bitmap): Triple<OnnxTensor, Float, Float> {
    // Pozn.: tady děláme "stretch" do 640x640 (bez letterbox). Mapování zpět proto používá scaleX/scaleY zvlášť.
    val resized = Bitmap.createScaledBitmap(src, inputSize, inputSize, true)

    val floatBuffer = FloatBuffer.allocate(1 * 3 * inputSize * inputSize)
    val pixels = IntArray(inputSize * inputSize)
    resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

    var idxR = 0
    var idxG = inputSize * inputSize
    var idxB = 2 * inputSize * inputSize
    for (p in pixels) {
      floatBuffer.put(idxR++, ((p shr 16) and 0xFF) / 255f)
      floatBuffer.put(idxG++, ((p shr 8) and 0xFF) / 255f)
      floatBuffer.put(idxB++, (p and 0xFF) / 255f)
    }

    val shape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
    val tensor = OnnxTensor.createTensor(env, floatBuffer, shape)

    val scaleX = src.width.toFloat() / inputSize.toFloat()
    val scaleY = src.height.toFloat() / inputSize.toFloat()
    return Triple(tensor, scaleX, scaleY)
  }

  private fun postprocess(
    outputs: OrtSession.Result,
    scaleX: Float,
    scaleY: Float,
    frameW: Int,
    frameH: Int
  ): List<Detection> {

    val outVal = outputs[0]
    val outTensor = (outVal as? OnnxTensor) ?: run {
      logger.logE("output[0] is not OnnxTensor, type=${outVal.javaClass.name}")
      return emptyList()
    }

    val info = outTensor.info as? TensorInfo ?: run {
      logger.logE("output tensor info is not TensorInfo: ${outTensor.info}")
      return emptyList()
    }

    val shape = info.shape
    logger.log("output_shape=${shape.joinToString("x")}")

    val fb = outTensor.floatBuffer
    fb.rewind()
    val arr = FloatArray(fb.remaining())
    fb.get(arr)

    if (shape.size != 3) {
      logger.logE("unexpected output rank=${shape.size}")
      return emptyList()
    }

    val candidates = mutableListOf<Candidate>()
    val d1 = shape[1].toInt()
    val d2 = shape[2].toInt()

    var maxScore = 0f
    var coordScaleHint = "unknown"
    var sampleLogged = false

    // YOLOv11 export typicky: [1,84,8400] => channels_first
    if (d1 == 84) {
      val n = d2
      for (i in 0 until n) {
        var cx = arr[0 * n + i]
        var cy = arr[1 * n + i]
        var w  = arr[2 * n + i]
        var h  = arr[3 * n + i]

        // ✅ KRITICKÝ FIX: některé exporty dávají xywh normalizované 0..1 (místo 0..640).
        // Pokud to vypadá jako normalizované, přepočti na pixelové souřadnice v input prostoru (0..640).
        val looksNormalized = (max(max(cx, cy), max(w, h)) <= 1.5f)
        if (looksNormalized) {
          cx *= inputSize
          cy *= inputSize
          w *= inputSize
          h *= inputSize
          coordScaleHint = "normalized_xywh"
        } else {
          coordScaleHint = "pixel_xywh"
        }

        var bestC = -1
        var bestS = 0f
        for (c in 0 until 80) {
          val s = arr[(4 + c) * n + i]
          if (s > bestS) { bestS = s; bestC = c }
        }

        if (!sampleLogged && bestS > 0.6f) {
          sampleLogged = true
          logger.log("sample_xywh $coordScaleHint cx=$cx cy=$cy w=$w h=$h score=$bestS cls=$bestC scaleX=$scaleX scaleY=$scaleY frame=${frameW}x${frameH}")
        }

        if (bestS > maxScore) maxScore = bestS
        if (bestS >= confThreshold) candidates.add(Candidate(cx, cy, w, h, bestS, bestC))
      }
      logger.log("decode_layout=channels_first n=$n maxScore=$maxScore candidates=${candidates.size} coordScale=$coordScaleHint")
    } else if (d2 == 84) {
      val n = d1
      for (i in 0 until n) {
        val base = i * 84
        var cx = arr[base + 0]
        var cy = arr[base + 1]
        var w  = arr[base + 2]
        var h  = arr[base + 3]

        val looksNormalized = (max(max(cx, cy), max(w, h)) <= 1.5f)
        if (looksNormalized) {
          cx *= inputSize
          cy *= inputSize
          w *= inputSize
          h *= inputSize
          coordScaleHint = "normalized_xywh"
        } else {
          coordScaleHint = "pixel_xywh"
        }

        var bestC = -1
        var bestS = 0f
        for (c in 0 until 80) {
          val s = arr[base + 4 + c]
          if (s > bestS) { bestS = s; bestC = c }
        }

        if (!sampleLogged && bestS > 0.6f) {
          sampleLogged = true
          logger.log("sample_xywh $coordScaleHint cx=$cx cy=$cy w=$w h=$h score=$bestS cls=$bestC scaleX=$scaleX scaleY=$scaleY frame=${frameW}x${frameH}")
        }

        if (bestS > maxScore) maxScore = bestS
        if (bestS >= confThreshold) candidates.add(Candidate(cx, cy, w, h, bestS, bestC))
      }
      logger.log("decode_layout=anchors_last n=$n maxScore=$maxScore candidates=${candidates.size} coordScale=$coordScaleHint")
    } else {
      logger.logE("unexpected output dims d1=$d1 d2=$d2 (expected 84xN or Nx84)")
      return emptyList()
    }

    val nms = nonMaxSuppression(candidates, iouThreshold)
    logger.log("nms kept=${nms.size}")

    return nms.map { cand ->
      // cx,cy,w,h jsou v INPUT prostoru (0..640) => převod na frame pixely přes scaleX/scaleY
      val x1px = (cand.cx - cand.w / 2f) * scaleX
      val y1px = (cand.cy - cand.h / 2f) * scaleY
      val x2px = (cand.cx + cand.w / 2f) * scaleX
      val y2px = (cand.cy + cand.h / 2f) * scaleY

      // normalizace do 0..1 ve FRAME prostoru
      val nx1 = (x1px / frameW).coerceIn(0f, 1f)
      val ny1 = (y1px / frameH).coerceIn(0f, 1f)
      val nx2 = (x2px / frameW).coerceIn(0f, 1f)
      val ny2 = (y2px / frameH).coerceIn(0f, 1f)

      val label = labelsCs[cand.cls] ?: "objekt"

      // vzdálenost: z výšky bboxu v pixelech + FOV => fy
      val boxHPx = max(1f, (ny2 - ny1) * frameH.toFloat())
      val fy = (frameH.toFloat() / 2f) / tan(Math.toRadians(assumedVerticalFovDeg / 2.0)).toFloat()
      val realH = classHeightsM[label] ?: 0.50f
      val dist = (realH * fy / boxHPx).coerceIn(0.05f, 20f)

      // směr v rámci obrazu (po rotaci)
      val cxn = (nx1 + nx2) / 2f
      val pos = when {
        cxn < 0.33f -> "vlevo"
        cxn > 0.66f -> "vpravo"
        else -> "uprostřed"
      }

      // jednorázově zaloguj pár bboxů po převodu, ať hned vidíme jestli to sedí
      if (!sampleLogged) {
        sampleLogged = true
        logger.log("sample_bbox label=$label nx1=$nx1 ny1=$ny1 nx2=$nx2 ny2=$ny2 boxHPx=$boxHPx dist=$dist pos=$pos")
      }

      Detection(label, cand.score, nx1, ny1, nx2, ny2, dist, pos)
    }
  }

  private data class Candidate(
    val cx: Float, val cy: Float, val w: Float, val h: Float,
    val score: Float, val cls: Int
  )

  private fun nonMaxSuppression(cands: List<Candidate>, iouTh: Float): List<Candidate> {
    val sorted = cands.sortedByDescending { it.score }.toMutableList()
    val keep = mutableListOf<Candidate>()

    fun iou(a: Candidate, b: Candidate): Float {
      val ax1 = a.cx - a.w/2f
      val ay1 = a.cy - a.h/2f
      val ax2 = a.cx + a.w/2f
      val ay2 = a.cy + a.h/2f
      val bx1 = b.cx - b.w/2f
      val by1 = b.cy - b.h/2f
      val bx2 = b.cx + b.w/2f
      val by2 = b.cy + b.h/2f
      val ix1 = max(ax1, bx1)
      val iy1 = max(ay1, by1)
      val ix2 = min(ax2, bx2)
      val iy2 = min(ay2, by2)
      val iw = max(0f, ix2 - ix1)
      val ih = max(0f, iy2 - iy1)
      val inter = iw * ih
      val union = a.w*a.h + b.w*b.h - inter
      return if (union <= 0f) 0f else inter / union
    }

    while (sorted.isNotEmpty()) {
      val best = sorted.removeAt(0)
      keep.add(best)
      val it = sorted.iterator()
      while (it.hasNext()) {
        val c = it.next()
        if (c.cls == best.cls && iou(c, best) > iouTh) it.remove()
      }
    }
    return keep
  }

  private fun loadLabels(asset: String): Map<Int, String> {
    val txt = context.assets.open(asset).bufferedReader().use { it.readText() }
    val json = Json.parseToJsonElement(txt).jsonObject
    return json.mapNotNull { (k, v) ->
      k.toIntOrNull()?.let { it to v.jsonPrimitive.content }
    }.toMap()
  }

  override fun close() {
    scope.cancel()
    session.close()
    env.close()
  }
}

private fun ImageProxy.toBitmap(): Bitmap {
  val yBuffer = planes[0].buffer
  val uBuffer = planes[1].buffer
  val vBuffer = planes[2].buffer

  val ySize = yBuffer.remaining()
  val uSize = uBuffer.remaining()
  val vSize = vBuffer.remaining()

  val nv21 = ByteArray(ySize + uSize + vSize)
  yBuffer.get(nv21, 0, ySize)
  vBuffer.get(nv21, ySize, vSize)
  uBuffer.get(nv21, ySize + vSize, uSize)

  val yuvImage = android.graphics.YuvImage(nv21, android.graphics.ImageFormat.NV21, width, height, null)
  val out = java.io.ByteArrayOutputStream()
  yuvImage.compressToJpeg(android.graphics.Rect(0, 0, width, height), 80, out)
  val bytes = out.toByteArray()
  return android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
}

private fun Bitmap.rotate(deg: Float): Bitmap {
  if (deg == 0f) return this
  val m = Matrix().apply { postRotate(deg) }
  return Bitmap.createBitmap(this, 0, 0, width, height, m, true)
}
