package com.example.blinddetektor.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import androidx.camera.core.ImageProxy
import ai.onnxruntime.*
import kotlinx.coroutines.*
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.nio.FloatBuffer
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.*

class YoloV8OnnxDetector(
  private val context: Context,
  modelAssetName: String,
  labelsAssetName: String,
) : AutoCloseable {

  companion object { private const val TAG = "YoloV8OnnxDetector" }

  private val inputSize = 640
  private val confThreshold = 0.35f
  private val iouThreshold = 0.45f

  private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
  private val session: OrtSession
  private val labelsCs: Map<Int, String>

  private val minInferenceIntervalMs = 250L
  private val lastInferenceAt = AtomicLong(0L)
  private val busy = AtomicBoolean(false)

  var onDetections: ((List<Detection>, Int, Int) -> Unit)? = null
  private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

  init {
    val modelBytes = context.assets.open(modelAssetName).readBytes()

    val opts = OrtSession.SessionOptions().apply {
      setIntraOpNumThreads(2)
      setInterOpNumThreads(1)
      try {
        addNnapi()
        Log.i(TAG, "NNAPI EP enabled")
      } catch (t: Throwable) {
        Log.w(TAG, "NNAPI EP not available: ${t.javaClass.simpleName}: ${t.message}")
      }
    }

    try {
      session = env.createSession(modelBytes, opts)
    } catch (e: OrtException) {
      Log.e(TAG, "OrtException while creating session: ${e.message}", e)
      throw e
    }

    labelsCs = loadLabels(labelsAssetName)
  }

  fun detect(image: ImageProxy, rotationDegrees: Int) {
    val now = System.currentTimeMillis()
    val last = lastInferenceAt.get()
    if (now - last < minInferenceIntervalMs) return
    if (!lastInferenceAt.compareAndSet(last, now)) return
    if (!busy.compareAndSet(false, true)) return

    val bitmap = image.toBitmap()
    val rotated = bitmap.rotate(rotationDegrees.toFloat())

    scope.launch {
      try {
        val (inputTensor, scaleX, scaleY) = preprocess(rotated)
        val inputName = session.inputNames.first()
        val outputs = session.run(mapOf(inputName to inputTensor))
        inputTensor.close()

        val dets = postprocess(outputs, scaleX, scaleY, rotated.width, rotated.height)
        outputs.close()

        withContext(Dispatchers.Main) { onDetections?.invoke(dets, rotated.width, rotated.height) }
      } catch (t: Throwable) {
        Log.w(TAG, "Inference failed: ${t.javaClass.simpleName}: ${t.message}")
      } finally {
        busy.set(false)
      }
    }
  }

  private fun preprocess(src: Bitmap): Triple<OnnxTensor, Float, Float> {
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

  private fun postprocess(outputs: OrtSession.Result, scaleX: Float, scaleY: Float, frameW: Int, frameH: Int): List<Detection> {
    val tensor = outputs[0].value as? FloatArray ?: return emptyList()
    val info = outputs[0].info as? TensorInfo ?: return emptyList()
    val shape = info.shape
    if (shape.size != 3) return emptyList()

    val candidates = mutableListOf<Candidate>()
    val d1 = shape[1].toInt()
    val d2 = shape[2].toInt()

    if (d2 == 84) {
      val n = d1
      for (i in 0 until n) {
        val base = i * 84
        val cx = tensor[base + 0]; val cy = tensor[base + 1]; val w = tensor[base + 2]; val h = tensor[base + 3]
        var bestC = -1; var bestS = 0f
        for (c in 0 until 80) { val s = tensor[base + 4 + c]; if (s > bestS) { bestS = s; bestC = c } }
        if (bestS >= confThreshold) candidates.add(Candidate(cx, cy, w, h, bestS, bestC))
      }
    } else if (d1 == 84) {
      val n = d2
      for (i in 0 until n) {
        val cx = tensor[0 * n + i]; val cy = tensor[1 * n + i]; val w = tensor[2 * n + i]; val h = tensor[3 * n + i]
        var bestC = -1; var bestS = 0f
        for (c in 0 until 80) { val s = tensor[(4 + c) * n + i]; if (s > bestS) { bestS = s; bestC = c } }
        if (bestS >= confThreshold) candidates.add(Candidate(cx, cy, w, h, bestS, bestC))
      }
    } else return emptyList()

    val nms = nonMaxSuppression(candidates, iouThreshold)
    return nms.map { cand ->
      val x1 = (cand.cx - cand.w / 2f) * scaleX
      val y1 = (cand.cy - cand.h / 2f) * scaleY
      val x2 = (cand.cx + cand.w / 2f) * scaleX
      val y2 = (cand.cy + cand.h / 2f) * scaleY
      val nx1 = (x1 / frameW).coerceIn(0f, 1f)
      val ny1 = (y1 / frameH).coerceIn(0f, 1f)
      val nx2 = (x2 / frameW).coerceIn(0f, 1f)
      val ny2 = (y2 / frameH).coerceIn(0f, 1f)
      val hasLabel = labelsCs.containsKey(cand.cls)
      val label = labelsCs[cand.cls] ?: "objekt"
      val dist = (0.35f / sqrt(max(1e-6f, (nx2 - nx1) * (ny2 - ny1)))).coerceIn(0.3f, 8f)
      val cxn = (nx1 + nx2) / 2f
      val pos = when { cxn < 0.33f -> "vlevo"; cxn > 0.66f -> "vpravo"; else -> "uprostřed" }
      Detection(label, cand.score, nx1, ny1, nx2, ny2, dist, pos, hasLabel)
    }
  }

  private data class Candidate(val cx: Float, val cy: Float, val w: Float, val h: Float, val score: Float, val cls: Int)

  private fun nonMaxSuppression(cands: List<Candidate>, iouTh: Float): List<Candidate> {
    val sorted = cands.sortedByDescending { it.score }.toMutableList()
    val keep = mutableListOf<Candidate>()
    fun iou(a: Candidate, b: Candidate): Float {
      val ax1 = a.cx - a.w/2f; val ay1 = a.cy - a.h/2f; val ax2 = a.cx + a.w/2f; val ay2 = a.cy + a.h/2f
      val bx1 = b.cx - b.w/2f; val by1 = b.cy - b.h/2f; val bx2 = b.cx + b.w/2f; val by2 = b.cy + b.h/2f
      val ix1 = max(ax1, bx1); val iy1 = max(ay1, by1); val ix2 = min(ax2, bx2); val iy2 = min(ay2, by2)
      val iw = max(0f, ix2 - ix1); val ih = max(0f, iy2 - iy1)
      val inter = iw * ih; val union = a.w*a.h + b.w*b.h - inter
      return if (union <= 0f) 0f else inter / union
    }
    while (sorted.isNotEmpty()) {
      val best = sorted.removeAt(0); keep.add(best)
      val it = sorted.iterator()
      while (it.hasNext()) { val c = it.next(); if (c.cls == best.cls && iou(c, best) > iouTh) it.remove() }
    }
    return keep
  }

  private fun loadLabels(asset: String): Map<Int, String> {
    val txt = context.assets.open(asset).bufferedReader().use { it.readText() }
    val json = Json.parseToJsonElement(txt).jsonObject
    return json.mapNotNull { (k, v) -> k.toIntOrNull()?.let { it to v.jsonPrimitive.content } }.toMap()
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
  val ySize = yBuffer.remaining(); val uSize = uBuffer.remaining(); val vSize = vBuffer.remaining()
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
