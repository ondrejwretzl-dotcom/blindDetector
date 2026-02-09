package com.example.blinddetektor.camera

import android.annotation.SuppressLint
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.example.blinddetektor.util.BDLogger
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import kotlin.system.measureTimeMillis

class CameraController(
  private val activity: LifecycleOwner,
  private val previewView: PreviewView,
  private val logger: BDLogger,
  private val onFrame: (image: androidx.camera.core.ImageProxy, rotationDegrees: Int) -> Unit
) {

  private val cameraExecutor = Executors.newSingleThreadExecutor()
  private var cameraProvider: ProcessCameraProvider? = null

  private val frameCounter = AtomicInteger(0)
  @Volatile private var lastFpsAt = System.currentTimeMillis()

  @SuppressLint("UnsafeOptInUsageError")
  fun start() {
    val providerFuture = ProcessCameraProvider.getInstance(previewView.context)
    providerFuture.addListener({
      cameraProvider = providerFuture.get()
      val provider = cameraProvider ?: return@addListener

      val preview = Preview.Builder().build().also {
        it.setSurfaceProvider(previewView.surfaceProvider)
      }

      val analysis = ImageAnalysis.Builder()
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
        .build()

      analysis.setAnalyzer(cameraExecutor) { image ->
        val rotation = image.imageInfo.rotationDegrees
        try {
          val ms = measureTimeMillis { onFrame(image, rotation) }
          val n = frameCounter.incrementAndGet()
          val now = System.currentTimeMillis()
          if (now - lastFpsAt >= 2000) {
            val fps = n * 1000.0 / (now - lastFpsAt).toDouble()
            logger.log("camera_frames fps=${String.format("%.1f", fps)} lastOnFrameMs=$ms rot=$rotation size=${image.width}x${image.height}")
            frameCounter.set(0)
            lastFpsAt = now
          }
        } catch (t: Throwable) {
          logger.logE("camera_onFrame exception", t)
        } finally {
          image.close()
        }
      }

      val selector = androidx.camera.core.CameraSelector.DEFAULT_BACK_CAMERA
      provider.unbindAll()
      provider.bindToLifecycle(activity, selector, preview, analysis)
      logger.log("camera_bind OK")
    }, ContextCompat.getMainExecutor(previewView.context))
  }

  fun stop() {
    logger.log("camera_stop")
    cameraProvider?.unbindAll()
    cameraExecutor.shutdown()
  }
}
