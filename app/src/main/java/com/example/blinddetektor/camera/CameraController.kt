package com.example.blinddetektor.camera

import android.annotation.SuppressLint
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.Executors

class CameraController(
  private val activity: LifecycleOwner,
  private val previewView: PreviewView,
  private val onFrame: (image: androidx.camera.core.ImageProxy, rotationDegrees: Int) -> Unit
) {

  private val cameraExecutor = Executors.newSingleThreadExecutor()
  private var cameraProvider: ProcessCameraProvider? = null

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
        onFrame(image, rotation)
        image.close()
      }

      val selector = androidx.camera.core.CameraSelector.DEFAULT_BACK_CAMERA

      provider.unbindAll()
      provider.bindToLifecycle(activity, selector, preview, analysis)
    }, ContextCompat.getMainExecutor(previewView.context))
  }

  fun stop() {
    cameraProvider?.unbindAll()
    cameraExecutor.shutdown()
  }
}
