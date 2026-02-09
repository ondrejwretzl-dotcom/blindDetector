package com.example.blinddetektor.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import com.example.blinddetektor.ml.Detection

class OverlayView @JvmOverloads constructor(
  context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

  private val boxPaint = Paint().apply {
    style = Paint.Style.STROKE
    strokeWidth = 6f
    color = Color.GREEN
    isAntiAlias = true
  }

  private val textPaint = Paint().apply {
    style = Paint.Style.FILL
    color = Color.GREEN
    textSize = 44f
    isAntiAlias = true
    typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
  }

  private val bgPaint = Paint().apply {
    style = Paint.Style.FILL
    color = Color.argb(140, 0, 0, 0)
  }

  @Volatile private var detections: List<Detection> = emptyList()

  fun updateDetections(dets: List<Detection>, frameW: Int, frameH: Int) {
    detections = dets
    postInvalidateOnAnimation()
  }

  fun getLastDetections(): List<Detection> = detections

  override fun onDraw(canvas: Canvas) {
    super.onDraw(canvas)

    val w = width.toFloat()
    val h = height.toFloat()

    for (d in detections) {
      val left = d.x1 * w
      val top = d.y1 * h
      val right = d.x2 * w
      val bottom = d.y2 * h

      canvas.drawRect(left, top, right, bottom, boxPaint)

      val label = "${d.labelCs} • ~${"%.1f".format(d.distanceMeters)} m"
      val textW = textPaint.measureText(label)
      val textH = textPaint.textSize

      val bgRect = RectF(left, top - textH - 14f, left + textW + 18f, top)
      canvas.drawRoundRect(bgRect, 10f, 10f, bgPaint)
      canvas.drawText(label, left + 9f, top - 10f, textPaint)
    }
  }
}
