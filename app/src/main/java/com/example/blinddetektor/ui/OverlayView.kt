package com.example.blinddetektor.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import com.example.blinddetektor.ml.Detection
import kotlin.math.min

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
  @Volatile private var frameW: Int = 0
  @Volatile private var frameH: Int = 0

  /**
   * dets mají normalizované souřadnice vůči frameW/frameH (po rotaci do portrait).
   * PreviewView nastavujeme na fitCenter, takže mapování = letterbox:
   *   scale = min(viewW/frameW, viewH/frameH), dx/dy = centrování.
   */
  fun updateDetections(dets: List<Detection>, frameW: Int, frameH: Int) {
    this.detections = dets
    this.frameW = frameW
    this.frameH = frameH
    postInvalidateOnAnimation()
  }

  fun getLastDetections(): List<Detection> = detections

  override fun onDraw(canvas: Canvas) {
    super.onDraw(canvas)
    val vw = width.toFloat()
    val vh = height.toFloat()
    val fw = frameW.toFloat()
    val fh = frameH.toFloat()
    if (fw <= 0f || fh <= 0f) return

    val scale = min(vw / fw, vh / fh)
    val dx = (vw - fw * scale) / 2f
    val dy = (vh - fh * scale) / 2f

    for (d in detections) {
      // detection coords are normalized in frame space
      val left = dx + (d.x1 * fw) * scale
      val top = dy + (d.y1 * fh) * scale
      val right = dx + (d.x2 * fw) * scale
      val bottom = dy + (d.y2 * fh) * scale

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
