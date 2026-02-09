package com.example.blinddetektor.speech

import com.example.blinddetektor.ml.Detection
import kotlin.math.abs
import kotlin.math.max

class RelevancePolicy {

  private val labelCooldownMs = 8_000L
  private val globalMinIntervalMs = 2_500L
  private var lastGlobalSpokenAt = 0L
  private val lastSpokenPerLabel = HashMap<String, Long>()

  private var lastSignature: String = ""

  fun resetAutoState() {
    lastGlobalSpokenAt = 0L
    lastSpokenPerLabel.clear()
    lastSignature = ""
  }

  fun pickForManualSpeech(dets: List<Detection>): List<Detection> {
    return dets.sortedByDescending { relevanceScore(it) }
      .take(4)
  }

  fun pickForAutoSpeech(dets: List<Detection>): List<Detection> {
    val now = System.currentTimeMillis()
    if (now - lastGlobalSpokenAt < globalMinIntervalMs) return emptyList()

    val sorted = dets.sortedByDescending { relevanceScore(it) }
    val top = sorted.take(3)

    val signature = top.joinToString("|") { "${it.labelCs}:${it.positionHint}:${it.distanceBucket()}" }
    val changed = signature != lastSignature
    if (!changed) return emptyList()

    val filtered = top.filter {
      val last = lastSpokenPerLabel[it.labelCs] ?: 0L
      now - last >= labelCooldownMs
    }
    if (filtered.isEmpty()) return emptyList()

    lastSignature = signature
    lastGlobalSpokenAt = now
    filtered.forEach { lastSpokenPerLabel[it.labelCs] = now }
    return filtered
  }

  fun formatForSpeech(dets: List<Detection>, withPositions: Boolean = true): String {
    val parts = dets.map {
      val dist = it.distanceMeters
      val distText = when {
        dist < 2.0f -> "asi ${"%.1f".format(dist)} metru"
        else -> "asi ${"%.0f".format(dist)} metry"
      }
      if (withPositions) "${it.labelCs} ${it.positionHint}, $distText"
      else "${it.labelCs}, $distText"
    }
    return parts.joinToString(". ")
  }

  private fun relevanceScore(d: Detection): Float {
    val cx = (d.x1 + d.x2) / 2f
    val center = 1f - abs(cx - 0.5f) * 2f
    val proximity = (1f / max(0.3f, d.distanceMeters))
    return d.score * 0.65f + center * 0.15f + proximity * 0.20f
  }

  private fun Detection.distanceBucket(): String =
    when {
      distanceMeters < 1.0f -> "near"
      distanceMeters < 2.5f -> "mid"
      else -> "far"
    }
}
