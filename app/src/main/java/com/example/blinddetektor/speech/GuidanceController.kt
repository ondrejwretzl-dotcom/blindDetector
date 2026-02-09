package com.example.blinddetektor.speech

import com.example.blinddetektor.ml.Detection
import kotlin.math.abs
import kotlin.math.max

class GuidanceController(private val speech: SpeechManager) {

  private var target: String? = null
  private var active = false
  private var lastPromptAt = 0L
  private var lastUpdateAt = 0L
  private var lastSpoken = ""

  fun isActive(): Boolean = active

  fun start(targetLabel: String) {
    target = targetLabel
    active = true
    lastPromptAt = 0L
    lastUpdateAt = 0L
    lastSpoken = ""
    speech.speak("Hledám $targetLabel. Hýbej prosím telefonem, dokud se objekt neobjeví v záběru.")
  }

  fun stop() {
    if (active) speech.speak("Navigace ukončena.")
    active = false
    target = null
  }

  fun onDetections(dets: List<Detection>) {
    if (!active) return
    val t = target ?: return
    val now = System.currentTimeMillis()

    if (now - lastUpdateAt < 900) return
    lastUpdateAt = now

    val match = dets
      .filter { it.labelCs.equals(t, ignoreCase = true) }
      .maxByOrNull { relevanceForGuidance(it) }

    if (match == null) {
      if (now - lastPromptAt > 2500) {
        lastPromptAt = now
        speech.speak("Zatím nevidím $t. Pomalu otáčej telefonem do stran.")
      }
      return
    }

    val dir = when {
      match.x2 < 0.45f -> "je vlevo"
      match.x1 > 0.55f -> "je vpravo"
      else -> "je uprostřed"
    }

    val action = when {
      match.x2 < 0.45f -> "Otoč trochu doleva."
      match.x1 > 0.55f -> "Otoč trochu doprava."
      match.distanceMeters > 0.6f -> "Přibliž se."
      else -> "Jsi u něj."
    }

    val distCm = (match.distanceMeters * 100f).toInt().coerceAtLeast(1)
    val msg = "$t $dir, asi $distCm centimetrů. $action"

    if (msg == lastSpoken) return
    lastSpoken = msg
    speech.speak(msg)

    if (match.distanceMeters < 0.25f && match.x1 < 0.55f && match.x2 > 0.45f) {
      speech.speak("Hotovo.")
      stop()
    }
  }

  private fun relevanceForGuidance(d: Detection): Float {
    val cx = (d.x1 + d.x2) / 2f
    val center = 1f - abs(cx - 0.5f) * 2f
    val proximity = 1f / max(0.15f, d.distanceMeters)
    return d.score * 0.6f + center * 0.2f + proximity * 0.2f
  }
}
