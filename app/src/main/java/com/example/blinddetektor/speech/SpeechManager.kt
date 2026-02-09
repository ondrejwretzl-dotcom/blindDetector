package com.example.blinddetektor.speech

import android.content.Context
import android.speech.tts.TextToSpeech
import java.util.Locale

class SpeechManager(context: Context) : AutoCloseable {

  private var tts: TextToSpeech? = null
  @Volatile private var ready = false

  init {
    tts = TextToSpeech(context.applicationContext) { status ->
      ready = (status == TextToSpeech.SUCCESS).also {
        if (it) {
          tts?.language = Locale("cs", "CZ")
          tts?.setSpeechRate(1.0f)
        }
      }
    }
  }

  fun speak(text: String) {
    val engine = tts ?: return
    if (!ready) return
    engine.speak(text, TextToSpeech.QUEUE_FLUSH, null, "blinddetektor")
  }

  override fun close() {
    tts?.stop()
    tts?.shutdown()
    tts = null
  }
}
