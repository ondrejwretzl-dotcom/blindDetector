package com.example.blinddetektor.util

import java.text.Normalizer
import java.util.Locale

fun normalizeCs(s: String): String {
  val lower = s.lowercase(Locale("cs", "CZ")).trim()
  val norm = Normalizer.normalize(lower, Normalizer.Form.NFD)
  return norm.replace("\\p{InCombiningDiacriticalMarks}+".toRegex(), "")
}

