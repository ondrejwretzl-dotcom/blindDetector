package com.example.blinddetektor.ml

data class Detection(
  val labelCs: String,
  val score: Float,
  val x1: Float, val y1: Float, val x2: Float, val y2: Float,
  val distanceMeters: Float,
  val positionHint: String,
  val hasLabel: Boolean
)
