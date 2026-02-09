package com.example.blinddetektor.util

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Logger který ukládá soubory do veřejné složky:
 *   Download/blindDetector/
 *
 * API 29+ používá MediaStore + RELATIVE_PATH (bez zvláštních permission).
 * API <=28 zapisuje přímo do Environment.getExternalStoragePublicDirectory (vyžaduje WRITE_EXTERNAL_STORAGE).
 *
 * Pro jednoduchost ukládáme jeden log soubor na "session" (spuštění appky).
 */
class BDLogger(private val context: Context) {

  companion object {
    private const val TAG = "BDLogger"
    private const val DIR = "Download/blindDetector/"
    private val tsFmt = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US)
    private val fileFmt = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
  }

  private var uri: Uri? = null
  private var legacyFile: File? = null

  val fileName: String = "log_${fileFmt.format(Date())}.txt"

  fun init(): Result<String> {
    return try {
      if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
        val values = ContentValues().apply {
          put(MediaStore.Downloads.DISPLAY_NAME, fileName)
          put(MediaStore.Downloads.MIME_TYPE, "text/plain")
          put(MediaStore.Downloads.RELATIVE_PATH, DIR)
          put(MediaStore.Downloads.IS_PENDING, 1)
        }
        val resolver = context.contentResolver
        val u = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
          ?: return Result.failure(IllegalStateException("MediaStore insert failed"))
        uri = u
        resolver.openOutputStream(u, "w")?.use { /* create */ }
        values.clear()
        values.put(MediaStore.Downloads.IS_PENDING, 0)
        resolver.update(u, values, null, null)
        Result.success("Download/blindDetector/$fileName")
      } else {
        val downloads = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
        val dir = File(downloads, "blindDetector")
        if (!dir.exists()) dir.mkdirs()
        val f = File(dir, fileName)
        legacyFile = f
        FileOutputStream(f, true).use { /* create */ }
        Result.success(f.absolutePath)
      }
    } catch (t: Throwable) {
      Log.e(TAG, "init failed: ${t.message}", t)
      Result.failure(t)
    }
  }

  fun log(msg: String) {
    val line = "${tsFmt.format(Date())} | $msg\n"
    try {
      if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
        val u = uri ?: run {
          Log.w(TAG, "uri not initialized yet")
          return
        }
        // "wa" = write append (funguje na většině zařízení; když by selhalo, uvidíš to v Logcat)
        context.contentResolver.openOutputStream(u, "wa")?.use { out ->
          out.write(line.toByteArray(Charsets.UTF_8))
        } ?: run {
          Log.w(TAG, "openOutputStream returned null")
        }
      } else {
        val f = legacyFile ?: run {
          Log.w(TAG, "legacyFile not initialized yet")
          return
        }
        FileOutputStream(f, true).use { out ->
          out.write(line.toByteArray(Charsets.UTF_8))
        }
      }
    } catch (t: Throwable) {
      Log.e(TAG, "log failed: ${t.message}", t)
    }
  }

  fun logE(msg: String, t: Throwable? = null) {
    log("ERROR | $msg" + (t?.let { " | ${it.javaClass.simpleName}: ${it.message}" } ?: ""))
    if (t != null) Log.e(TAG, msg, t) else Log.e(TAG, msg)
  }
}
