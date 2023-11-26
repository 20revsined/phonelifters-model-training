package com.phonelifters.pose_detection

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.widget.Button
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.phonelifters.pose_detection.R
import com.phonelifters.pose_detection.ml.LiteModelMovenetSingleposeLightningTfliteFloat164
import com.phonelifters.pose_detection.ml.Model1
import com.phonelifters.pose_detection.ml.Model2
//import kotlinx.coroutines.NonCancellable.message
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : AppCompatActivity() {

    val paint = Paint()
    lateinit var imageProcessor: ImageProcessor
    lateinit var model: LiteModelMovenetSingleposeLightningTfliteFloat164
    lateinit var bitmap: Bitmap
    lateinit var imageView: ImageView
    lateinit var handler:Handler
    lateinit var handlerThread: HandlerThread
    lateinit var textureView: TextureView
    lateinit var cameraManager: CameraManager
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        get_permissions()

        imageProcessor = ImageProcessor.Builder().add(ResizeOp(192, 192, ResizeOp.ResizeMethod.BILINEAR)).build()
        model = LiteModelMovenetSingleposeLightningTfliteFloat164.newInstance(this)
        val classifier1 = Model1.newInstance(this)
        val classifier2 = Model2.newInstance(this)
        imageView = findViewById(R.id.imageView)
        textureView = findViewById(R.id.textureView)
        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        handlerThread = HandlerThread("videoThread")
        handlerThread.start()
        handler = Handler(handlerThread.looper)

        paint.setColor(Color.YELLOW)

        textureView.surfaceTextureListener = object:TextureView.SurfaceTextureListener{
            override fun onSurfaceTextureAvailable(p0: SurfaceTexture, p1: Int, p2: Int) {
                open_camera()
            }

            override fun onSurfaceTextureSizeChanged(p0: SurfaceTexture, p1: Int, p2: Int) {

            }

            override fun onSurfaceTextureDestroyed(p0: SurfaceTexture): Boolean {
                return false
            }

            override fun onSurfaceTextureUpdated(p0: SurfaceTexture) {
                bitmap = textureView.bitmap!!
                var tensorImage = TensorImage(DataType.UINT8)
                tensorImage.load(bitmap)
                tensorImage = imageProcessor.process(tensorImage)

                val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 192, 192, 3), DataType.UINT8)

                inputFeature0.loadBuffer(tensorImage.buffer)

                val outputs = model.process(inputFeature0)
                val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

                var mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                var canvas = Canvas(mutable)
                var h = bitmap.height
                var w = bitmap.width
                var x = 0

                var currentRow = mutableListOf<Float>()
                var a = 0f
                var b = 0f
                while(x <= 49){
                    if(outputFeature0.get(x+2) > 0.45){
                        canvas.drawCircle(outputFeature0.get(x+1)*w, outputFeature0.get(x)*h, 10f, paint)
                        //append to CSV file
                        a = outputFeature0.get(x+1)*w
                        b = outputFeature0.get(x)*h

                    }

                    else
                    {
                        a = 0f
                        b = 0f
                    }

                    currentRow.add(a)
                    currentRow.add(b)

                    /*
                    val a = outputFeature0.get(x+1)*w
                    val b = outputFeature0.get(x)*h
                    currentRow.add(a)
                    currentRow.add(b)
                     */
                    x+=3
                }

                val input1 = TensorBuffer.createDynamic(DataType.FLOAT32)
                input1.loadArray(currentRow.toFloatArray(), intArrayOf(34))
                val output1 = classifier1.process(input1)
                val outputFeature1 = output1.outputFeature0AsTensorBuffer.floatArray

                val input2 = TensorBuffer.createDynamic(DataType.FLOAT32)
                input2.loadArray(currentRow.toFloatArray(), intArrayOf(34))
                val output2 = classifier2.process(input2)
                val outputFeature2 = output2.outputFeature0AsTensorBuffer.floatArray

                lateinit var message: Button
                lateinit var done: Button
                message = findViewById(R.id.accuracy)
                done = findViewById(R.id.done)
                //message = findViewById(R.id.accuracy_message)

                if (outputFeature1.get(0) == 1f && outputFeature2.get(0) == 1f)
                {

                    Log.d("accuracy", "correct")
                    message.text = "Correct!"
                }

                else
                {
                    Log.d("accuracy", "incorrect")
                    message.text = "Keep Trying"
                }

                message.display
                done.display

                done.setOnClickListener({ returnToMainApp() })


                imageView.setImageBitmap(mutable)
            }
        }

    }

    override fun onDestroy() {
        super.onDestroy()
        model.close()
    }

    @SuppressLint("MissingPermission")
    fun open_camera(){
        cameraManager.openCamera(cameraManager.cameraIdList[0], object:CameraDevice.StateCallback(){
            override fun onOpened(p0: CameraDevice) {
                var captureRequest = p0.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                var surface = Surface(textureView.surfaceTexture)
                captureRequest.addTarget(surface)
                p0.createCaptureSession(listOf(surface), object:CameraCaptureSession.StateCallback(){
                    override fun onConfigured(p0: CameraCaptureSession) {
                        p0.setRepeatingRequest(captureRequest.build(), null, null)
                    }
                    override fun onConfigureFailed(p0: CameraCaptureSession) {

                    }
                }, handler)
            }

            override fun onDisconnected(p0: CameraDevice) {

            }

            override fun onError(p0: CameraDevice, p1: Int) {

            }
        }, handler)
    }

    fun get_permissions(){
        if(checkSelfPermission(android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED){
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), 101)
        }
    }
    override fun onRequestPermissionsResult(  requestCode: Int, permissions: Array<out String>, grantResults: IntArray  ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if(grantResults[0] != PackageManager.PERMISSION_GRANTED) get_permissions()
    }

    fun returnToMainApp()
    {
        finish()
        /*
        val returnIntent = packageManager.getLaunchIntentForPackage("com.phonelifters.armenu")
        if (returnIntent != null)
        {
            startActivity(returnIntent, null)
        }

         */
    }
}

/*
sources: https://github.com/Pawandeep-prog/realtime_pose_detection_android,
https://github.com/doyaaaaaken/kotlin-csv,
https://www.javatpoint.com/kotlin-android-read-and-write-internal-storage,
https://www.educba.com/kotlin-empty-list/
https://www.quora.com/How-do-I-solve-this-problem-in-Android-studio-Java-lang-IllegalArgumentException-The-size-of-byte-buffer-and-the-shape-do-not-match,
https://www.geeksforgeeks.org/textview-in-android-with-example/,
https://developer.android.com/reference/kotlin/android/widget/TextView,
https://developer.android.com/reference/android/widget/Button,
https://stackoverflow.com/a/31696644,
https://stackoverflow.com/a/31696491,
https://stackoverflow.com/a/76680021,
https://stackoverflow.com/a/4038637
 */
