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
import com.google.android.gms.tasks.Task
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.pose.Pose
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.PoseDetectorOptionsBase
import com.google.mlkit.vision.pose.PoseLandmark
import com.google.mlkit.vision.pose.accurate.AccuratePoseDetectorOptions
//import com.google.mlkit.vision.pose.defaults.PoseDetectorOptions
import com.phonelifters.pose_detection.R
import com.phonelifters.pose_detection.ml.LiteModelMovenetSingleposeLightningTfliteFloat164
import com.phonelifters.pose_detection.ml.MlkitModel2
import com.phonelifters.pose_detection.ml.Model1
import com.phonelifters.pose_detection.ml.Model2
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import okhttp3.internal.wait
//import kotlinx.coroutines.NonCancellable.message

import org.apache.commons.lang3.ObjectUtils.Null
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

        //PoseDetectorOptions.Builder()
        //            .setDetectorMode(PoseDetectorOptions.STREAM_MODE)
        //            .build()

        var options : AccuratePoseDetectorOptions = AccuratePoseDetectorOptions.Builder()
            .setDetectorMode(AccuratePoseDetectorOptions.STREAM_MODE)
            .build()

        val poseDetector = PoseDetection.getClient(options)
        var m: Mutex
        m = Mutex()
        imageProcessor = ImageProcessor.Builder().add(ResizeOp(192, 192, ResizeOp.ResizeMethod.BILINEAR)).build()
        model = LiteModelMovenetSingleposeLightningTfliteFloat164.newInstance(this)
        val classifier1 = MlkitModel2.newInstance(this)
        //val classifier2 = Model2.newInstance(this)
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

            fun getResult(bitmap: Bitmap) = runBlocking {
                launch {
                    m.lock()
                    println("ENTERING LOCK")
                    val image = InputImage.fromBitmap(bitmap, 0)
                    var success = mutableListOf<PoseLandmark>()
                    var input_list = mutableListOf<Float>()
                    val result = poseDetector.process(image).addOnSuccessListener { results ->



                        var landmarks = results.getAllPoseLandmarks()
                        if(landmarks != null){
                            println("GETTING HERE")
                            for(landmark in landmarks){
                                input_list.add(landmark.position3D.x)
                                input_list.add(landmark.position3D.y)
                                input_list.add(landmark.position3D.z)
                            }
                            println(input_list.size)
                            //Log.d("numbers?", results.getPoseLandmark(PoseLandmark.LEFT_SHOULDER)!!.position3D.x.toString()
                            //val outputFeature1 = mutableListOf<Float>(0f)
                            if(input_list.size != 0) {
                                val input1 = TensorBuffer.createDynamic(DataType.FLOAT32)
                                input1.loadArray(input_list.toFloatArray(), intArrayOf(99))
                                val output1 = classifier1.process(input1)
                                val outputFeature1 = output1.outputFeature0AsTensorBuffer.floatArray

                                lateinit var message: Button

                                message = findViewById(R.id.accuracy)

                                println(outputFeature1.get(0))
                                if (outputFeature1.get(0) > 0.7f) {// && outputFeature2.get(0) == 1f) {
                                    message.text = "Correct!"
                                } else {
                                    //Log.d("accuracy", "incorrect")
                                    message.text = "Keep Trying"
                                }

                                message.display
                            }


                            input_list.clear()
                        }

                        //Log.d("success", "ues")
                    }
                        .addOnFailureListener { e ->
                            Log.d("failed", "error in recognizing body")
                        }
                    m.unlock()
                }
            }

            override fun onSurfaceTextureUpdated(p0: SurfaceTexture) {
                //Log.d("apps", packageManager.getInstalledApplications(0).toString())
                bitmap = textureView.bitmap!!

                getResult(bitmap)

                //while(!result.isComplete){}
                //println(input_list.size)
                //Log.d("results", result.toString())

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
                        //a = outputFeature0.get(x+1)*w
                        //b = outputFeature0.get(x)*h

                    }

                    else
                    {
                        //a = 0f
                        //b = 0f
                    }

                    //currentRow.add(a)
                    //currentRow.add(b)

                    /*
                    val a = outputFeature0.get(x+1)*w
                    val b = outputFeature0.get(x)*h
                    currentRow.add(a)
                    currentRow.add(b)
                     */
                    x+=3
                }

                /*if(input_list != null && input_list.size > 0) {
                    val input1 = TensorBuffer.createDynamic(DataType.FLOAT32)
                    input1.loadArray(input_list.toFloatArray(), intArrayOf(99))
                    val output1 = classifier1.process(input1)
                    val outputFeature1 = output1.outputFeature0AsTensorBuffer.floatArray

                    /*val input2 = TensorBuffer.createDynamic(DataType.FLOAT32)
                    input2.loadArray(currentRow.toFloatArray(), intArrayOf(34))
                    val output2 = classifier2.process(input2)
                    val outputFeature2 = output2.outputFeature0AsTensorBuffer.floatArray*/

                    lateinit var message: Button

                    message = findViewById(R.id.accuracy)

                    //message = findViewById(R.id.accuracy_message)
                    //Log.d("output 1 results", outputFeature1.size.toString())
                    //Log.d("output 2 results", outputFeature2.size.toString())

                    //Log.d("output feature 1 value", outputFeature1.get(0).toString())
                    //Log.d("output feature 2 value", outputFeature2.get(0).toString())

                    if (outputFeature1.get(0) == 1f) {// && outputFeature2.get(0) == 1f) {
                        message.text = "Correct!"
                    } else {
                        //Log.d("accuracy", "incorrect")
                        message.text = "Keep Trying"
                    }

                    message.display
                }*/
                lateinit var done: Button
                done = findViewById(R.id.done)
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
https://stackoverflow.com/a/4038637,
https://stackoverflow.com/a/4767832 (investigate later),
https://developers.google.com/ml-kit/vision/pose-detection/android,
https://developers.google.com/ml-kit/vision/pose-detection/android#using-a-bitmap,
https://github.com/googlesamples/mlkit/blob/bc243262d40eba8eeac12556fe756450b692019a/android/vision-quickstart/app/src/main/java/com/google/mlkit/vision/demo/kotlin/posedetector/PoseDetectorProcessor.kt#L55,
https://medium.com/@juancoutomayero/ml-kit-and-pose-detection-e22ae3a241a6 (investigate later),
https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-play-services/kotlinx.coroutines.tasks/await.html,
https://stackoverflow.com/a/45213749,
https://kotlinlang.org/api/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines.sync/-mutex/,
https://medium.com/@android-world/kotlin-mutex-a-comprehensive-guide-a79d0f4f2de7#:~:text=A%20Mutex%20is%20a%20synchronization,task%20at%20a%20given%20time.,
https://medium.com/@diousk507/kotlin-coroutine-await-for-listener-5148c85e001f,
https://kotlinlang.org/docs/coroutines-basics.html,
https://developer.android.com/kotlin/coroutines
 */
