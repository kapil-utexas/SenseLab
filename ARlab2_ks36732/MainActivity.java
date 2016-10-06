package com.hylio.nikhildixit.datacollection;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.media.MediaScannerConnection;
import android.net.Uri;
import android.os.Environment;
import android.os.Handler;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.Surface;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;

public class MainActivity extends AppCompatActivity implements SensorEventListener {

    private SensorManager mSensorManager;
    private Sensor mAccel;
    private Sensor mGyro;
    private File data_file;
    private static Context context;
    private float[] last_accel_value;
    private float[] last_gyro_value;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Initialize sensor values
        last_accel_value = new float[3];
        last_gyro_value = new float[3];

        // Get write permission
        context = getApplicationContext();
        int REQUEST_WRITE_STORAGE = 112;
        boolean hasPermission = (ContextCompat.checkSelfPermission(context,
                Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED);
        if (!hasPermission) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    REQUEST_WRITE_STORAGE);
        }

        // Open file
        data_file = new File(Environment.getExternalStorageDirectory(), "data_sensing.csv");

        // Setup linear acceleration sensor
        mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        mAccel = mSensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
        mGyro = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        mSensorManager.registerListener(this, mAccel, SensorManager.SENSOR_DELAY_NORMAL);
        mSensorManager.registerListener(this, mGyro, SensorManager.SENSOR_DELAY_NORMAL);

        final Handler handler = new Handler();
        Runnable runnable = new Runnable() {

            @Override
            public void run() {
                try{

                    FileOutputStream fOut = new FileOutputStream(data_file, true);
                    OutputStreamWriter myOutWriter = new OutputStreamWriter(fOut);

                    String output = last_accel_value[0] + "," + last_accel_value[1] + "," + last_accel_value[2] + "," + last_gyro_value[0] + "," + last_gyro_value[1] + "," + last_gyro_value[2] + "\n";

                    myOutWriter.append(output);
                    myOutWriter.flush();
                    myOutWriter.close();

                    fOut.flush();
                    fOut.close();

                }
                catch (Exception e) {
                    // TODO: handle exception
                }
                finally{
                    //also call the same runnable to call it at regular interval
                    handler.postDelayed(this, 100);
                }
            }
        };
        handler.postDelayed(runnable, 1000);
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
    }

    public void onAccuracyChanged(Sensor sensor, int accuracy) {
    }

    public void onSensorChanged(SensorEvent event) {

        if (event.sensor.getType() == Sensor.TYPE_LINEAR_ACCELERATION) {
            //System.out.println("AX: " + event.values[0] + " AY: " + event.values[1] + " AZ: " + event.values[2]);
            last_accel_value[0] = event.values[0];
            last_accel_value[1] = event.values[1];
            last_accel_value[2] = event.values[2];
        } else if (event.sensor.getType() == Sensor.TYPE_GYROSCOPE) {
            //System.out.println("GX: " + event.values[0] + " GY: " + event.values[1] + " GZ: " + event.values[2]);
            last_gyro_value[0] = event.values[0];
            last_gyro_value[1] = event.values[1];
            last_gyro_value[2] = event.values[2];
        }
    }
}
