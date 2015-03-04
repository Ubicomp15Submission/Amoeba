package edu.mit.csail.sensors;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import edu.mit.csail.amoeba.Global;

/**
 * 
 * @author yuhan
 * 
 */
public class Accel {
	private static SensorManager sensorManager;
	private static Sensor accelerometer;
	private static AcclListener accelListerner = new AcclListener();
	private static int _period = -1;
	
	// Maintain accel samples //
	private static long tw_ts = -1;
	private static long pred_ts = -1;
	private static double[] accelMag = new double[300 * Global.ACCEL_TIMEWINDOW_SEC];
	private static int accelIndex = 0;
	private static List<AccelFeatureItem> features = new ArrayList<AccelFeatureItem>();
	
	//
	private static double[] accelFeatures = new double[Global.ACCEL_FEATURE_NUM];
	private static double[][] accel_debug = new double[Global.ACTIVITY_NUM][Global.ACCEL_FEATURE_NUM];
	private static double[] accel_post = new double[Global.ACTIVITY_NUM];
	
	static long prevTimeStamp = 0;
	public static void init() {
		for(int i = 0; i < Global.ACTIVITY_NUM; ++i){
			accel_post[i] = -1.0;
			for(int j = 0; j < Global.ACCEL_FEATURE_NUM; ++j){
				accel_debug[i][j] = -1.0;
				accelFeatures[j] = Global.INVALID_FEATURE;
			}
		}
		sensorManager = (SensorManager) Global.context
				.getSystemService(Context.SENSOR_SERVICE);
		accelerometer = sensorManager.getSensorList(Sensor.TYPE_ACCELEROMETER)
				.get(0);
	}

	public static void start(int period) {
		_period = period;
		sensorManager.registerListener(accelListerner, accelerometer, period);
	}

	public static void changeSampleRate(int period) {
		
		if (_period != period) {
			sensorManager.unregisterListener(accelListerner);
			sensorManager.registerListener(accelListerner, accelerometer,
					period);
			_period = period;
		}
	}

	/**
	 * 
	 * @author yuhan AccelListener is the callback of the Accelerometer sensor.
	 */
	private static class AcclListener implements SensorEventListener {
		
		public void onSensorChanged(SensorEvent event) {
			
			long curTime = System.nanoTime();
			if (tw_ts == -1){
				tw_ts = curTime;
			}
			if (pred_ts == -1){
				pred_ts = curTime;
			}
			//System.out.println("timeDiff (ms):" + (event.timestamp - prevTimeStamp)/1000000);
			prevTimeStamp = event.timestamp;
			float x = event.values[0];
			float y = event.values[1];
			float z = event.values[2];
			double mag = (double) Math.sqrt(x * x + y * y + z * z);
			
			accelMag[accelIndex++] = mag;
			
			if ((curTime - tw_ts) >= (((long)Global.ACCEL_TIMEWINDOW_SEC) * 1000000000)){
			
				// compute features
				int N = accelIndex;
				System.out.println("sampling rate:"+ N/5.0);
				double sum = 0.0;
				double sqSum = 0.0;

				for (int i = 0; i < N; ++i) {
					double magItem = accelMag[i];
					sum += magItem;
					sqSum += magItem * magItem;
				}

				double currentWindowFs = (N / (double) (Global.ACCEL_TIMEWINDOW_SEC)); // sampling
																						// frequency
				double currentWindowFFT[] = new double[((N / 2) + 1)];
				computeDFT(accelMag, currentWindowFFT, N);

				double peakPower = -Double.MAX_VALUE;
				int peakPowerLocation = -1;
				for (int j = 1; j < currentWindowFFT.length; ++j) {
					if (currentWindowFFT[j] > peakPower) {
						peakPower = currentWindowFFT[j];
						peakPowerLocation = j;
					}
				}
				double[] adaptAccelFeatures = new double[Global.ACCEL_FEATURE_NUM];
				double mean = sum / (N * 1.0);
				double std = Math.sqrt(sqSum / (N * 1.0) - (mean * mean));
				double peakFreq = peakPowerLocation / (N * 1.0) * currentWindowFs; // peak
				adaptAccelFeatures[0] = mean;
				adaptAccelFeatures[1] = std;
				adaptAccelFeatures[2] = peakFreq;
				
				double[][] accel_adapt_debug = new double[Global.ACTIVITY_NUM][Global.ACCEL_FEATURE_NUM];
				double[] accel_adapt_post = new double[Global.ACTIVITY_NUM];
				Global.getAccelFeaturePostProb(adaptAccelFeatures, accel_adapt_debug, accel_adapt_post);
				AccelFeatureItem item = new AccelFeatureItem(tw_ts, mean, std, peakFreq);
				features.add(item);
				
				System.out.println("debug accel pred:" + Global.getPrediction(accel_adapt_post));
				System.out.print("\ndebug accel Window - ");
				for(int i = 0; i < adaptAccelFeatures.length; ++i){
					System.out.print(i + ":" + adaptAccelFeatures[i]);
				}
				System.out.print("\ndebug accel post Window - ");
				for (int i = 0; i < Global.ACTIVITY_NUM; ++i) {
					System.out.print(accel_adapt_post[i] + ",");
				}
				System.out.println("");
				// need to ramp
				boolean ramp_up;
				int accel_prediction = Global.getPrediction(accel_adapt_post);
				if (accel_prediction == Global.BIKING || accel_prediction == Global.DRIVING){
					ramp_up = true;
				}else{
					ramp_up = false;
				}
				
				if (ramp_up) {
					System.out.println("ramp up");
					changeSampleRate(SensorManager.SENSOR_DELAY_UI);
				} else {
					System.out.println("ramp down");
					changeSampleRate(SensorManager.SENSOR_DELAY_NORMAL);
				}
				
				
				tw_ts = -1;
				accelIndex = 0;
				
				// time to take the average and update accel belief //
				if ((curTime - pred_ts + (long)(Global.ACCEL_TIMEWINDOW_SEC) * 1000000000) 
						>= ((long)Global.LATENCY_IN_SEC) * 1000000000){
					
					for (int i = 0; i < Global.ACCEL_FEATURE_NUM; ++i) {
						accelFeatures[i] = 0.0;
					}

					// update accel posterior
					for (int i = 0; i < features.size(); ++i){
						accelFeatures[0] += features.get(i).mean;
						accelFeatures[1] += features.get(i).std;
						accelFeatures[2] += features.get(i).peakFreq;	
					}
					for (int i = 0; i < accelFeatures.length; ++i){
						accelFeatures[i] /= (features.size() * 1.0);
					}
					Global.getAccelFeaturePostProb(accelFeatures, accel_debug, accel_post);
					
					pred_ts += ((long)Global.LATENCY_IN_SEC * 1000000000);
					features.clear();
				}
				//
				
			}
			
		}

		public void onAccuracyChanged(Sensor sensor, int accuracy) {
		}
	}
	public static double[] getAccelFeatures(){
		return accelFeatures;
	}
	public static double[] getAccelLikelihood(){
		return accel_post;
	}
	public static double[][] getAccelDebugInfo(){
		return accel_debug;
	}
	/**
	 * Transform the accelList into the frequency domain
	 * 
	 * @param accelList
	 *            : array in the time domain
	 * @param fftOutBuffer
	 *            : output array in the frequency domain
	 * @param len
	 */
	private static void computeDFT(double[] accelList,
			double[] fftOutBuffer, int len) {
		int N = len;
		for (int i = 1; i < (N / 2 + 1); ++i) {
			double realPart = 0;
			double imgPart = 0;
			for (int j = 0; j < N; ++j) {
				realPart += (accelList[j] * Math
						.cos(-(2.0 * Math.PI * i * j) / N));
				imgPart += (accelList[j] * Math
						.sin(-(2.0 * Math.PI * i * j) / N));
			}
			realPart /= N;
			imgPart /= N;
			fftOutBuffer[i] = 2 * Math.sqrt(realPart * realPart + imgPart
					* imgPart);
		}
	}

	/**
	 * Update the sampling rate
	 */
	public static void updateSamplingPeriod(int period) {
		stop();
		sensorManager.registerListener(accelListerner, accelerometer, period);
	}

	
	/**
	 * Stop the Accelerometer sensor
	 */
	public static void stop() {
		sensorManager.unregisterListener(accelListerner);
	}
}
