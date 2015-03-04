package edu.mit.csail.amoeba;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import android.content.res.AssetManager;
import android.hardware.SensorManager;
import android.os.Handler;
import android.util.Log;
import edu.mit.csail.kde.KDE;
import edu.mit.csail.sensors.Accel;
import edu.mit.csail.sensors.AccelFeatureItem;
import edu.mit.csail.sensors.GPS;
import edu.mit.csail.sensors.WiFi;

/**
 * 
 * @author yuhan SensorProcessor processes data from sensors and implements the
 *         fusing algorithm.
 * 
 */

public class SensorProcessor {

	
	ScheduledThreadPoolExecutor algoExec = new ScheduledThreadPoolExecutor(1);
	FusionAlgoTask fusionAlgoTask = new FusionAlgoTask();
	double[] final_confidence = new double[Global.ACTIVITY_NUM];
	

	// TO-DO: Set accel_only value based on user input
	private boolean accel_only = false;

	DecimalFormat df = new DecimalFormat("#.####");
	/**
	 * Finite State machine - state 0: nothing, state 1: accelerometer only, state 2:
	 * accel + wifi, state 3: accel + gps + wifi monitoring
	 * 
	 **/
	private int state = 0;

	public SensorProcessor(int state) {
		this.state = state;

		Global.accelKdeEstimators = new KDE[Global.ACTIVITY_NUM][Global.ACCEL_FEATURE_NUM];
		Global.wifiKdeEstimator = new KDE[Global.ACTIVITY_NUM];
		Global.gpsKdeEstimator = new KDE[Global.ACTIVITY_NUM];
		for (int i = 0; i < Global.ACTIVITY_NUM; ++i) {
			Global.accelKdeEstimators[i][0] = new KDE(Global.ACCEL_FEATURE_KDE_BW);
			Global.accelKdeEstimators[i][1] = new KDE(Global.ACCEL_FEATURE_KDE_BW);
			Global.accelKdeEstimators[i][2] = new KDE(Global.ACCEL_FEATURE_KDE_BW);
			Global.wifiKdeEstimator[i] = new KDE(Global.WIFI_FEATURE_KDE_BW);
			Global.gpsKdeEstimator[i] = new KDE(Global.GPS_FEATURE_KDE_BW);
			
			final_confidence[i] = 1 / (Global.ACTIVITY_NUM * 1.0);
		}

		loadKDEClassifier();
		Accel.init();
		WiFi.init();
		GPS.init();
		Debug.init();
	}

	public void run() {
		
		Accel.start(SensorManager.SENSOR_DELAY_NORMAL);
		WiFi.start();
		algoExec.scheduleAtFixedRate(fusionAlgoTask, Global.LATENCY_IN_SEC * 1000, Global.LATENCY_IN_SEC * 1000, TimeUnit.MILLISECONDS);
		
	}

	public void stop() {
		algoExec.remove(fusionAlgoTask);
		Accel.stop();
		WiFi.stop();
		GPS.stop();
		Debug.stop();
	}

	/**
	 * Load training data points from a file and generate KDE estimators
	 */
	public void loadKDEClassifier() {
		AssetManager am = Global.context.getAssets();
		try {
			InputStream is = am.open(Global.accelTrainingDataFilename);
			BufferedReader r = new BufferedReader(new InputStreamReader(is));
			String line;

			// Read accel data points
			while ((line = r.readLine()) != null) {
				String[] segs = line.split(",");
				int gt = Integer.parseInt(segs[3]);
				double mean = Double.parseDouble(segs[0]);
				double sigma = Double.parseDouble(segs[1]);
				double pf = Double.parseDouble(segs[2]);

				Global.accelKdeEstimators[gt][0].addValue(mean, 1.0);
				Global.accelKdeEstimators[gt][1].addValue(sigma, 1.0);
				Global.accelKdeEstimators[gt][2].addValue(pf, 1.0);
			}
			is.close();
			r.close();

			// Read wifi data points
			is = am.open(Global.wifiTrainingDataFilename);
			r = new BufferedReader(new InputStreamReader(is));
			while ((line = r.readLine()) != null) {
				String[] segs = line.split(",");
				int gt = Integer.parseInt(segs[1]);
				double tanimotoDist = Double.parseDouble(segs[0]);
				Global.wifiKdeEstimator[gt].addValue(tanimotoDist, 1.0);
			}
			is.close();
			r.close();

			// Read gps data points
			is = am.open(Global.gpsTrainingDataFilename);
			r = new BufferedReader(new InputStreamReader(is));
			while ((line = r.readLine()) != null) {
				String[] segs = line.split(",");
				int gt = Integer.parseInt(segs[1]);
				double speed = Double.parseDouble(segs[0]);
				Global.gpsKdeEstimator[gt].addValue(speed, 1.0);
			}
			is.close();
			r.close();

		} catch (IOException e) {
			Log.e(MainActivity.TAG, "Error open training data");
			e.printStackTrace();
		}
	}

	
	public class FusionAlgoTask implements Runnable {
		
		@Override
		public void run() {
			long curTime = System.nanoTime();
			android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_BACKGROUND);
			Thread t = Thread.currentThread();
			System.out.println("fusion algo thread:" + t.getId());
			
			
			long currentTime = System.nanoTime();
			System.out.println("isWiFiworking? " + WiFi.isWorking());
			if (WiFi.isWorking())
				WiFi.scan();

			/** Process accelerometer data **/
			double[][] accel_debug = new double[Global.ACTIVITY_NUM][Global.ACCEL_FEATURE_NUM];
			double[] accel_post = new double[Global.ACTIVITY_NUM];
			double[] accel_features = new double[Global.ACCEL_FEATURE_NUM];
			accel_features = Accel.getAccelFeatures();
			accel_debug = Accel.getAccelDebugInfo();
			accel_post = Accel.getAccelLikelihood();
			int accel_prediction = Global.getPrediction(accel_post);
			System.out.println("Accel pred. - " + accel_prediction);
			Global.adaFeatureString = "";
			for (int i = 0; i < Global.ACCEL_FEATURE_NUM; ++i){
				if (i != Global.ACCEL_FEATURE_NUM-1){
					Global.adaFeatureString += df.format(accel_features[i]) + " ,";
				}else{
					Global.adaFeatureString += df.format(accel_features[i]);
				}
			}
			System.out.println("Accel features:" + Global.adaFeatureString);
			Global.adaAccelConfString = "";
			for (int i = 0; i < Global.ACTIVITY_NUM; ++i) {
				if (i != Global.ACTIVITY_NUM-1){
					Global.adaAccelConfString += df.format(accel_post[i]) + " ,";
				}else{
					Global.adaAccelConfString += df.format(accel_post[i]);	
				}
			}
			System.out.println("Accel post:" + Global.adaAccelConfString);
			
			/** Done with accel processing **/

			/** Posterior prob from WiFi **/
			// If wifi is not working, we get the default invalid feature
			double wifiFeature = WiFi.getFeature();
			double[] wifi_post_bounded = new double[Global.ACTIVITY_NUM];
			double[] wifi_post_unbounded = new double[Global.ACTIVITY_NUM];
			Global.getPostProb(Global.wifiKdeEstimator, wifiFeature, wifi_post_bounded,
					wifi_post_unbounded);

			
			System.out.print("\nWiFi- ");
			for (int i = 0; i < Global.ACTIVITY_NUM; ++i) {
				System.out.print(wifi_post_unbounded[i] + ", ");
			}
			/**
			System.out.println("wifiFeature:" + wifiFeature); 
			for (int i = 0; i < Global.ACTIVITY_NUM; ++i){
				System.out.println("WiFi - Activity:" + i + ": unbounded:" +
						  wifi_post_unbounded[i]); 
			}
			**/
			/** Done with WiFi **/
			
			/** Posterior prob from GPS **/
			double gpsFeature = GPS.getFeature();
			double[] gps_post_bounded = new double[Global.ACTIVITY_NUM];
			double[] gps_post_unbounded = new double[Global.ACTIVITY_NUM];
			Global.getPostProb(Global.gpsKdeEstimator, gpsFeature, gps_post_bounded,
					gps_post_unbounded);
			/** Done with GPS **/
			
			/** Make prediction and adapt sensors **/

			/** Smooth accelerometer data with m_estimator**/
			double[] smoothedAccelProb = new double[Global.ACTIVITY_NUM];
			double smoothWeight = Global.ACTIVITY_NUM;
			for (int i = 0; i < Global.ACTIVITY_NUM; ++i){
				smoothedAccelProb[i] = (smoothWeight * accel_post[i] + 
				1/(Global.ACTIVITY_NUM * 1.0)) / ((smoothWeight + 1) *1.0);  
			}
			
			
			/** Soft voting **/
			double[] combined_post_prob = new double[Global.ACTIVITY_NUM];
			double combined_denominator = 0;
			for (int i = 0; i < Global.ACTIVITY_NUM; ++i) {
				combined_post_prob[i] = smoothedAccelProb[i]
						* wifi_post_unbounded[i] * gps_post_unbounded[i];
				combined_denominator += combined_post_prob[i];
			}
			for (int i = 0; i < Global.ACTIVITY_NUM; ++i) {
				combined_post_prob[i] /= combined_denominator;
			}
			
			System.out.print("\nsoft voting- ");
			for (int i = 0; i < Global.ACTIVITY_NUM; ++i) {
				System.out.print(combined_post_prob[i] + ", ");
			}
			
			/** Finally - EWMA smoothing **/
			double ewma_denominator = 0.0;
			for (int i = 0; i < Global.ACTIVITY_NUM; ++i) {
				final_confidence[i] = (Global.EWMA_ALPHA * combined_post_prob[i])
						+ ((1 - Global.EWMA_ALPHA) * final_confidence[i]);
				ewma_denominator += final_confidence[i];
			}
			for (int i = 0; i < Global.ACTIVITY_NUM; ++i) {
				final_confidence[i] /= ewma_denominator;
			}
		
			System.out.print("\nEWMA- ");
			Global.adaConfString = "";
			for (int i = 0; i < Global.ACTIVITY_NUM; ++i) {
				System.out.print(final_confidence[i] + ", ");
				if (i != Global.ACTIVITY_NUM-1){
					Global.adaConfString +=  df.format(final_confidence[i]) +" ,";
				}else{
					Global.adaConfString +=  df.format(final_confidence[i]);
				}
			}
			int final_prediction = Global.getPrediction(final_confidence);
			Global.AdaPrediction = final_prediction;
			System.out.println("\nprediction: " + final_prediction + " google Pred:" + Global.GooglePrediction);
			System.out.println("total time:" + (System.nanoTime() - curTime) / 1000000.0);
			
			/** Write debug message into file **/
			Debug.logPrediction(currentTime, Global.getGroundTruth(),
					final_prediction, accel_post, wifi_post_bounded,
					gps_post_bounded, combined_post_prob, final_confidence,
					accel_features, wifiFeature, gpsFeature, accel_debug ,state,
					Global.GooglePrediction);

			/** Adapt sensors **/
			if (!accel_only) {
				if (accel_prediction == Global.STATIC
						|| accel_prediction == Global.WALKING
						|| accel_prediction == Global.RUNNING) {
					state = 1;

					// Stop other sensors
					if (WiFi.isWorking()) {
						WiFi.stop();
					}
					if (GPS.isWorking()) {
						GPS.stop();
					}
				} else {

					// Accel is uncertain
					// Turn on GPS when WiFi is low (but still keep sampling WiFi)
					// Turn off GPS when WiFi is high
					if (!WiFi.isWorking()) {
						// WiFi is off, turn it on
						state = 2;
						WiFi.start();
					} else if (WiFi.isDensityHigh()) {
						// turn off GPS
						state = 2;
						if (GPS.isWorking())
							GPS.stop();

					} else { // turn on GPS
						state = 3;
						if (!GPS.isWorking())
							GPS.start(Global.LATENCY_IN_SEC);
					}
				}
			}
			
		}
	};

	
	
}
