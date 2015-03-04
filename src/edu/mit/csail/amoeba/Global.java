package edu.mit.csail.amoeba;

import com.google.android.gms.location.DetectedActivity;

import edu.mit.csail.kde.KDE;
import android.R.string;
import android.content.Context;

public class Global {
	public static final int LATENCY_IN_SEC = 15;
	/** Constants */
	public static final int ACTIVITY_NUM = 5;
	public static final int ACCEL_FEATURE_NUM = 3;
	public static final String accelTrainingDataFilename = "accel_train";
	public static final String wifiTrainingDataFilename = "wifi_15sec_train";
	public static final String gpsTrainingDataFilename = "gps_train";
	
	public static final int ACCEL_TIMEWINDOW_SEC = 5; //SEC
	public static final double ACCEL_FEATURE_KDE_BW = 0.1;
	public static final double WIFI_FEATURE_KDE_BW = 0.3;
	public static final double GPS_FEATURE_KDE_BW = 0.5;
	public static final int STATIC = 0;
	public static final int WALKING = 1;
	public static final int RUNNING = 2;
	public static final int BIKING = 3;
	public static final int DRIVING = 4;
	public static final int UNKNOWN = -1;
	
	public static final int LOOKBACK_NUM = 2;
	public static final int INVALID_FEATURE = -1;
	public static long startTime = 0;
	public static Context context;
	public static final double EWMA_ALPHA = 0.2; // Current weight
	
	public static int GooglePrediction = -1;
	public static int AdaPrediction = -1;
	public static String adaFeatureString = "";
	public static String adaConfString = "";
	public static String adaAccelConfString = ""; 
	public static int gt = 0;
	
	public static final int UPDATE_UI_MSG = 2;
	public static final int MSG_REGISTER_CLIENT = 0;
	public static final int MSG_UNREGISTER_CLIENT = 1;
	public static int NUMBER_OF_CORES =
	            Runtime.getRuntime().availableProcessors();
	
	/** Store kde estimators (one for each activity) */
	public static KDE[][] accelKdeEstimators;
	public static KDE[] wifiKdeEstimator;
	public static KDE[] gpsKdeEstimator;
	
	
	public static void setContext(Context ctx){
		context = ctx;
	}
	
	public static void setGroundTruth(String activity){
		if (activity.equals("Static")){
			gt = STATIC;
		}else if(activity.equals("Walking")){
			gt = WALKING;
		}else if(activity.equals("Running")){
			gt = RUNNING;
		}else if(activity.equals("Biking")){
			gt = BIKING;
		}else if(activity.equals("Driving")){
			gt = DRIVING;
		}else{
			gt = UNKNOWN;
		}
	}
	
	public static String getAdaFriendlyGroundTruth(int gt){
	
		switch (gt){
		case STATIC:
			return "Still";
		case WALKING:
			return "Walking";
		case RUNNING:
			return "Running";
		case BIKING:
			return "Biking";
		case DRIVING:
			return "Driving";
		default:
			return "Unknown";
		}
	}
	
	public static String getGoogleFriendlyName(int detected_activity_type) {
		switch (detected_activity_type) {
		case DetectedActivity.IN_VEHICLE:
			return "Vehicle";
		case DetectedActivity.ON_BICYCLE:
			return "Biking";
		case DetectedActivity.ON_FOOT:
			return "Foot";
		case DetectedActivity.TILTING:
			return "Tilting";
		case DetectedActivity.STILL:
			return "Still";
		default:
			return "Unknown";
		}
	}
	
	/**
	 * KDE evaluation for single feature sensors
	 * 
	 * @param kdeEstimator
	 * @param featureValue
	 * @param post_bounded
	 * @param post_unbounded
	 */
	public static void getPostProb(KDE[] kdeEstimator, double featureValue,
			double[] post_bounded, double[] post_unbounded) {
		if (featureValue == Global.INVALID_FEATURE) {
			for (int i = 0; i < Global.ACTIVITY_NUM; ++i) {
				post_bounded[i] = 1 / (Global.ACTIVITY_NUM * 1.0);
				post_unbounded[i] = 1 / (Global.ACTIVITY_NUM * 1.0);

			}
			return;
		}
		double unbounded_denominator = 0;
		double bounded_denominator = 0;

		for (int i = 0; i < Global.ACTIVITY_NUM; ++i) {
			post_bounded[i] = kdeEstimator[i].evaluate_unbounded(featureValue);
			post_unbounded[i] = kdeEstimator[i].evaluate_renorm(featureValue);

			unbounded_denominator += post_unbounded[i];
			bounded_denominator += post_bounded[i];
		}

		// Normalize
		for (int i = 0; i < Global.ACTIVITY_NUM; ++i) {
			post_bounded[i] /= bounded_denominator;
			post_unbounded[i] /= unbounded_denominator;

		}
	}
	
	public static void getAccelFeaturePostProb(double[] accelFeatures,
			double[][] accel_debug_unbounded,
			double[] accel_post_unbounded) {
		
		boolean isValid = true;

		for (int i = 0; i < accelFeatures.length; ++i) {
			if (accelFeatures[i] == Global.INVALID_FEATURE) {
				isValid = false;
				break;
			}
		}
		if (!isValid) {
			for (int i = 0; i < Global.ACTIVITY_NUM; ++i) {
				for (int j = 0; j < Global.ACCEL_FEATURE_NUM; ++j) {
					accel_debug_unbounded[i][j] = 1 / (Global.ACTIVITY_NUM * 1.0);
				}
				accel_post_unbounded[i] = 1 / (Global.ACTIVITY_NUM * 1.0);
			}
			return;
		}
		
		double unbounded_denominator = 0;
		for (int i = 0; i < Global.ACTIVITY_NUM; ++i) {
			accel_post_unbounded[i] = 1.0;
			for (int j = 0; j < Global.ACCEL_FEATURE_NUM; ++j) {
				accel_post_unbounded[i] *= Global.accelKdeEstimators[i][j]
						.evaluate_python_unbounded(accelFeatures[j]);
				accel_debug_unbounded[i][j] = Global.accelKdeEstimators[i][j]
						.evaluate_python_unbounded((accelFeatures[j]));
			}
			unbounded_denominator += accel_post_unbounded[i];
		}
		// Normalize
		for (int i = 0; i < Global.ACTIVITY_NUM; ++i) {
			accel_post_unbounded[i] /= unbounded_denominator;
		}

	}
	/**
	 * Get the maximum likelihood prediction
	 * 
	 * @param postProb
	 * @return prediction
	 */
	public static int getPrediction(double[] postProb) {
		int prediction = 0;
		double max = postProb[0];
		for (int i = 1; i < postProb.length; ++i) {
			if (postProb[i] > max) {
				max = postProb[i];
				prediction = i;
			}
		}
		return prediction;
	}

	
	
	public static int getGroundTruth(){
		return gt;
	}
}
