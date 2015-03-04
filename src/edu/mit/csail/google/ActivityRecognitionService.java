package edu.mit.csail.google;

import com.google.android.gms.location.ActivityRecognitionResult;

import edu.mit.csail.amoeba.Global;
import android.app.IntentService;
import android.content.Intent;
/**
 * Service that receives ActivityRecognition updates. It receives
 * updates in the background, even if the main Activity is not visible.
 */

public class ActivityRecognitionService extends IntentService {
	
	public ActivityRecognitionService() {
		super("ActivityRecognitionService");
	}

	/**
	 * Google Play Services calls this once it has analyzed the sensor data
	 */
	 /**
     * Called when a new activity detection update is available.
     */
	@Override
	protected void onHandleIntent(Intent intent) {
		System.out.println("Google onHandleIntent called");
		
		if (ActivityRecognitionResult.hasResult(intent)) {
			ActivityRecognitionResult result = ActivityRecognitionResult.extractResult(intent);
			Global.GooglePrediction = result.getMostProbableActivity().getType();
			System.out.println("Google Result:" + Global.getGoogleFriendlyName(result.getMostProbableActivity()
					.getType()) + "\n" + result.toString());
		}
	}

	
}