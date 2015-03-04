package edu.mit.csail.google;

import com.google.android.gms.common.ConnectionResult;
import com.google.android.gms.common.GooglePlayServicesClient;
import com.google.android.gms.location.ActivityRecognitionClient;

import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;

public class ActivityRecognitionScan implements
		GooglePlayServicesClient.ConnectionCallbacks,
		GooglePlayServicesClient.OnConnectionFailedListener {
	private Context context;
	private static final String TAG = "GoogleActivityRecog";
	private static ActivityRecognitionClient mActivityRecognitionClient;
	private static PendingIntent callbackIntent;
	private int latencyInSec;

	public ActivityRecognitionScan(Context context, int latency) {
		this.context = context;
		this.latencyInSec = latency;
	}

	/**
	 * Call this to start a scan - don't forget to stop the scan once it's done.
	 * Note the scan will not start immediately, because it needs to establish a
	 * connection with Google's servers - you'll be notified of this at
	 * onConnected
	 */
	public void startActivityRecognitionScan() {
		/*
         * Instantiate a new activity recognition client. Since the
         * parent Activity implements the connection listener and
         * connection failure listener, the constructor uses "this"
         * to specify the values of those parameters.
         */
		mActivityRecognitionClient = new ActivityRecognitionClient(context,
				this, this);
		mActivityRecognitionClient.connect();
		Log.d(TAG, "startActivityRecognitionScan");
		
	}

	public void stopActivityRecognitionScan() {
		try {
			mActivityRecognitionClient.removeActivityUpdates(callbackIntent);
			Log.d(TAG, "stopActivityRecognitionScan");
			
		} catch (IllegalStateException e) {
			// probably the scan was not set up, we'll ignore
		}
	}

	@Override
	public void onConnectionFailed(ConnectionResult result) {
		Log.d(TAG, "onConnectionFailed:" + result.toString());
	}

	/**
	 * Connection established - start listening now
	 */
	@Override
	public void onConnected(Bundle connectionHint) {
		/*
         * Create the PendingIntent that Location Services uses
         * to send activity recognition updates back to this app.
         */
		Intent intent = new Intent(context, ActivityRecognitionService.class);
		/*
         * Return a PendingIntent that starts the IntentService.
         */
		callbackIntent = PendingIntent.getService(context, 0, intent,
				PendingIntent.FLAG_UPDATE_CURRENT);
		
		mActivityRecognitionClient.requestActivityUpdates(latencyInSec * 1000, callbackIntent);
		Log.d(TAG, "google connected: latency - " +  (latencyInSec * 1000));
	}

	@Override
	public void onDisconnected() {
		mActivityRecognitionClient = null;
	}

}