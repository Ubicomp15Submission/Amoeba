package edu.mit.csail.sensors;

public class AccelItem{
	public long time;
	public double mag;
	
	public AccelItem(){
	}

	public AccelItem(long time, double mag) {
		this.time = time;
		this.mag = mag;
	}
}