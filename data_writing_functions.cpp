#include "opencv2/opencv.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <windows.h>

#include <seam4us_functions.h>

void writeResults(double value, int cycle_position) {
	// this should write the data to a text file
}

void convertFeaturesToPeople(int features, int cycle_position) {
	// this should convert the features to people, perhaps per camera
	double people = 0;

	writeResults(people,cycle_position);
}
