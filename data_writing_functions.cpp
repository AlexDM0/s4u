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

void convertFeaturesToPeople(double features, int cycle_position, std::vector<std::vector<float> >& training_coefficients) {
	// this should convert the features to people
	// the formula used is a*x^b + c
	float a = training_coefficients[cycle_position][0];
	float b = training_coefficients[cycle_position][1];
	float c = training_coefficients[cycle_position][2];
	double people = a*pow(features,b) + c;

	writeResults(people,cycle_position);
}
