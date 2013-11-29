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
#include <ctime>

#include <seam4us_functions.h>
#include <template_functions.hpp>

void writeResults(double value, double std, int cycle_position) {
	// this should write the data to a text file
	time_t t = time(0);   // get time now
	struct tm * now = localtime( & t );
/*
	std::cout << "Category: \t\tNumber of People" << std::endl;
	std::cout << "Date: \t\t\t" << (now->tm_year + 1900) << '-' << (now->tm_mon + 1) << '-' <<  now->tm_mday << std::endl;
	std::cout << "Number of People: \t" << value <<  std::endl;
	std::cout << "Standard Dev.: \t\t" << std <<  std::endl;
	std::cout << "Cycle Pos: \t\t" << cycle_position <<  std::endl;
	std::cout << "Time: \t\t\t" << (now->tm_hour < 10 ? addStr("0",now->tm_hour) : toStr(now->tm_hour)) << ':' << (now->tm_min < 10 ? addStr("0",now->tm_min) : toStr(now->tm_min)) << ':' <<  (now->tm_sec < 10 ? addStr("0",now->tm_sec) : toStr(now->tm_sec)) << std::endl;
*/
	std::ofstream datafile;
	datafile.open("output_data.txt");
	datafile << "Category: \t\tNumber of People" << "\n";
	datafile << "Date: \t\t\t" << (now->tm_year + 1900) << '-' << (now->tm_mon + 1) << '-' <<  now->tm_mday << "\n";
	datafile << "Number of People: \t" << value  << "\n";
	datafile << "Standard Dev.: \t\t" << std << "\n";
	datafile << "Cycle Pos: \t\t" << cycle_position << "\n";
	datafile << "Time: \t\t\t" << (now->tm_hour < 10 ? addStr("0",now->tm_hour) : toStr(now->tm_hour)) << ':' << (now->tm_min < 10 ? addStr("0",now->tm_min) : toStr(now->tm_min)) << ':' <<  (now->tm_sec < 10 ? addStr("0",now->tm_sec) : toStr(now->tm_sec));
	datafile.close();

}

void convertFeaturesToPeople(double features, int cycle_position, std::vector<std::vector<float> >& training_coefficients) {
	// this should convert the features to people
	// the formula used for the features is a*x^b where x is features
	// the forumla used for the std is features is [(c*x^d + e) - number_of_people] where x is number of people
	// the format of the csv is a,b,c,d,e
	float a = training_coefficients[cycle_position][0];
	float b = training_coefficients[cycle_position][1];
	float c = training_coefficients[cycle_position][2];
	float d = training_coefficients[cycle_position][3];
	float e = training_coefficients[cycle_position][4];
	double people = round(a*pow(features,b));
	double std = round(c*pow(people,d) + e) - people;

	writeResults(people,std,cycle_position);
}
