#include "opencv2/opencv.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <windows.h> //profiling

#include <seam4us_functions.h>
#include <template_functions.hpp>

bool readCSV(std::string filename, int x_ref, int y_ref) {
	std::ifstream datafile(filename.c_str());
	int x = 0;
	int y = 0;
	if (datafile) {
		std::string line;
		while(std::getline(datafile,line)) {
			std::stringstream  lineStream(line);
			std::string        cell;
			std::vector<int> row_content;
			x = 0;
			while(std::getline(lineStream,cell,',')) {
				x += 1;
			}
			y += 1;
		}
		datafile.close();
		if (x == x_ref && y == y_ref)
			return true;
		else
			return false;
	}
	else {
		datafile.close();
		return false;
	}
}

bool readCSV(std::string filename, std::vector< std::vector<float> >& content) {
	std::ifstream datafile(filename.c_str());
	if (datafile) {
		std::string line;
		while(std::getline(datafile,line)) {
			std::stringstream  lineStream(line);
			std::string        cell;
			std::vector<float> row_content;
			while(std::getline(lineStream,cell,',')) {
				row_content.push_back(atof(cell.c_str()));
			}
			content.push_back(row_content);
			row_content.clear();
		}
		datafile.close();
		return true;
	}
	else {
		datafile.close();
		return false;
	}
}

bool readCSV(std::string filename, std::vector< std::vector<cv::Point> >& content) {
	std::ifstream datafile(filename.c_str());
	if (datafile) {
		int counter = 0;
		int sizeof_point = 2;
		std::string line;
		while(std::getline(datafile,line)) {
			std::stringstream  lineStream(line);
			std::string        cell;
			std::vector<cv::Point> row_content;
			std::vector<int> set_content;
			counter = 0;
			cv::Point tmp;
			while(std::getline(lineStream,cell,',')) {
				set_content.push_back(atoi(cell.c_str()));
				counter += 1;
				if (counter == sizeof_point && set_content.size() != 0) {
					tmp.x = set_content[0];
					tmp.y = set_content[1];
					row_content.push_back(tmp);
					set_content.clear();
					counter = 0;
				}
			}
			content.push_back(row_content);
			row_content.clear();
		}
		datafile.close();
		return true;
	}
	else {
		datafile.close();
		return false;
	}
}

bool readCSV(std::string filename, cv::Mat& content) {
	std::ifstream datafile(filename.c_str());
	if (datafile) {
		std::string line;
		int i = 0;
		int j = 0;
		while(std::getline(datafile,line)) {
			std::stringstream  lineStream(line);
			std::string        cell;
			i = 0;
			while(std::getline(lineStream,cell,',')) {
				content.at<float>(j,i) = atof(cell.c_str());
				i += 1;
			}
			j += 1;
		}
		datafile.close();
		return true;
	}
	else {
		datafile.close();
		return false;
	}
}

bool checkCSV(std::string filename) {
	std::ifstream datafile(filename.c_str());
	if (datafile) {
		datafile.close();
		return true;
	}
	else {
		datafile.close();
		return false;
	}
}

