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

bool checkTrainingStatus(int amount_of_cameras, int amount_of_training_coefficients) {
	return readCSV("data/machine_learning/system_trained_coefficients/coefficients.txt",amount_of_training_coefficients,amount_of_cameras);
}

bool getTrainedCoefficients(std::vector<std::vector<float> >& coefficients) {
	return readCSV("data/machine_learning/system_trained_coefficients/coefficients.txt",coefficients);
}

void serializeImage(cv::Mat& data, cv::Mat& image, int columns, int rows) {
	// check the dimensions
	if (rows != image.rows && columns != image.cols) {
		throw("The image dimensions do not match.");
	}

	// serialize the data
	for (int k = 0; k < columns; k++) {
		for (int l = 0; l < rows; l++) {
			data.at<float>(0,l + k*rows) = float(image.at<uchar>(l, k));
		}
	}
}

void serializeImage(cv::Mat& data, cv::Mat& image, int columns, int rows, int offset) {
	// check the dimensions
	if (rows != image.rows && columns != image.cols) {
		throw("The image dimensions do not match.");
	}

	// serialize the data
	for (int k = 0; k < columns; k++) {
		for (int l = 0; l < rows; l++) {
			data.at<float>(offset,l + k*rows) = float(image.at<uchar>(l, k));
		}
	}
}

void testOCR(cv::KNearest& OCR) {
	// test by classification
	int image_rows = 30;
	int image_columns = 17;

	int match = 0;
	int fail  = 0;
	std::stringstream filename;

	cv::Mat sample_loader;
	cv::Mat test_data(1,image_rows*image_columns,CV_32FC1);
	for (int i = 0; i <= 10; i++) {
		//std::cout << "test number: " << i << std::endl;
		for (int j = 0; j < 50; j++) {
			// get the filename
			filename.str("");
			filename << "data/machine_learning/ocr_test_set/" << i << "/00";
			if (j < 9)
				filename << "0";
			filename << j+1 << ".jpg";

			// load the image
			sample_loader = cv::imread(filename.str(), 0);
			if(!sample_loader.data ){
				std::cout <<  "Could not open or find the image: " << filename.str() << std::endl ;
			}

			// serialize
			serializeImage(test_data,sample_loader,image_columns,image_rows);

			// match
			int found = OCR.find_nearest(test_data,1);

			// check results
			if (found == i)
				match += 1;
			else
				fail += 1;
		}
	}
	// report status
	std::cout << "Out of 550, Match: " << match << "\t Fail: " << fail << std::endl;
}

bool trainOCR(cv::KNearest& OCR) {
	int image_rows = 30;
	int image_columns = 17;
	int categories = 11;
	int samples_per_category = 75;

	//initialize the matrices
	cv::Mat sample_loader;
	cv::Mat training_response(categories*samples_per_category,1, CV_32FC1);
	cv::Mat training_data	 (categories*samples_per_category,image_rows*image_columns,CV_32FC1);
	std::stringstream filename;
	int training_index = 0;
	for (int i = 0; i <= 10; i++) {
		for (int j = 0; j < 75; j++) {
			// get the filename
			filename.str("");
			filename << "data/machine_learning/ocr_training_set/" << i << "/00";
			if (j < 9)
				filename << "0";
			filename << j+1 << ".jpg";

			// open the file
			sample_loader = cv::imread(filename.str(), 0);

			if (image_rows != sample_loader.rows && image_columns != sample_loader.cols) {
				throw("The image dimensions do not match.");
			}

			// check if file is successfully opened
			if(!sample_loader.data ){
				std::cout <<  "Could not open or find the image: " << filename.str() << std::endl;
				return false;
			}

			// add the response to the list of responses
			training_response.at<float>(training_index,0) = float(i);

			// serialize
			serializeImage(training_data,sample_loader,image_columns,image_rows,training_index);

			training_index += 1;
		}
	}
	OCR.train(training_data, training_response);
	return true;
}
