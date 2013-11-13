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
			filename << "data/ocr_test_set/" << i << "/00";
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
			filename << "data/ocr_training_set/" << i << "/00";
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


int identifyNumber(cv::KNearest& OCR, cv::Mat& number) {
	int image_rows = 30;
	int image_columns = 17;

	cv::Mat image_data(1,image_rows*image_columns,CV_32FC1);
	serializeImage(image_data,number,image_columns,image_rows);
	return  OCR.find_nearest(image_data,1);
}

int identifyCamera(cv::KNearest& OCR, cv::Mat& number1, cv::Mat& number2, cv::Mat& number3) {
	// if the returned number is 10, the image has been identified as Not a Number
	int first_number,second_number,third_number;
	first_number = identifyNumber(OCR,number1);
	if (first_number == 10)
		return -1;

	second_number = identifyNumber(OCR,number2);
	if (second_number == 10)
		return -1;

	third_number = identifyNumber(OCR,number3);
	if (third_number == 10)
		return (700 + 10* first_number + second_number);
	else
		return (7000 + 100* first_number + 10*second_number + third_number);
}

int getPositionFromOCR(
						std::vector<int>& camera_cycle,
						cv::KNearest& OCR,
						cv::Mat& number1,
						cv::Mat& number2,
						cv::Mat& number3,
						int cycle_position
						) {
	int camera_id = identifyCamera(OCR, number1, number2, number3);
	int amount_of_cameras = camera_cycle.size();
	int expected_id = camera_cycle[cycle_position%amount_of_cameras];

	// the third number slot can be empty. If we expect it to be, we ignore it to avoid false positives.
	if (expected_id < 1000 && camera_id > 1000) {
		camera_id = int(floor(double(camera_id)/10.0));
	}

	// if the camera id has been read successfully
	if (camera_id != -1) {
		for (int i = 0; i < amount_of_cameras; i++) {
			if (camera_id == camera_cycle[i]) {
				cycle_position = i;
				break;
			}
		}
		return cycle_position;
	}
	else
		return -1;

}

bool determineCyclePosition(
							cv::Mat& gray,
							std::vector<int>& camera_cycle,
							cv::KNearest& OCR,
							int& prev_found_position,
							int& cycle_position,
							int& successful_ocr,
							int& failed_ocr,
							int& number_of_deviations
							) {
	cv::Mat subset, number1, number2, number3;
	int found_position;

	// extract the 3 ID numbers
	subset = gray(cv::Rect(125,35,17,30));
	subset.copyTo(number1);
	subset = gray(cv::Rect(142,35,17,30));
	subset.copyTo(number2);
	subset = gray(cv::Rect(159,35,17,30));
	subset.copyTo(number3);

	// get location in the camera cycle
	found_position = getPositionFromOCR(
										camera_cycle,
										OCR,
										number1,
										number2,
										number3,
										cycle_position
										);
	// after the first ID, reference with previous ID
	if (found_position > 0) {
		successful_ocr += 1;
		if (successful_ocr > 1) {
			if (found_position == prev_found_position) {
				// if the OCR is not the predicted camera, ignore the OCR twice, if this happens again, accept OCR
				if (found_position != cycle_position) {
					if (number_of_deviations >= 2){
						// We accept the identification.
						cycle_position = found_position;
						number_of_deviations = 0;
					}
					else {
						// We use the expectation
						number_of_deviations += 1;
					}
				}
				else {
					// We accept the identification.
					// Since it is the same as the expected position, we don't need to set cycle_position
					number_of_deviations = 0;
				}
				return true;
			}
			else
				failed_ocr += 1;
		}
		prev_found_position = found_position;
	}
	else {
		failed_ocr += 1;
	}

	if (failed_ocr == 3) {
		// the OCR was not successful. We assume this camera is next in the sequence.
//		std::cout << "Bad OCR. Maybe not a number in selection. Used expectation" << std::endl;
		return true;
	}
	return false;
}
