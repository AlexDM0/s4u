#include "opencv2/opencv.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"

#include <algorithm>    // std::min
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <windows.h> //profiling

#include <seam4us_functions.h>
#include <template_functions.hpp>

const int XPOS_ON_BACKGROUND = 23;
const int YPOS_ON_BACKGROUND = 106;

void applyROImaskToFrame(cv::Mat& frame, std::vector<cv::Mat>& ROI_masks, int cycle_position) {
	frame.setTo(0,ROI_masks[cycle_position]);
}

void clickDrawROI(int evt, int x, int y, int flags, void* param){
    std::vector <cv::Point>* draw_vector = (std::vector <cv::Point>*) param;
	if(evt==CV_EVENT_LBUTTONDOWN){
        draw_vector->push_back(cv::Point(x-XPOS_ON_BACKGROUND,y-YPOS_ON_BACKGROUND));
    }
}

bool checkROIForCamera(int camera) {
	if (!checkCSV(addStr("data/roi/roi_data_idx__",camera,".csv"))) {
		std::cout << "File does not exist: " << addStr("data/roi_cam_",camera,".dat") << std::endl;
		return false;
	}
	return true;
}

bool getROIForCamera(int camera,std::vector < std::vector<cv::Point> >& ROI_vector) {
	if (!readCSV(addStr("data/roi/roi_data_idx__",camera,".csv"), ROI_vector))
		return false;
	return true;
}

bool checkROI(int amount_of_cameras) {
	for (int i = 0; i < amount_of_cameras; i++) {
		if (!checkROIForCamera(i))
			return false;
	}
	return true;
}

void saveROIdata(std::vector < std::vector<cv::Point> >& ROI_vector, int camera_idx) {
	std::ofstream myfile;
	std::string filename = addStr("data/roi/roi_data_idx__",camera_idx,".csv");
	myfile.open(filename.c_str());
	for (int i = 0; i < int(ROI_vector.size()); i++) {
		for (int j = 0; j < int(ROI_vector[i].size()); j++) {
			myfile << ROI_vector[i][j].x << "," << ROI_vector[i][j].y << ",";
		}
		myfile << "\n";
	}
	ROI_vector.clear();
	myfile.close();
}

void createROImasks(std::vector<cv::Mat>& ROI_masks, int amount_of_cameras, cv::Mat& frame, double scale_factor) {
	std::vector < std::vector<cv::Point> > ROI_vector;
	bool read_file = false;
	for (int i = 0; i < amount_of_cameras; i++) {
		ROI_vector.clear();
		read_file = getROIForCamera(i,ROI_vector);
		if (read_file) {
			cv::Mat mask(frame.size(),CV_8U,cv::Scalar(255));
			drawPolygons(ROI_vector,mask,cv::Scalar(0));

			if (scale_factor != 1) {
				cv::Mat scaled_mask;
				cv::resize(mask, scaled_mask, cv::Size(scale_factor*mask.cols, scale_factor*mask.rows));
				ROI_masks.push_back(scaled_mask);
			}
			else
				ROI_masks.push_back(mask);
			}
		else
			std::cout << "CANNOT READ ROI FILE!" << std::endl;
	}
}

bool setupROI(std::vector<cv::Mat>& camera_frames,std::vector<cv::Mat>& ROI_masks, int amount_of_cameras) {
	std::vector < std::vector<cv::Point> > ROI_vector;
	std::vector <cv::Point> draw_vector;
	cv::Mat canvas, canvas_clone, setup_background, review_background;

	setup_background = cv::imread("data/system_images/setup_roi_gui.png");
	review_background = cv::imread("data/system_images/review_roi_gui.png");

	int camera_idx = 0;
	int key;
	cv::Scalar color = cv::Scalar(255,0,0);
	float opacity = 0.3;
	bool camera_roi_checked = false;
	bool review_roi = false;
	bool show_message = false;

	while (camera_idx < amount_of_cameras) {
		camera_frames[camera_idx].copyTo(canvas);
		canvas_clone = canvas.clone();

		// check if there is a region of interest for this camera
		if (!camera_roi_checked) {
			review_roi = getROIForCamera(camera_idx,ROI_vector);
			camera_roi_checked = true;
		}

		drawPolygons(ROI_vector,canvas,color);
		cv::addWeighted(canvas_clone,opacity,canvas,1 - opacity, 0, canvas);
		cv::Rect roi( cv::Point( XPOS_ON_BACKGROUND, YPOS_ON_BACKGROUND ), canvas.size() );

		if (review_roi) {
			canvas.copyTo( review_background( roi ) );
			cv::imshow("ROI camera",review_background);
			key = cv::waitKey(1);

			if (key == 32) // space
				review_roi = false;
			else if (key == 13) { // enter
				camera_roi_checked = false;
				saveROIdata(ROI_vector,camera_idx);
				ROI_vector.clear();
				camera_idx += 1;
			}
		}
		else {
			// setup roi
			drawLines(draw_vector,canvas,4,CV_AA,cv::Scalar(255,0,0));
			canvas.copyTo( setup_background( roi ) );

			// show an error message if something is wrong
			if (show_message) {
				cv::putText(setup_background, "Define a Region of Interest!", cv::Point(canvas.cols/2 - 300,canvas.rows/2-50), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0,0,255),2);
				if (ROI_vector.size() > 0)
					show_message = false;
			}
			cv::imshow("ROI camera",setup_background);
			cvSetMouseCallback("ROI camera", clickDrawROI, (void*)&draw_vector);
			key = cv::waitKey(1);

			if (key == 32) { // space
				ROI_vector.push_back(draw_vector);
				draw_vector.clear();
			}
			else if (key == 8) { // backspace
				if (draw_vector.size() > 0)
					draw_vector.pop_back();
				else {
					if (ROI_vector.size() > 0)
						ROI_vector.pop_back();
				}
			}
			else if (key == 13) {  // enter
				if (ROI_vector.size() == 0) {
					if (draw_vector.size() == 0)
						show_message = true;
					else {
						ROI_vector.push_back(draw_vector);
						draw_vector.clear();
					}
				}
				else
					review_roi = true;
			}
		}

		if (key == 27)
			return false;
	}
	return true;
}

bool handleROI(std::vector<cv::Mat>& camera_frames, cv::Mat& frame, int cycle_position, int amount_of_cameras, std::vector<cv::Mat>& ROI_masks) {
	bool all_frames_collected = true;
	int filled_frames = 0;
	for (int i = 0; i < amount_of_cameras; i++) {
		if (camera_frames[i].empty())
			all_frames_collected = false;
		else
			filled_frames += 1;
	}

	if (!all_frames_collected) {
		if (camera_frames[cycle_position].empty()) {
			camera_frames[cycle_position] = frame.clone();
			std::cout << "SETTING UP ROI: collecting frame for camera: " << cycle_position << std::endl;
		}
		cv::putText(frame, addStr("SETTING UP ROI: collecting frames (",filled_frames," / ",amount_of_cameras,")"), cv::Point(200,100), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,255));
	}
	else {
		std::cout<< "Setting up ROI" << std::endl;
		cvDestroyWindow("feed");
		if (!setupROI(camera_frames,ROI_masks,amount_of_cameras)) {
			return false;
		}
	}
	return true;
}
