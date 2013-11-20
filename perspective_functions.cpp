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
#include <perspective_class.hpp>

const int XPOS_ON_BACKGROUND = 23;
const int YPOS_ON_BACKGROUND = 106;
const int ICONS_X_POS = 200;
const int ICONS_Y_POS = 50;
const int ICONS_HEIGHT = 55;
const int ICONS_SPACING = 50;

int SELECTED_ICON = 1;
int CUSTOM_LENGTH = 17; // in dm (0,1m) == size of average human
int SELECTED_PERSPECTIVE = -1;

enum modes{SETUP_MODE, MARKERS_MODE, REVIEW_MODE};


void clickDraw(int evt, int x, int y, int flags, void* param){
    std::vector <cv::Point>* draw_vector = (std::vector <cv::Point>*) param;
	if(evt==CV_EVENT_LBUTTONDOWN){
        draw_vector->push_back(cv::Point(x-XPOS_ON_BACKGROUND,y-YPOS_ON_BACKGROUND));
    }
}

void clickSelectPerspective(int evt, int x, int y, int flags, void* param){
	if(evt==CV_EVENT_LBUTTONDOWN){
		std::vector <perspective>* perspectives = (std::vector <perspective>*) param;
		for (int i = 0; i < int(perspectives->size()); i++) {
			if (perspectives->at(i).detectClick(x-XPOS_ON_BACKGROUND,y-YPOS_ON_BACKGROUND)) {
				perspectives->at(i).select();
				SELECTED_PERSPECTIVE = i;
				std::cout  << "I HAVE BEEN CLICKED!" << std::endl;
				break;
			}
		}
	}
}


bool checkPerspectiveForCamera(int camera) {
	if (!checkCSV(addStr("data/perspectives/perspective_data_idx__",camera,".csv"))) {
		std::cout << "File does not exist: " << addStr("data/perspective_cam_",camera,".dat") << std::endl;
		return false;
	}
	return true;
}

bool getPerspectiveForCamera(int camera,cv::Mat& perpective) {
	if (!readCSV(addStr("data/perspectives/perspective_data_idx__",camera,".csv"), perpective))
		return false;
	return true;
}

bool checkPerspectives(int amount_of_cameras) {
	for (int i = 0; i < amount_of_cameras; i++) {
		if (!checkPerspectiveForCamera(i))
			return false;
	}
	return true;
}

void savePerspectiveData(std::vector <perspective>& perspectives,cv::Mat& canvas, int camera_idx) {
	cv::Mat multipliers(canvas.size(), CV_32F, cv::Scalar(0));
	for (int i = 0; i < int(perspectives.size()); i++) {
		perspectives[i].getMultiplierMatrix(multipliers);
	}

	if (perspectives.size() == 0)
		multipliers.setTo(1.0);

	std::ofstream myfile;
	std::string filename = addStr("data/perspectives/perspective_data_idx__",camera_idx,".csv");
	myfile.open(filename.c_str());
	for (int i = 0; i < int(multipliers.rows); i++) {
		for (int j = 0; j < int(multipliers.cols); j++) {
			myfile << multipliers.at<float>(i,j) << ",";
		}
		myfile << "\n";
	}
	myfile.close();

	perspectives.clear();
}

int getSelectedPerspective(std::vector<perspective>& perspectives) {
	for (int i = 0; i < int(perspectives.size()); i++) {
		if (perspectives.at(i).selected == 1) {
			return i;
			break;
		}
	}
	return -1;
}

void deselectAllPerspectives(std::vector<perspective>& perspectives) {
	for (int i = 0; i < int(perspectives.size()); i++) {
		perspectives.at(i).deselect();
	}
}

bool setupPerspectives(std::vector<std::vector<cv::Mat> >& camera_clips, int amount_of_cameras, int frames_per_clip) {
	int camera_idx = 0;
	int key = 0;
	double opacity = 0.75;
	int mode = SETUP_MODE;

	std::vector <perspective> perspectives;
	std::vector <cv::Point> draw_vector;
	cv::Mat canvas, canvas_clone, drawing;

	cv::Mat existing_perspective(camera_clips[0][0].size(),CV_32F);

	std::string message = "<- message here ->";

	// backgrounds
	cv::Mat setup_background, markers_background, review_background, review_existing_background;
	setup_background 	= cv::imread("data/system_images/perspective_setup.png");
	markers_background 	= cv::imread("data/system_images/perspective_markers.png");
	review_background 	= cv::imread("data/system_images/perspective_review.png");
	review_existing_background= cv::imread("data/system_images/perspective_review_existing.png");

	bool play = true;
	bool show_message = false;
	bool show_notification = false;
	bool check_if_exists = false;
	bool perspective_exists = false;

	int frame_idx = 0;
	std::cout << "setting up the perspectives" << std::endl;
	while (camera_idx < amount_of_cameras) {


		camera_clips[camera_idx][frame_idx].copyTo(canvas);
		canvas_clone = canvas.clone();
		cv::Rect position_on_background( cv::Point( XPOS_ON_BACKGROUND, YPOS_ON_BACKGROUND ), canvas.size() );

		if (!check_if_exists) {
			perspective_exists = checkPerspectiveForCamera(camera_idx);

			if (perspective_exists) {
				existing_perspective.setTo(0);
				//std::cout << "here" << std::endl;
				getPerspectiveForCamera(camera_idx,existing_perspective);
				//std::cout << existing_perspective << std::endl;
				mode = REVIEW_MODE;
			}
			check_if_exists = true;
		}

		SELECTED_PERSPECTIVE = getSelectedPerspective(perspectives);
		if (play) {
			frame_idx += 1;
			if (frame_idx%frames_per_clip == 0)
				frame_idx = 0;
		}

		// warning message
		if (show_message)
			cv::putText(canvas, message, cv::Point(canvas.cols/2 - 300,30), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(30,30,255),2);
		if (show_notification)
			cv::putText(canvas, message, cv::Point(canvas.cols/2 - 300,30), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(255,255,255),2);

		switch(mode) {
			case SETUP_MODE: // setup
				for (int i = 0; i < int(perspectives.size()); i++) {
					perspectives[i].drawGradient(canvas_clone);
				}
				cv::addWeighted(canvas_clone,0.5,canvas,0.5, 0, canvas);
				drawLines(draw_vector,canvas,4,CV_AA,cv::Scalar(255,0,0));

				canvas.copyTo( setup_background( position_on_background ) );
				cv::imshow("perspective_clip", setup_background);
				cvSetMouseCallback("perspective_clip", clickDraw, (void*)&draw_vector);
				break;
			case MARKERS_MODE: // set two distances
				// store the icon data in the perspective class
				perspectives[SELECTED_PERSPECTIVE].setLength(CUSTOM_LENGTH);

				// draw the perspective
				perspectives[SELECTED_PERSPECTIVE].drawFilled(canvas);
				cv::addWeighted(canvas_clone,opacity,canvas,1 - opacity, 0, canvas);

				// make sure the draw vector is in the correct format
				if (int(draw_vector.size()) == 2) {
					perspectives[SELECTED_PERSPECTIVE].markers.push_back(draw_vector);
					draw_vector.clear();
				}

				if (int(perspectives[SELECTED_PERSPECTIVE].markers.size()) != 2) {
					show_notification = true;
					message = "Two markers are required.";
				}
				else {
					//draw_vector.clear();
					show_notification = false;
				}


				// draw the markers (completed measurement lines)
				perspectives[SELECTED_PERSPECTIVE].drawMarkers(canvas);

				drawLines(draw_vector,canvas,4,CV_AA,cv::Scalar(255,0,0));

				canvas.copyTo( markers_background( position_on_background ) );
				cv::imshow("perspective_clip", markers_background);
				cv::createTrackbar("L=0.1m x ","perspective_clip",&CUSTOM_LENGTH,100);
				cvSetMouseCallback("perspective_clip", clickDraw, (void*)&draw_vector);
				break;
			case REVIEW_MODE: // review
				if (perspective_exists) {
					//std::cout << "drawing" << std::endl;
					perspective tmp;
					int xmin,xmax,ymin,ymax;
					tmp.getMinMax(existing_perspective,xmin,xmax,ymin,ymax);
					tmp.drawGradient(canvas,existing_perspective,xmin,xmax,ymin,ymax);
				}

				if (SELECTED_PERSPECTIVE != -1) {
					draw_vector.swap(perspectives[SELECTED_PERSPECTIVE].points);
					mode = SETUP_MODE;
				}

				for (int i = 0; i < int(perspectives.size()); i++) {
					perspectives[i].drawGradient(canvas);
				}

				if (perspective_exists) {
					canvas.copyTo( review_existing_background( position_on_background ) );
					cv::imshow("perspective_clip", review_existing_background);
				}
				else {
					canvas.copyTo( review_background( position_on_background ) );
					cv::imshow("perspective_clip", review_background);
					cvSetMouseCallback("perspective_clip", clickSelectPerspective, (void*)&perspectives);
				}
				break;
		}

		key = cv::waitKey(1);

		if (key == 32) { 		// space
			if (mode != REVIEW_MODE)
				play = !play;
			else{
				if (perspective_exists)
					perspective_exists = false;
				mode = SETUP_MODE;
				SELECTED_PERSPECTIVE = -1;
			}
		}
		else if (key == 8){		// delete
			if (mode == SETUP_MODE) {
				if (draw_vector.size() > 0)
					draw_vector.pop_back();
				else {
					if (SELECTED_PERSPECTIVE != -1 && perspectives.size() > 0) {
						perspectives.erase(perspectives.begin() + SELECTED_PERSPECTIVE);
						mode = REVIEW_MODE;
					}
					else if (SELECTED_PERSPECTIVE == -1 && perspectives.size() > 0) {
						mode = REVIEW_MODE;
					}
				}
			}
			if (mode == MARKERS_MODE) {
				if (perspectives[SELECTED_PERSPECTIVE].markers.size() > 0) {
					perspectives[SELECTED_PERSPECTIVE].markers.pop_back();
				}
			}
		}
		else if (key == 2424832){		// left arrow
			play = false;
			frame_idx -= 1;
			if (frame_idx < 0)
				frame_idx = frames_per_clip-1;
		}
		else if (key == 2555904){		// right arrow
			play = false;
			frame_idx += 1;
			if (frame_idx%frames_per_clip == 0)
				frame_idx = 0;
		}
		else if (key == 13) {  	// enter
			switch(mode) {
				case SETUP_MODE: // setup
					if (draw_vector.size() < 3) {
						if (draw_vector.size() == 0) {
							if (SELECTED_PERSPECTIVE != -1)
								perspectives.erase(perspectives.begin() + SELECTED_PERSPECTIVE);
							mode = REVIEW_MODE;
						}
						else {
							show_message = true;
							message = "Define an area as perspective";
						}
					}
					else {
						if (SELECTED_PERSPECTIVE == -1) {
							perspectives.push_back(perspective(draw_vector));
							perspectives.end()->select();
							SELECTED_PERSPECTIVE = getSelectedPerspective(perspectives);
						}
						else {
							draw_vector.swap(perspectives[SELECTED_PERSPECTIVE].points);
						}
						draw_vector.clear();
						mode = MARKERS_MODE;
						cvDestroyWindow("perspective_clip");
					}
					break;
				case MARKERS_MODE: // set two distances
					if (int(perspectives[SELECTED_PERSPECTIVE].markers.size()) == 2) {
						perspectives[SELECTED_PERSPECTIVE].calculatePerspective(canvas);
						mode = REVIEW_MODE;
						deselectAllPerspectives(perspectives);
						SELECTED_PERSPECTIVE = -1;
						cvDestroyWindow("perspective_clip");
					}
					break;
				case REVIEW_MODE: // review
					if (!perspective_exists)
						savePerspectiveData(perspectives,canvas, camera_idx);
					camera_idx += 1;
					show_message = false;
					show_notification = false;
					mode = SETUP_MODE;
					check_if_exists = false;
					break;
			}

		}
		else if (key == 27) { 	//esc
			std::cout << "PRESSED ESCAPE" << std::endl;
			return false;
		}
	}

	return true;
}


bool handlePerspectives(
						std::vector<std::vector<cv::Mat> >& camera_clips,
						cv::Mat& frame,
						int cycle_position,
						int amount_of_cameras,
						int& save_frame
						) {
	bool all_clips_collected = true;
	int frames_per_clip = 6;
	int frames_to_skip = 20;

	for (int i = 0; i < amount_of_cameras; i++) {
		if (int(camera_clips[i].size()) < frames_per_clip)
			all_clips_collected = false;
	}

	if (!all_clips_collected) {
		if (int(camera_clips[cycle_position].size()) < frames_per_clip && save_frame == 0) {
			camera_clips[cycle_position].push_back(frame.clone());
			save_frame = frames_to_skip;
			cv::putText(frame, addStr("SETTING UP Perspectives: collecting frames (",camera_clips[cycle_position].size()," / ",frames_per_clip,")"), cv::Point(150,30), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,255));
			cv::imshow("Frame_saved",frame);
			cv::waitKey(1);
		}
	}
	else {
		cvDestroyWindow("feed");
		cvDestroyWindow("Frame_saved");
		if (!setupPerspectives(camera_clips,amount_of_cameras,frames_per_clip)) {
			return false;
		}
	}
	save_frame -= 1;
	return true;
}

bool handlePerspectivesDebug(
						std::vector<std::vector<cv::Mat> >& camera_clips,
						cv::Mat& frame,
						int cycle_position,
						int amount_of_cameras,
						int& save_frame
						) {

	bool all_clips_collected = true;
	int frames_per_clip = 25;
	int frames_to_skip = 1;

	if (int(camera_clips[0].size()) < frames_per_clip)
		all_clips_collected = false;

	if (!all_clips_collected) {
		if (int(camera_clips[0].size()) < frames_per_clip && save_frame == 0) {
			camera_clips[0].push_back(frame.clone());
			save_frame = frames_to_skip;
		}

		cv::putText(frame, addStr("SETTING UP Perspectives: collecting frames (",camera_clips[cycle_position].size()," / ",frames_per_clip,")"), cv::Point(200,100), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,255));
	}
	else {
		cvDestroyWindow("feed");
		if (!setupPerspectives(camera_clips,amount_of_cameras,frames_per_clip)) {
			return false;
		}
	}
	save_frame -= 1;
	return true;
}


void loadPerspectiveMatrices(std::vector<cv::Mat>& perspective_matrices, int amount_of_cameras, cv::Mat& frame, double scale_factor) {
	for (int i = 0; i < amount_of_cameras; i++) {
		cv::Mat multipliers(frame.size(), CV_32F, cv::Scalar(0));
		getPerspectiveForCamera(i, multipliers);

		if (scale_factor != 1) {
			cv::Mat scaled_matrix;
			cv::resize(multipliers, scaled_matrix, cv::Size(scale_factor*multipliers.cols, scale_factor*multipliers.rows));
			perspective_matrices.push_back(scaled_matrix);
		}
		else
			perspective_matrices.push_back(multipliers);

		std::cout << ".";
	}
}
