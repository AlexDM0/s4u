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
#include <template_functions.hpp>
#include <profiling_class.hpp>

void applyErosionParameters(cv::Mat& input_image, cv::Mat& output_image, int& para) {
  // erosion operation, erosion_type: MORPH_RECT = 0; MORPH_CROSS = 1; MORPH_ELLIPSE = 2;
  cv::Mat element = cv::getStructuringElement( 2, cv::Size( 2*para + 1, 2*para+1 ),
	                    cv::Point( para, para ) );
  cv::erode(input_image, output_image, element);

}

void applyDilationParameters(cv::Mat& input_image, cv::Mat& output_image, int para) {
  cv::Mat element = cv::getStructuringElement( 2, cv::Size( 2*para + 1, 2*para+1 ),
		                    cv::Point( para, para ) );
  cv::dilate(input_image, output_image, element);
}

void applyEdgeThreshold(cv::Mat img, cv::Mat& edgemask, int& threshold) {
  cv::Mat img_b, img_g, img_r;
  std::vector<std::vector<cv::Point> > contours;
  std::vector<cv::Vec4i> hierarchy;
  std::vector <cv::Mat> rgbChannels;
  /* edge detection on RGB three channels and merger the edges on the three channels */
  cv::split(img, rgbChannels);
  img_b = rgbChannels[0];
  img_g = rgbChannels[1];
  img_r = rgbChannels[2];
  cv::Canny(img_b, img_b, threshold, threshold*2, 3);
  cv::Canny(img_g, img_g, threshold, threshold*2, 3);
  cv::Canny(img_r, img_r, threshold, threshold*2, 3);
  cv::add(img_b, img_g, img_g);
  cv::add(img_g, img_r, edgemask);

}



void bgfgImage(cv::Mat& resized_frame,
				cv::BackgroundSubtractorMOG2& bg_model,
				std::vector<cv::Mat>& ROI_masks,
				int frame_counter,
				int cycle_position,
				double scale_factor,
				double learning_rate,
				int background_training_frames,
				int processing_frames,
				bool training_only,
				cv::Mat perspective_matrix,
				double& frame_feature,
				double max_perspective_multiplier,
				int training_cycles
				) {
	cv::Mat img, background_image, edgemask, fgmask, fgmask1, fgmask2, fgmask3;
	cv::Mat individual_blob = cv::Mat::zeros( resized_frame.size(), CV_8U);
	cv::Mat combined_blobs  = cv::Mat::zeros( resized_frame.size(), CV_8U);
	int threshold = 10;
	int contour_size = 0;
	int contour_size_threshold = 120;
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	int min_y = 10000;
	int max_y = 0;
	int min_x = 10000;
	int max_x = 0;
	int blob_xy_ratio_threshold = 4;
	int parameter;
	int number_of_contours;
	double perspective_multiplier = 0;
	cv::Point lowest_point;

	resized_frame.copyTo(img);

	// blur the color image
	cv::GaussianBlur(img, img, cv::Size(9,9), 1.5, 1.5);

	// background subtraction -- this trains the background
	bg_model(img, fgmask, learning_rate);

	// the last time this camera background is trained we save it
	if (frame_counter == background_training_frames+processing_frames) {
		bg_model.getBackgroundImage(background_image);
		cv::imwrite(addStr("data/backgrounds/background_",cycle_position,".png"),background_image);
		if (training_only)
			std::cout << "Training Backgrounds: Cycles left: " << training_cycles << " camera pos: " << cycle_position << std::endl;
	}

	// run the processing part of the algorithm after the initial background training for the amount of processing_frames
	if (frame_counter > background_training_frames && frame_counter <= (background_training_frames+processing_frames) && !training_only) {
		cv::threshold(fgmask, fgmask, 0, 255, 0);

		/* Apply dilation operation */
		parameter = 4;
		applyDilationParameters(fgmask, fgmask2, parameter);
		applyEdgeThreshold(img, edgemask, threshold);


		// apply mask on edges and add to fgmask
		cv::bitwise_and(fgmask2, edgemask, fgmask2);
		cv::add(fgmask, fgmask2, fgmask3);
		cv::threshold(fgmask3, fgmask3, 0, 255, 0);

		//cv::imshow("mask3",fgmask3);

		/* Try to connect edges */
		parameter = 2;
		applyDilationParameters(fgmask3, fgmask3, parameter);
		applyErosionParameters(fgmask3, fgmask3, parameter);

		// because we do a dilation, we have to reapply the ROI to the mask
		applyROImaskToFrame(fgmask3,ROI_masks,cycle_position);

		// Find contours in the FGMASK3
		contours.clear();
		hierarchy.clear();
		cv::findContours( fgmask3, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

		// Find the contours and draw them. We have to do this to make sure we do not count blobs double.
		number_of_contours = contours.size();
		combined_blobs = cv::Scalar(0);
		for(int i = 0; i < number_of_contours; i++ ) {
			contour_size = contours[i].size() ;
			if (contour_size > contour_size_threshold) {
				// get lowest point
				min_y = 10000;
				max_y = 0;
				min_x = 100000;
				max_x = 0;
				lowest_point.y = contours[i][0].y;
				for(int j = 1; j < contour_size; j++) {
					// get min max x
					if (min_x > contours[i][j].x)
						min_x = contours[i][j].x;
					if (max_x < contours[i][j].x)
						max_x = contours[i][j].x;
					// get min max y
					if (min_y > contours[i][j].y)
						min_y = contours[i][j].y;
					if (max_y < contours[i][j].y)
						max_y = contours[i][j].y;
				}
				if ((max_x - min_x)/(max_y - min_y) < blob_xy_ratio_threshold)
					cv::drawContours( combined_blobs, contours, i, 255, CV_FILLED, 8, hierarchy, 0, cv::Point() );
			}
		}

		/*Debug code for viewing the process*/
		//bg_model.getBackgroundImage(background_image);
		//cv::imshow("BG", background_image);
		//cv::imshow("FG", fgmask);
		//cv::imshow("edges",edgemask);
		//cv::imshow("img",img);
		//cv::imshow("combined_blobs",combined_blobs);
		//cv::waitKey(1);

		// find the contours again.
		contours.clear();
		hierarchy.clear();
		cv::findContours( combined_blobs, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

		frame_feature = 0;
		number_of_contours = contours.size();
		for(int i = 0; i < number_of_contours; i++ ) {
			// get the lowest point
			lowest_point = contours[i][0];
			contour_size = contours[i].size() ;
			for(int j = 1; j < contour_size; j++) {
				if (contours[i][j].y > lowest_point.y )
					lowest_point = contours[i][j];
			}

			// correct for perspective based on lowest point
			for (int j = lowest_point.y; j < perspective_matrix.rows ;j++){
				perspective_multiplier = perspective_matrix.at<float>(j, lowest_point.x);
				if (perspective_multiplier > 0 )
					break;
			}

			// get the features
			individual_blob = cv::Scalar(0);
			cv::drawContours(individual_blob, contours, i, 1, CV_FILLED, 8, hierarchy, 0, cv::Point() );
			frame_feature += sum(individual_blob)[0] * pow(perspective_multiplier,2.0);

		}
	}

}

bool checkIfBackgroundsExist(int amount_of_cameras) {
	cv::Mat background;
	for (int i = 0; i < amount_of_cameras; i++){
		background = cv::imread(addStr("data/backgrounds/background_",i,".png"));
		if (background.empty())
			return false;
	}
	return true;
}

int initializeBackgrounds(std::vector<cv::BackgroundSubtractorMOG2>& background_model_vector,
							double learning_rate,
							int amount_of_training_cycles,
							int amount_of_training_cycles_from_nothing) {
	bool backgrounds_available = checkIfBackgroundsExist((int)background_model_vector.size());
	// define settings for all models
	for (int i = 0; i < (int)background_model_vector.size(); i++){
		background_model_vector.at(i).set ("varThreshold", 16);
		background_model_vector.at(i).set ("history", 1);
	}

	if (backgrounds_available) {
		cv::Mat img, fgmask;
		for (int i = 0; i < (int)background_model_vector.size(); i++){
			img = cv::imread(addStr("data/backgrounds/background_",i,".png"));
			background_model_vector.at(i)(img, fgmask, learning_rate);
		}
		return amount_of_training_cycles;
	}
	else
		return amount_of_training_cycles_from_nothing;


}


