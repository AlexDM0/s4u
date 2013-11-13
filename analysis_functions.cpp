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

void applyErosionParameters(cv::Mat& input_image, cv::Mat& output_image, int& para) {
  // erosion operation, erosion_type: MORPH_RECT = 0; MORPH_CROSS = 1; MORPH_ELLIPSE = 2;
  cv::Mat element = cv::getStructuringElement( 2, cv::Size( 2*para + 1, 2*para+1 ),
	                    cv::Point( para, para ) );
  cv::erode(input_image, output_image, element);

}

void applyDilationParameters(cv::Mat& input_image, cv::Mat& output_image, int& para) {
  cv::Mat element = cv::getStructuringElement( 2, cv::Size( 2*para + 1, 2*para+1 ),
		                    cv::Point( para, para ) );
  cv::dilate(input_image, output_image, element);
}

void applyEdgeThreshold(cv::Mat img, cv::Mat& edgemask, int& threshold) {
  cv::RNG rng(12345);
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

  /// Find contours
//   cv::findContours( edgemask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0,0));

  /// Draw contours
//  cv::Mat drawing = cv::Mat::zeros( edgemask.size(), CV_8UC3 );
//  for( int i = 0; i< contours.size(); i++ )
//     {
//       cv::Scalar color = cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
//       cv::drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, cv::Point() );
//     }

  /// Show in a window
//  cv::namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
//  cv::imshow( "Contours", drawing );
//  cv::imshow("edges", edgemask);
//  cv::waitKey(0);
}



void bgfgImage(cv::Mat& resized_frame,
				cv::BackgroundSubtractorMOG2& bg_model,
				int frame_counter,
				int cycle_position,
				double scale_factor,
				double learning_rate,
				int image_processing_threshold,
				int maximum_frame_threshold,
				bool training_only,
				cv::Mat perspective_matrix,
				double& frame_feature) {
	cv::Mat img, background_image, edgemask, fgmask, fgmask1, fgmask2, fgmask3;
	int threshold = 10;
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;

	resized_frame.copyTo(img);

	// blur the color image
	cv::GaussianBlur(img, img, cv::Size(9,9), 1.5, 1.5);

	// background subtraction -- this trains the background
	bg_model(img, fgmask, learning_rate);

	if (frame_counter == maximum_frame_threshold - 1) {
		std::cout << "saving background" << std::endl;
		bg_model.getBackgroundImage(background_image);
		cv::imwrite(addStr("data/backgrounds/background_",cycle_position,".png"),background_image);
	}

	if (frame_counter > 60 && !training_only) {
		bg_model.getBackgroundImage(background_image);
		cv::threshold(fgmask, fgmask, 0, 255, 0);
		/* Apply erosion operation */
		int parameter = 1;
		applyErosionParameters(fgmask, fgmask1, parameter);

		/* Apply dilation operation */
		parameter = 1;
		applyDilationParameters(fgmask1, fgmask2, parameter);
		//  cv::createTrackbar( " Canny thresh:", "image", &thresh, max_thresh);
		applyEdgeThreshold(img, edgemask, threshold);
		cv::bitwise_and(fgmask2, edgemask, fgmask2);

		cv::add(fgmask1, fgmask2, fgmask3);
		// cv::imshow( "fgmask3", fgmask3);

		// fill holes on the foreground mask
		parameter = 3;
		applyDilationParameters(fgmask3, fgmask3, parameter);
		parameter = 3;
		applyErosionParameters(fgmask3, fgmask3, parameter);

		cv::threshold(fgmask3, fgmask3, 0, 255, 0);
		// Find contours
		contours.clear();
		cv::findContours( fgmask3, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

		  //  Draw contours
		  frame_feature = 0;
		  int tmpi = contours.size();
		  for(int i = 0; i< tmpi; i++ )
		  {
			   int c_size = contours[i].size() ;
			   if (c_size > 30)
			   {
				cv::Scalar color = cv::Scalar( 1, 0, 0);
				cv::Mat Individual_blob = cv::Mat::zeros( fgmask3.size(), CV_64FC1);
				cv::drawContours( Individual_blob, contours, i, color, CV_FILLED, 8, hierarchy, 0, cv::Point() );
				cv::Point low_p, temp;
				low_p = contours[i][0];
				for(int j = 1; j< c_size; j++)
				{
					temp = contours[i][j];
					if (temp.y > low_p.y )   low_p = temp;
				}

				// perspective correction
				double perspective_para = 0;
				cv::Point pp;
				pp.x = low_p.x ;
				for (int jj = low_p.y; jj < perspective_matrix.rows ;jj ++){
					perspective_para = perspective_matrix.at<float>(jj, low_p.x);
					if (perspective_para > 0 )
					{
						pp.y = jj;
						break;
					}
					pp.y = jj;
				}
			    cv::Scalar pre = sum(Individual_blob);
				double feature =  pre.val[0] * perspective_para;
				frame_feature = frame_feature + feature;
				// cv::imshow( "Blob", Individual_blob);
				// cv::waitKey(0);
			   }
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


