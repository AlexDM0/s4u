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
#include <profiling_class.hpp>

bool ENABLE_PROFILER = false;

int main(int argc, const char** argv) {
	/* ===========================================================================================*/
	/* ============  				INITIAL SETUP AND CONFIGURATION			======================*/
	/* ===========================================================================================*/

		std::string video_address = "rtsp://127.0.0.1:8554/";

		// connect to video stream
		std::cout << "Opening video stream at: \n" << video_address;
		cv::VideoCapture img_stream;
		if (!getImageStream(video_address,img_stream)) {
			std::cout << "Could not open video feed @ " << video_address << std::endl;
			return -1;
		}
		std::cout << "\n Done.\n" << std::endl;

		// get the camera cycle vector
		std::vector<int> camera_cycle;
		getCameraCycle(camera_cycle);
		int amount_of_cameras = camera_cycle.size();

		// train the OCR
		std::cout << "Training OCR... ";
		cv::KNearest OCR;
		if (!trainOCR(OCR)) {
			std::cout << "Failed getting OCR Data" << std::endl;
			return -1;
		}
		std::cout << "Done." << std::endl;

		// check if the Regions of Interest are defined
		std::cout << "Checking Region of Interest... ";
		bool setup_ROI = !checkROI(amount_of_cameras);
		if (setup_ROI)
			std::cout << "ROI is not defined yet." << std::endl;
		else
			std::cout << "Done." << std::endl;

		// check if the Perspectives are defined
		std::cout << "Checking perspectives... ";
		bool setup_perspective = !checkPerspectives(amount_of_cameras);
		if (setup_perspective)
			std::cout << "Perspectives are not defined yet." << std::endl;
		else
			std::cout << "Done." << std::endl;


	/* ===========================================================================================*/
	/* ============  		Finished setup and config, create vars			======================*/
	/* ===========================================================================================*/


		// create profiler
		profiler timer(ENABLE_PROFILER);

		// initialize the variables
		cv::Mat frame, resized_frame, gray, gray_previous, gray_difference, foreground_mask, test;
		std::vector<cv::Mat> backgrounds, ROI_masks, perspective_matrices;
		std::vector<cv::Mat> camera_frames (amount_of_cameras);
		std::vector<std::vector<cv::Mat> > camera_clips (amount_of_cameras);
		std::vector<cv::BackgroundSubtractorMOG2> background_model_vector(amount_of_cameras);
		std::vector<std::vector<float> > training_coefficients;

		// initialize the integers
		int frame_counter = 0;					int cycle_position = 0;				int previous_cycle_position = 0;
		int prev_found_position = 0;			int successful_ocr = 0;				int failed_ocr = 0;
		int amount_of_camera_switches = 0;		int number_of_deviations = 0; 		int save_frame = 0;
		int cycle = 0;
		//int key = -1;

		// initialize the booleans
		bool initialization_complete = false;	bool previous_camera_offline = false;		bool start_analysis = false;
		bool roi_masks_created = false;			bool perspective_matrices_loaded = false;	bool offline_camera_switched = false;

		// initialize and set configuration variables
		int minimum_amount_of_switches = 10;
		double scale_factor = 1;
		double learning_rate = 0.005;
		int amount_of_training_cycles = 5;
		int amount_of_training_cycles_from_nothing = 20;
		int background_training_frames = 55;
		int processing_frames = 5;
		int maximum_frame_threshold = background_training_frames + processing_frames + 1;
		int amount_of_training_coefficients = 5;
		double frame_feature = 0;
		double sum_feature = 0;
		double max_perspective_multiplier = 5;


	/* ===========================================================================================*/
	/* ============  	Determining if system has trained coefficients		======================*/
	/* ===========================================================================================*/

		// check if the system is Trained (features-to-people).
		// We allow the system to pass if the ROI or the Perspectives need to be set up.
		// After ROI or perspective changes, the system HAS to be retrained.
		if (!setup_ROI && !setup_perspective) {
			std::cout << "Checking training status... ";
			if (checkTrainingStatus(amount_of_cameras,amount_of_training_coefficients)) {
				getTrainedCoefficients(training_coefficients);
				std::cout << "Done." << std::endl;
			}
			else {
				std::cout << "The system needs to be trained first: coefficients.txt cannot be found or has the wrong dimensions." << std::endl;
				std::cout << "Expecting format: " << amount_of_training_coefficients << "x" << amount_of_cameras << ", comma separated. Quitting.." << std::endl;
				return -1;
			}
		}

		// determine how many cycles the background should be trained before the analysis begins.
		// this requires a LOT of memory
		std::cout << "Getting the amount of required training cycles: ";
		int training_cycles = initializeBackgrounds(background_model_vector, learning_rate, amount_of_training_cycles, amount_of_training_cycles_from_nothing) + 1;
		std::cout << training_cycles - 1 << "." << std::endl;


	/* ===========================================================================================*/
	/* ============  					Starting algorithm					======================*/
	/* ===========================================================================================*/


	// start the image cycle
	for(;;) {
		// get a frame from the stream
		img_stream >> frame;

		// if the frame is empty, close the program.
		if(frame.empty()) {
			std::cout << "No more frames available. Closing program." << std::endl;
			break;
		}

		// Convert frame to gray scale image
		gray.copyTo(gray_previous);
		cv::cvtColor(frame, gray, CV_BGR2GRAY);

		// Detect if the camera switches
		if (isCameraSwitching(frame_counter,gray,gray_previous,gray_difference,frame,previous_camera_offline,offline_camera_switched)) {
			successful_ocr = 0;
			failed_ocr = 0;
			start_analysis = false;
			if (amount_of_camera_switches < 100)
				amount_of_camera_switches += 1;

			if (offline_camera_switched) {
				writeResults(-1.0,0,cycle_position);
				offline_camera_switched = false;
			}
			// Set expectation
			cycle_position = (cycle_position+1)%amount_of_cameras;
			if (cycle_position%amount_of_cameras == 0)
				cycle_position = 0;
			save_frame = 0;
		}

		// At the switch of the camera the frame counter is set to 0. After 5 frames we start OCR detection.
		// We want either 2 consecutive matches or we take the expected position
		// We accept two deviations from the expectations. During these two we take the expectation.
		if (frame_counter >= 5 && !start_analysis) {
			start_analysis = determineCyclePosition(
													gray,
													camera_cycle,
													OCR,
													prev_found_position,
													cycle_position,
													successful_ocr,
													failed_ocr,
													number_of_deviations
													);
			if (number_of_deviations == 0 && start_analysis && amount_of_camera_switches > minimum_amount_of_switches)
				initialization_complete = true;
		}

		if (start_analysis && initialization_complete && frame_counter < maximum_frame_threshold) {
			// resize if required
			if (!setup_ROI && !setup_perspective && scale_factor != 1)
				cv::resize(frame, resized_frame, cv::Size(scale_factor*frame.cols, scale_factor*frame.rows));
			else
				frame.copyTo(resized_frame);

			// get the ROI and perspective if needed and apply the ROI
			if (!setup(	setup_ROI,
						roi_masks_created,
						setup_perspective,
						perspective_matrices_loaded,
						cycle_position,
						amount_of_cameras,
						save_frame,
						scale_factor,
						resized_frame,
						frame,
						camera_frames,
						ROI_masks,
						perspective_matrices,
						camera_clips
					))
				break;

			// start the processing of the video data
			if (!setup_ROI && !setup_perspective) {
				// counting the cycle position
				if (((cycle_position + 1) == amount_of_cameras) && (previous_cycle_position != cycle_position)){
					cycle += 1;
					if (training_cycles > 0)
						training_cycles -= 1;
				}

				// once the backgrounds are trained, start the processing
				if (training_cycles <= 0) {
					// run the processing function
					bgfgImage(resized_frame,
								background_model_vector.at(cycle_position),
								ROI_masks,
								frame_counter,
								cycle_position,
								scale_factor,
								learning_rate,
								background_training_frames,
								processing_frames,
								false,
								perspective_matrices.at(cycle_position),
								frame_feature,
								max_perspective_multiplier,
								training_cycles
								);

					// calculations are running and feature average for process_frames amount of frames are calculated
				    if (frame_counter > background_training_frames)
						sum_feature = sum_feature + frame_feature;

				    // calculations for camera are complete
				    if (frame_counter == background_training_frames + processing_frames){
				    	// get the average
				    	sum_feature = sum_feature/processing_frames;

				    	// write the framedata to disk
				    	convertFeaturesToPeople(sum_feature,cycle_position,training_coefficients,camera_cycle[cycle_position]);

				    	// reset sum feature
				    	sum_feature = 0;
				    }
				}
				else {
					// this only trains the background because training_only == true
					bgfgImage(resized_frame,
								background_model_vector.at(cycle_position),
								ROI_masks,
								frame_counter,
								cycle_position,
								scale_factor,
								learning_rate,
								background_training_frames,
								processing_frames,
								true,
								perspective_matrices.at(cycle_position),
								frame_feature,
								max_perspective_multiplier,
								training_cycles
								);
				}

				// updating the cycle position
				previous_cycle_position = cycle_position;
			}
		}
		else if (start_analysis) {
			// intermediate step while the position within the cycle is being determined
		}


		/*
		if (!setup_ROI && !setup_perspective) {
			cv::imshow("feed", frame);
			key = cv::waitKey(1);
		}


		if (key == 112) { // p -- open perspective editor
			setup_perspective = true;
			key = -1;
			roi_masks_created = false;
			std::cout << "entering the perspective editor." << std::endl;
			cv::destroyWindow("feed");
		}
		else if (key == 114) { // r -- open Region of Interest editor
			setup_ROI = true;
			roi_masks_created = false;
			key = -1;
			std::cout << "entering the region-of-interest editor." << std::endl;
		}
		else if(key == 27) {
			std::cout << key << std::endl;
			break;
		}
		*/
		frame_counter += 1;
	}
    return 0;
}
