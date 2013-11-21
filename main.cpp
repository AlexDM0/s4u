/*
int main()
{
  LARGE_INTEGER start, end, freq;

  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&start);

  some_operation();

  QueryPerformanceCounter(&end);

  std::cout << "The resolution of this timer is: " << freq.QuadPart << " Hz." << std::endl;
  std::cout << "Time to calculate some_operation(): "
            << (end.QuadPart - start.QuadPart) * 1000000 / freq.QuadPart
            << " microSeconds" << std::endl;
}

*/
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

bool setup(
			bool& setup_ROI,
			bool& roi_masks_created,
			bool& setup_perspective,
			bool& perspective_matrices_loaded,
			int& cycle_position,
			int& amount_of_cameras,
			int& save_frame,
			double& scale_factor,
			cv::Mat& resized_frame,
			cv::Mat& frame,
			std::vector<cv::Mat>& camera_frames,
			std::vector<cv::Mat>& ROI_masks,
			std::vector<cv::Mat>& perspective_matrices,
			std::vector<std::vector<cv::Mat> >& camera_clips
			) {
	// after creation of the ROI the program has to be restarted.
	if (setup_ROI) {
		if (!handleROI(camera_frames, frame, cycle_position, amount_of_cameras, ROI_masks))
			return false;
	}


	if (!roi_masks_created && !setup_ROI) {
		std::cout << "Creating the Region of Interest masks... ";
		// we use the fullsize region of interest with the perspective setup.
		ROI_masks.clear();
		if (setup_perspective) {
			createROImasks(ROI_masks, amount_of_cameras, frame, 1.0);
			std::cout << "Creating ROI masks for perspective...";
		}
		else
			createROImasks(ROI_masks, amount_of_cameras, frame, scale_factor);
		std::cout << "Done." << std::endl;
		roi_masks_created = true;
	}

	if (!setup_ROI)
		applyROImaskToFrame(resized_frame,ROI_masks,cycle_position);

	// after creation of the perspectives the program has to be restarted.
	if (!setup_ROI) {
		if (setup_perspective) {
			applyROImaskToFrame(frame,ROI_masks,cycle_position);
			if (!handlePerspectives(camera_clips, frame, cycle_position, amount_of_cameras, save_frame))
				return false;
		}
	}

	// loading the perspective matrices into memory
	if (!perspective_matrices_loaded && !setup_perspective) {
		std::cout << "Loading the perspective matrices";
		perspective_matrices.clear();
		loadPerspectiveMatrices(perspective_matrices, amount_of_cameras, frame, scale_factor);
		perspective_matrices_loaded = true;
		std::cout << "Done" << std::endl;
	}
	return true;
}

int main(int argc, const char** argv) {
	std::string video_address = "C:/Data from server/Tuesday (24-09-2013)(19.03-20.00).mp4";

	// connect to video stream
	std::cout << "Opening video stream at: \n" << video_address;
	cv::VideoCapture img_stream;
	if (!getImageStream(video_address,img_stream)) {
		std::cout << "Could not open video feed." << std::endl;
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

	// initialize the variables
	cv::Mat frame, resized_frame, gray, gray_previous, gray_difference, foreground_mask, test;
	std::vector<cv::Mat> backgrounds, ROI_masks, perspective_matrices;
	std::vector<cv::Mat> camera_frames (amount_of_cameras);
	std::vector<std::vector<cv::Mat> > camera_clips (amount_of_cameras);
	std::vector<cv::BackgroundSubtractorMOG2> background_model_vector(amount_of_cameras);
	std::vector<std::vector<float> > training_coefficients;

	// initialize the integers
	int key = -1;							int frame_counter = 0;				int cycle_position = 0;		int previous_cycle_position = 0;
	int prev_found_position = 0;			int successful_ocr = 0;				int failed_ocr = 0;			int cycle = 0;
	int amount_of_camera_switches = 0;		int number_of_deviations = 0; 		int save_frame = 0;

	// initialize the booleans
	bool initialization_complete = false;	bool previous_camera_offline = false;		bool start_analysis = false;
	bool roi_masks_created = false;			bool perspective_matrices_loaded = false;	bool offline_camera_switched = false;

	// initialize and set configuration variables
	int minimum_amount_of_switches = 10;
	double scale_factor = 1;
	double learning_rate = 0.005;
	int maximum_frame_threshold = 65;
	int amount_of_training_cycles = 60;
	int amount_of_training_cycles_from_nothing = 1;
	int image_processing_threshold = 60;
	int averaging_frames = 4;
	int amount_of_training_coefficients = 5;
	double frame_feature = 0;
	double sum_feature = 0;
	double max_perspective_multiplier = 5;
	profiler timer(ENABLE_PROFILER);

	// save results in file
	std::ofstream Counting_file;
	Counting_file.open("friday_1_People_counting.txt");
	int sample_frame = 1 ;

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
	std::cout << "Getting the amount of required training cycles: ";
	int training_cycles = initializeBackgrounds(background_model_vector, learning_rate, amount_of_training_cycles, amount_of_training_cycles_from_nothing) + 1;
	std::cout << training_cycles - 1 << "." << std::endl;

	// start the image cycle
	for(;;) {
		// get a frame from the stream
		img_stream >> frame;

		// if the frame is empty, close the program.
		if(frame.empty()) {
			Counting_file.close();
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
			if (!setup_ROI && !setup_perspective)
				cv::resize(frame, resized_frame, cv::Size(scale_factor*frame.cols, scale_factor*frame.rows));
			else
				frame.copyTo(resized_frame);

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

			if (!setup_ROI && !setup_perspective) {
				// counting the cycle position
				if (((cycle_position + 1) == amount_of_cameras) && (previous_cycle_position != cycle_position)){
					cycle += 1;
					if (training_cycles > 0)
						training_cycles -= 1;
				}


				// once the backgrounds are trained, start the processing
				if (training_cycles <= 0) {
					timer.start("processing and background function");
					bgfgImage(resized_frame,
								background_model_vector.at(cycle_position),
								frame_counter,
								cycle_position,
								scale_factor,
								learning_rate,
								image_processing_threshold,
								maximum_frame_threshold,
								false,
								perspective_matrices.at(cycle_position),
								frame_feature,
								max_perspective_multiplier,
								training_cycles
								);
					timer.end();

				    if (frame_counter > image_processing_threshold)
						sum_feature  = sum_feature + frame_feature;

				    if (frame_counter == image_processing_threshold + averaging_frames){
				    	sum_feature = sum_feature/averaging_frames;
				    	convertFeaturesToPeople(sum_feature,cycle_position,training_coefficients);
				    	std::cout << sample_frame << ";" << cycle_position << ";" << sum_feature << std::endl;
				    	Counting_file << sample_frame << ";" << cycle_position << ";" << sum_feature << "\n";
				    	sum_feature = 0;
				    	sample_frame++;
				    }
				}
				else { // this only trains the background


					bgfgImage(resized_frame,
								background_model_vector.at(cycle_position),
								frame_counter,
								cycle_position,
								scale_factor,
								learning_rate,
								image_processing_threshold,
								maximum_frame_threshold,
								true,
								perspective_matrices.at(cycle_position),
								frame_feature,
								max_perspective_multiplier,
								training_cycles
								);
//					cv::putText(frame, "Training Backgrounds.", cv::Point(150,100), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(255,255,255),2);
//					cv::putText(frame, addStr("Cycles left: ",training_cycles), cv::Point(150,130), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(255,255,255),2);
				}

				// updating the cycle position
				previous_cycle_position = cycle_position;
			}
			//cv::putText(frame, addStr("camID: ", camera_cycle[cycle_position], " pos: ",cycle_position), cv::Point(200,60), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,255));
		}
		else if (start_analysis) {
			// intermediate step while the position within the cycle is being determined
			//cv::putText(frame, "Initializing System.", cv::Point(200,100), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(255,255,255),2);
		}


//		if (!setup_ROI && !setup_perspective) {
//			cv::imshow("feed", frame);
//			key = cv::waitKey(1);
//		}


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
		frame_counter += 1;
	}
	Counting_file.close();
    return 0;
}
