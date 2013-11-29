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

	// applying ROI mask to the frame
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
