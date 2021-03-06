#pragma once

// setup functions
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
			);

// misc functions
	bool readCSV(std::string filename, int x, int y);

	bool readCSV(std::string filename, std::vector< std::vector<float> >& content);

	bool readCSV(std::string filename, std::vector< std::vector<cv::Point> >& content);

	bool readCSV(std::string filename, cv::Mat& content);

	bool checkCSV(std::string filename);

// image functions
	int getPixelValue(cv::Mat& rgb_image, int x, int y, int channel);

	int getPixelValue(cv::Mat& gray_image, int x, int y);

	void drawLines(std::vector <cv::Point>& point_vector, cv::Mat& canvas, int line_thickness, int line_type, cv::Scalar color);

	void drawPolygons(std::vector < std::vector<cv::Point> >& line_vector, cv::Mat canvas, cv::Scalar color);

	bool getImageStream(std::string video_address, cv::VideoCapture& img_stream);

// machine learning functions
	bool checkTrainingStatus(int amount_of_cameras, int amount_of_training_coefficients);

	bool getTrainedCoefficients(std::vector<std::vector<float> >& coefficients);

	void serializeImage(cv::Mat& data, cv::Mat& image, int columns, int rows);

	void serializeImage(cv::Mat& data, cv::Mat& image, int columns, int rows, int offset);

	void testOCR(cv::KNearest& OCR);

	bool trainOCR(cv::KNearest& OCR);

// ocr functions
	int identifyNumber(cv::KNearest& OCR, cv::Mat& number);

	int identifyCamera(cv::KNearest& OCR, cv::Mat& number1, cv::Mat& number2, cv::Mat& number3);

	int getPositionFromOCR(
							std::vector<int>& camera_cycle,
							cv::KNearest& OCR,
							cv::Mat& number1,
							cv::Mat& number2,
							cv::Mat& number3,
							int cycle_position
							);

	bool determineCyclePosition(
								cv::Mat& gray,
								std::vector<int>& camera_cycle,
								cv::KNearest& OCR,
								int& prev_found_position,
								int& cycle_position,
								int& successful_ocr,
								int& failed_ocr,
								int& number_of_deviations
								);


// camera functions
	void getCameraCycle(std::vector<int>& camera_cycle);

	bool isCameraOffline(cv::Mat& rgb_image);

	bool isCameraSwitching(int& frame_counter,
							cv::Mat& gray,
							cv::Mat& gray_previous,
							cv::Mat& gray_difference,
							cv::Mat& frame,
							bool& previous_camera_offline,
							bool& offline_camera_switched);


// ROI functions
	bool checkROI(int amount_of_cameras);

	bool handleROI(std::vector<cv::Mat>& camera_frames, cv::Mat& frame, int cycle_position, int amount_of_cameras, std::vector<cv::Mat>& ROI_masks);

	void createROImasks(std::vector<cv::Mat>& ROI_masks, int amount_of_cameras, cv::Mat& frame, double scale_factor);

	void applyROImaskToFrame(cv::Mat& frame, std::vector<cv::Mat>& ROI_masks, int cycle_position);


// Perspective functions
	bool checkPerspectives(int amount_of_cameras);

	bool handlePerspectives(
							std::vector<std::vector<cv::Mat> >& camera_clips,
							cv::Mat& frame,
							int cycle_position,
							int amount_of_cameras,
							int& save_frame
							);

	bool handlePerspectivesDebug(
								std::vector<std::vector<cv::Mat> >& camera_clips,
								cv::Mat& frame,
								int cycle_position,
								int amount_of_cameras,
								int& save_frame
								);

	void loadPerspectiveMatrices(std::vector<cv::Mat>& perspective_matrices, int amount_of_cameras, cv::Mat& frame, double scale_factor);


// analysis functions
	void bgfgImage(
					cv::Mat& resized_frame,
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
					);

	int initializeBackgrounds(std::vector<cv::BackgroundSubtractorMOG2>& background_model_vector, double learning_rate, int amount_of_training_cycles, int amount_of_training_cycles_from_nothing);

// data writing functions
	void writeResults(double value, double std, int cameraID);

	void convertFeaturesToPeople(double features, int cycle_position, std::vector<std::vector<float> >& training_coefficients, int cameraID);









