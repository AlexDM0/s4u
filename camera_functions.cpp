#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <stdio.h>
#include <vector>

#include <seam4us_functions.h>
#include <template_functions.hpp>

/*
void getActiveCameras(std::vector<int>& active_cameras) {
	active_cameras.push_back(0);	// 0
	active_cameras.push_back(1);	// 1
	active_cameras.push_back(0);	// 2
	active_cameras.push_back(1);	// 3
	active_cameras.push_back(1);	// 4
	active_cameras.push_back(1);	// 5
	active_cameras.push_back(1);	// 6
	active_cameras.push_back(0);	// 7
	active_cameras.push_back(0);	// 8
	active_cameras.push_back(1);	// 9
	active_cameras.push_back(1);	// 10
	active_cameras.push_back(0);	// 11
	active_cameras.push_back(0);	// 12
	active_cameras.push_back(1);	// 13
	active_cameras.push_back(1);	// 14
	active_cameras.push_back(1);	// 15
	active_cameras.push_back(1);	// 16
	active_cameras.push_back(1);	// 17
	active_cameras.push_back(1);	// 18
	active_cameras.push_back(1);	// 19
	active_cameras.push_back(0);	// 20
	active_cameras.push_back(0);	// 21
	active_cameras.push_back(1);	// 22
	active_cameras.push_back(0);	// 23
	active_cameras.push_back(1);	// 24
	active_cameras.push_back(1);	// 25
	active_cameras.push_back(0);	// 26
	active_cameras.push_back(1);	// 27
	active_cameras.push_back(1);	// 28
	active_cameras.push_back(1);	// 29
	active_cameras.push_back(0);	// 30
}
*/
void getCameraCycle(std::vector<int>& camera_cycle) {
	//camera_cycle.push_back(734);	// 0  --  1
	camera_cycle.push_back(7112);	// 1  --  2
	//camera_cycle.push_back(718);	// 2  --  3
	camera_cycle.push_back(7111);	// 3  --  4
	camera_cycle.push_back(7110);	// 4  --  5
	camera_cycle.push_back(721);	// 5  --  6
	camera_cycle.push_back(722);	// 6  --  7
	//camera_cycle.push_back(7113);	// 7  --  8
	//camera_cycle.push_back(7114);	// 8  --  9
	camera_cycle.push_back(7100);	// 9  --  10
	camera_cycle.push_back(7104);	// 10 --  11
//	camera_cycle.push_back(747);	// 11 --  12
//	camera_cycle.push_back(748);	// 12 --  13
	camera_cycle.push_back(700);	// 13 --  14
	camera_cycle.push_back(701);	// 14 --  15
	camera_cycle.push_back(705);	// 15 --  16
	camera_cycle.push_back(704);	// 16 --  17
	camera_cycle.push_back(711);	// 17 --  18
	camera_cycle.push_back(710);	// 18 --  19
	camera_cycle.push_back(712);	// 19 --  20
	//camera_cycle.push_back(713);	// 20 --  21
	//camera_cycle.push_back(719);	// 21 --  22
	camera_cycle.push_back(714);	// 22 --  23
	//camera_cycle.push_back(715);	// 23 --  24
	camera_cycle.push_back(716);	// 24 --  25
	camera_cycle.push_back(720);	// 25 --  26
	//camera_cycle.push_back(730);	// 26 --  27
	camera_cycle.push_back(731);	// 27 --  28
	camera_cycle.push_back(732);	// 28 --  29
	camera_cycle.push_back(733);	// 29 --  30
	//camera_cycle.push_back(717);	// 30 --  31
}

bool isCameraOffline(cv::Mat& rgb_image) {
	int r_target = 29;
	int g_target = 43;
	int b_target = 182;
	int threshold = 2;
	bool offline = true;

	std::vector<int> x_positions;
	std::vector<int> y_positions;

	x_positions.push_back(rgb_image.cols*0.1); x_positions.push_back(rgb_image.cols*0.1);
	x_positions.push_back(rgb_image.cols*0.2); x_positions.push_back(rgb_image.cols*0.2);
	x_positions.push_back(rgb_image.cols*0.8); x_positions.push_back(rgb_image.cols*0.8);

	y_positions.push_back(rgb_image.rows*0.1); y_positions.push_back(rgb_image.rows*0.1);
	y_positions.push_back(rgb_image.rows*0.2); y_positions.push_back(rgb_image.rows*0.2);
	y_positions.push_back(rgb_image.rows*0.8); y_positions.push_back(rgb_image.rows*0.8);

	for (unsigned int i = 0; i < y_positions.size(); i++) {
		if (abs(getPixelValue(rgb_image,x_positions[i],y_positions[i],0) - r_target) > threshold)
			offline = false;
		if (abs(getPixelValue(rgb_image,x_positions[i],y_positions[i],1) - g_target) > threshold)
			offline = false;
		if (abs(getPixelValue(rgb_image,x_positions[i],y_positions[i],2) - b_target) > threshold)
			offline = false;
	}

	return offline;
}

bool isCameraSwitching(int& frame_counter,
						cv::Mat& gray,
						cv::Mat& gray_previous,
						cv::Mat& gray_difference,
						cv::Mat& frame,
						bool& previous_camera_offline,
						bool& offline_camera_switched) {
	if (frame_counter > 2) {
		bool camera_switched = false;
		double change_threshold = 0.05;
		int frames_threshold = 50;

		absdiff(gray, gray_previous, gray_difference);
		cv::Scalar temp = sum(gray_difference);
		double normalized_change = temp[0]/(gray_difference.rows*gray_difference.cols*255);
		if ((normalized_change>change_threshold)&&(frame_counter>frames_threshold))
			camera_switched = true;
		else
			camera_switched = false;


		if (isCameraOffline(frame))
			previous_camera_offline = true;
		else {
			if (previous_camera_offline) {
				camera_switched = true;
				offline_camera_switched = true;
			}
			previous_camera_offline = false;
		}

		if (camera_switched)
			frame_counter = 0;
		return camera_switched;
	}
	else
		return false;
}
