#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <stdio.h>
#include <vector>

#include <seam4us_functions.h>

void getCameraCycle(std::vector<int>& camera_cycle) {
	camera_cycle.push_back(734);	// 1
	camera_cycle.push_back(7112);	// 2
	camera_cycle.push_back(718);	// 3
	camera_cycle.push_back(7111);	// 4
	camera_cycle.push_back(7110);	// 5
	camera_cycle.push_back(721);	// 6
	camera_cycle.push_back(722);	// 7
	camera_cycle.push_back(7113);	// 8
	camera_cycle.push_back(7114);	// 9
	camera_cycle.push_back(7100);	// 10
	camera_cycle.push_back(7104);	// 11
	camera_cycle.push_back(747);	// 12
	camera_cycle.push_back(748);	// 13
	camera_cycle.push_back(700);	// 14
	camera_cycle.push_back(701);	// 15
	camera_cycle.push_back(705);	// 16
	camera_cycle.push_back(704);	// 17
	camera_cycle.push_back(711);	// 18
	camera_cycle.push_back(710);	// 19
	camera_cycle.push_back(712);	// 20
	camera_cycle.push_back(713);	// 21
	camera_cycle.push_back(719);	// 22
	camera_cycle.push_back(714);	// 23
	camera_cycle.push_back(715);	// 24
	camera_cycle.push_back(716);	// 25
	camera_cycle.push_back(720);	// 26
	camera_cycle.push_back(730);	// 27
	camera_cycle.push_back(731);	// 28
	camera_cycle.push_back(732);	// 29
	camera_cycle.push_back(733);	// 30
	camera_cycle.push_back(717);	// 31
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
			if (previous_camera_offline)
				camera_switched = true;
			offline_camera_switched = true;
			previous_camera_offline = false;
		}

		if (camera_switched)
			frame_counter = 0;
		return camera_switched;
	}
	else
		return false;
}
