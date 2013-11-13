#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <stdio.h>
#include <vector>

#include <seam4us_functions.h>

int getPixelValue(cv::Mat& rgb_image, int x, int y, int channel) {
	// channel: 0 = r, 1 = g, 2 = b
	switch (channel) {
	case 0:
		return (int)(rgb_image.ptr< cv::Point3_<uchar> >(x,y)->z);
		break;
	case 1:
		return (int)(rgb_image.ptr< cv::Point3_<uchar> >(x,y)->y);
		break;
	case 2:
		return (int)(rgb_image.ptr< cv::Point3_<uchar> >(x,y)->x);
		break;
	}
	return -1;
}

int getPixelValue(cv::Mat& gray_image, int x, int y) {
	return (int)(gray_image.at<uchar>(x,y));
}

void drawLines(std::vector <cv::Point>& point_vector, cv::Mat& canvas, int line_thickness = 4, int line_type=16, cv::Scalar color = cv::Scalar(255,0,0)) {
	cv::Point prev_point;
	cv::Point first_point;
	int i = 0;
	for (std::vector<cv::Point>::iterator it = point_vector.begin(); it != point_vector.end();  it++) {
		if (i == 0)
			first_point = *it;
		else
			cv::line(canvas,prev_point,*it,color,line_thickness,line_type);
		prev_point = *it;
		i += 1;
	}
	cv::line(canvas,prev_point,first_point,color,line_thickness,line_type);
}

void drawPolygons(std::vector < std::vector<cv::Point> >& line_vector, cv::Mat canvas, cv::Scalar color = cv::Scalar(255,255,255)) {
	std::vector <std::vector<cv::Point> > contourElement;
	for (unsigned int counter = 0; counter < line_vector.size(); counter ++) {
		contourElement.push_back(line_vector.at(counter));
		std::vector<cv::Point> tmp = line_vector[counter];
		const cv::Point* elementPoints[1] = { &tmp[0] };
		int numberOfPoints = (int)tmp.size();
		cv::fillPoly (canvas, elementPoints, &numberOfPoints, 1, color, 16);
	}
}
