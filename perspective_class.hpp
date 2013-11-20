
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <stdio.h>
#include <vector>

#include <seam4us_functions.h>
#include <template_functions.hpp>

class perspective {
public:
	std::vector<cv::Point> points;
	std::vector<std::vector<cv::Point> > markers;
	double length;
	double pixels_per_m;
	double reference_multiplier;
	double reference_distance;
	double vanishing_point_y;
	double size_at_base;
	double normalized_size_to_width;
		// formula: 1 - pf*(distance from beginline)
		// pf = (l_away/l_front)/(distance between away & front)
	int selected;
		// -1: undefined yet
		//  0: from 0 to 2
		//  1: from 2 to 0
		//  2: from 1 to 3
		//  2: from 3 to 1


	perspective(std::vector<cv::Point> points) {
		this->points.swap(points);
		selected = 1;
		length = -1;
		reference_multiplier = -1;
		reference_distance = -1;
		vanishing_point_y = -1;
		size_at_base = -1;
		normalized_size_to_width = -1;

		pixels_per_m = 100;
	}

	perspective() {
		selected = 1;
		length = -1;
		reference_multiplier = -1;
		reference_distance = -1;
		vanishing_point_y = -1;
		size_at_base = -1;
		normalized_size_to_width = -1;

		pixels_per_m = 100;
	}

	void calculatePerspective(cv::Mat& canvas) {
		double marker1_multiplier;
		double marker2_multiplier;
		double marker1_length;
		double marker2_length;
		double distance;
		cv::Point measurement_point1;
		cv::Point measurement_point2;

		// the first marker is closer to the direction. this is the close marker.
		marker1_length = getDistance(markers[0][0],markers[0][1]);
		marker2_length = getDistance(markers[1][0],markers[1][1]);

		measurement_point1 = getMeasurementPoint(markers[0][0],markers[0][1]);
		measurement_point2 = getMeasurementPoint(markers[1][0],markers[1][1]);

		// we assume an y = ax + b fall off of the multiplier over the height
		distance = abs(measurement_point1.y - measurement_point2.y);

		marker1_multiplier = pixels_per_m / (marker1_length/length);
		marker2_multiplier = pixels_per_m / (marker2_length/length);


		if (marker1_length > marker2_length)
			vanishing_point_y = measurement_point2.y - (marker2_length*distance)/(marker1_length - marker2_length);
		else
			vanishing_point_y =	measurement_point1.y - (marker1_length*distance)/(marker2_length - marker1_length);

		reference_multiplier = marker1_multiplier;
		reference_distance = abs(vanishing_point_y - measurement_point1.y);

		std::cout << "distance between parts in pixels: " << distance << "\t difference in size: " << marker1_length-marker2_length << std::endl;
		std::cout << "close length: " << marker1_length << "       distant line length " << marker2_length << std::endl;
		std::cout << "vanishing point: " << vanishing_point_y << "\t reference_multiplier: " << reference_multiplier << std::endl;
		std::cout << "marker1_measure point: " << measurement_point1.y << std::endl;
		std::cout << "point from bottom: " << canvas.rows - measurement_point1.y << std::endl;
		std::cout << "marker1_multiplier: " << marker1_multiplier << std::endl;
		std::cout << "marker2_multiplier: " << marker2_multiplier << std::endl;
	}

	cv::Point getMeasurementPoint(cv::Point& a, cv::Point& b) {
		if (abs(a.x - b.x) < abs(a.y - b.y)) { // more vertical than horizontal
			if (a.y > b.y)
				return a;
			else
				return b;
		}
		else
			return getMidPoint(a,b);
	}

	void setLength(int length) {
		this->length = double(length)/10.0;
	}

	void select() {
		selected = 1;
	}

	void deselect() {
		selected = 0;
	}

	bool detectClick(int x, int y) {
		if (cv::pointPolygonTest( points,cv::Point(x,y),false) >= 0)
			return true;
		else
			return false;
	}

	void getMultiplierMatrix(cv::Mat& matrix) {
		int xmin,xmax,ymin,ymax;
		getMinMax(xmin,xmax,ymin,ymax);

		double multiplier;
//		cv::Point midpoint(matrix.cols/2,matrix.rows/2);
//		double normalization = getDistance(midpoint,cv::Point(0,0));

		for (int x = xmin; x < xmax; x++) {
			for (int y = ymin; y < ymax; y++) {
				if (detectClick(x,y)) {
					multiplier = reference_multiplier * (reference_distance / std::abs(vanishing_point_y - y));
					matrix.at<float>(y,x) = multiplier;
					if (matrix.at<float>(y,x) == 0) // to prevent overwriting with multiple matrices
						matrix.at<float>(y,x) = multiplier;
					else	// we average if they overlap
						matrix.at<float>(y,x) = (matrix.at<float>(y,x) + multiplier)/2.0;
				}
			}
		}
	}

	void drawGradient(cv::Mat& canvas) {
		int xmin,xmax,ymin,ymax;
		getMinMax(xmin,xmax,ymin,ymax);

		cv::Mat multipliers(canvas.size(), CV_32F, cv::Scalar(0));
		getMultiplierMatrix(multipliers);

		drawGradient(canvas,multipliers,xmin,xmax,ymin,ymax);
	}

	void drawGradient(cv::Mat& canvas, cv::Mat& multipliers,int xmin, int xmax, int ymin, int ymax) {
		for (int x = xmin; x < xmax; x++) {
			for (int y = ymin; y < ymax; y++) {
				if (multipliers.at<float>(y,x) != 0) {
					canvas.at<cv::Vec3b>(y,x) = visualize_multiplier(multipliers.at<float>(y,x));
				}
			}
		}
		cv::Mat resized_mults;
		drawLines(points,canvas,2,CV_AA,cv::Scalar(80,80,80));
		cv::line(canvas,cv::Point(0,vanishing_point_y),cv::Point(canvas.cols,vanishing_point_y),cv::Scalar(255,100,100),1,CV_AA);
		cv::putText(canvas, "y = ref_M * (ref_D/(y-yv))", cv::Point(20,120), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0,0,255),1);
		cv::putText(canvas, addStr("yv =", vanishing_point_y), cv::Point(20,140), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0,0,255),1);
		cv::putText(canvas, addStr("ref_M =", reference_multiplier), cv::Point(20,160), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0,0,255),1);
		cv::putText(canvas, addStr("ref_D =", reference_distance), cv::Point(20,180), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0,0,255),1);
	}

	cv::Vec3b visualize_multiplier(double multiplier) {
		cv::Vec3b bgr;
		double threshold = 0.02;
		if (multiplier >= 1.0-threshold && multiplier <= 1.0+threshold) {
			bgr[0] = 255;
			bgr[1] = 255;
			bgr[2] = 255;
		}
		else {
			if (multiplier < 0.1) {
				bgr[0] = 0;
				bgr[1] = 0;
				bgr[2] = 0;
			}
			else if (multiplier < 1) {
				bgr[0] = std::max(0,std::min(255,int(255*multiplier )));
				bgr[1] = std::max(0,std::min(255,int(255*multiplier )));
				bgr[2] = 255;
			}
			else if (multiplier < 2) {
				bgr[0] = 255;
				bgr[1] = std::max(0,std::min(255,int(255*(2.0 - multiplier) )));
				bgr[2] = std::max(0,std::min(255,int(255*(2.0 - multiplier) )));
			}
			else if (multiplier < 3) {
				bgr[0] = std::max(0,std::min(255,int(255*(3.0 - multiplier) )));
				bgr[1] = std::max(0,std::min(255,int(255*(multiplier - 2.0) )));
				bgr[2] = 0;
			}
			else {
				bgr[0] = 0;
				bgr[1] = std::max(0,std::min(255,int(255*(4.0 - multiplier) )));
				bgr[2] = 0;
			}
		}
		return bgr;
	}


	void getMinMax(int& xmin, int& xmax, int& ymin, int& ymax) {
		xmin = 10000;
		ymin = 10000;
		xmax = 0;
		ymax = 0;
		for (int i = 0; i < int(points.size()); i++) {
			if (points[i].x > xmax)
				xmax = points[i].x;
			if (points[i].x < xmin)
				xmin = points[i].x;
			if (points[i].y > ymax)
				ymax = points[i].y;
			if (points[i].y < ymin)
				ymin = points[i].y;
		}
	}

	void getMinMax(cv::Mat& multipliers, int& xmin, int& xmax, int& ymin, int& ymax) {
		xmin = 10000;
		ymin = 10000;
		xmax = 0;
		ymax = 0;
		for (int i = 0; i < multipliers.cols; i++) {
			for (int j = 0; j < multipliers.rows; j++) {
				if (multipliers.at<float>(j,i) != 0) {
					if (i > xmax)
						xmax = i;
					if (i < xmin)
						xmin = i;
					if (j > ymax)
						ymax = j;
					if (j < ymin)
						ymin = j;
				}
			}
		}
	}

	void drawFilled(cv::Mat& canvas) {
		const cv::Point* elementPoints[1] = { &points[0] };
		int numberOfPoints = (int)points.size();
		cv::fillPoly (canvas, elementPoints, &numberOfPoints, 1, cv::Scalar(100,255,100), 16);
		drawLines(points,canvas,4,CV_AA,cv::Scalar(0,80,0));
	}

	void drawMarkers(cv::Mat& canvas) {
		std::vector<cv::Point> end_bar;
		std::vector<cv::Point> begin_bar;
		cv::Point text_position;
		for (int i = 0; i < int(markers.size()); i++) {
			end_bar.clear();
			begin_bar.clear();


			text_position = getMidPoint(markers[i][0],markers[i][1]);
			if (abs(markers[i][0].x - markers[i][1].x) < abs(markers[i][0].y - markers[i][1].y)) {
				begin_bar.push_back(cv::Point(markers[i][0].x + 10,markers[i][0].y));
				begin_bar.push_back(cv::Point(markers[i][0].x - 10,markers[i][0].y));
				end_bar.push_back(cv::Point(markers[i][1].x + 10,markers[i][1].y));
				end_bar.push_back(cv::Point(markers[i][1].x - 10,markers[i][1].y));
				text_position.x += 10;
			}
			else {
				begin_bar.push_back(cv::Point(markers[i][0].x,markers[i][0].y + 10));
				begin_bar.push_back(cv::Point(markers[i][0].x,markers[i][0].y - 10));
				end_bar.push_back(cv::Point(markers[i][1].x,markers[i][1].y + 10));
				end_bar.push_back(cv::Point(markers[i][1].x,markers[i][1].y - 10));
				text_position.x -= 30;
				text_position.y -= 10;
			}

			cv::circle(canvas,getMeasurementPoint(markers[i][0],markers[i][1]),8,cv::Scalar(0,0,255),2);

			drawLines(end_bar,canvas,4,CV_AA,cv::Scalar(0,0,0));
			drawLines(begin_bar,canvas,4,CV_AA,cv::Scalar(0,0,0));
			drawLines(markers[i],canvas,4,CV_AA,cv::Scalar(0,0,0));

			drawLines(end_bar,canvas,2,CV_AA,cv::Scalar(255,255,255));
			drawLines(begin_bar,canvas,2,CV_AA,cv::Scalar(255,255,255));
			drawLines(markers[i],canvas,2,CV_AA,cv::Scalar(255,255,255));



			cv::putText(canvas, addStr(addStr("l = ",length," m [x"),pixels_per_m /(getDistance(markers[i][0],markers[i][1])/length),"]"), text_position, cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(30,30,255),1);
		}
	}

	void draw(cv::Mat& canvas) {
		drawLines(points,canvas,4,CV_AA,cv::Scalar(0,80,0));
	}

	cv::Point getMidPoint(cv::Point& point1, cv::Point& point2) {
		cv::Point midpoint;
		midpoint.x = (point1.x + point2.x) / 2;
		midpoint.y = (point1.y + point2.y) / 2;
		return midpoint;
	}

	double getDistance(cv::Point a, cv::Point b) {
		return sqrt(pow(a.x-b.x,2) + pow(a.y-b.y,2));
	}

	double getAngle(cv::Point a,cv::Point b) {
		double x,y,angle;
		x = abs(a.x - b.x);
		y = abs(a.y - b.y);

		angle = atan(y/x);
		if (a.x > b.x) { // Q2 & Q3
			if (a.y < b.y) // Q3
				angle += 3.141592653;
			else // Q2
				angle = 3.141592653 - angle;
		}
		else { // Q1 & Q4
			if (a.y < b.y) // Q4
				angle = 6.283185306-angle;
			// Q1 does not need alteration
		}


		return angle;
	}

	void drawArrow(cv::Mat& canvas, cv::Point start_point, cv::Point end_point, cv::Scalar color, int thickness, int line_type, int arrow_size)	{
	    //Draw the main line
	    cv::line(canvas, start_point, end_point, color, thickness, line_type);
	    const double PI = 3.141592653;

	    //compute the angle alpha
	    double angle = atan2((double)start_point.y-end_point.y, (double)start_point.x-end_point.x);

	    //compute the coordinates of the first segment
	    start_point.x = (int) (end_point.x +  arrow_size * cos(angle + PI/4.0));
	    start_point.y = (int) (end_point.y +  arrow_size * sin(angle + PI/4.0));
	    //Draw the first segment
	    cv::line(canvas, start_point, end_point, color, thickness, line_type);

	    //compute the coordinates of the second segment
	    start_point.x = (int) (end_point.x +  arrow_size * cos(angle - PI/4.0));
		start_point.y = (int) (end_point.y +  arrow_size * sin(angle - PI/4.0));
	    //Draw the second segment
	    cv::line(canvas, start_point, end_point, color, thickness, line_type);
	}

	~perspective() {
		points.clear();
	}

};
