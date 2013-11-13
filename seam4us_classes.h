
class perspective {
public:
	std::vector<cv::Point> points;
	int direction;
	// 0: from 0 to 2
	// 1: from 2 to 0
	// 2: from 1 to 3
	// 2: from 3 to 1

	perspective(std::vector<cv::Point> points);

	void draw(cv::Mat& canvas);

	~perspective() ;
};
