#pragma once

#include <opencv2/opencv.hpp>
#include <QVector2D>

class KMeans {
public:
	std::vector<cv::MatND> meanLengths;
	std::vector<cv::MatND> meanCurvatures;
	std::vector<cv::MatND> histLengths;
	std::vector<cv::MatND> histCurvatures;
	std::vector<QVector2D> locations;
	std::vector<int> groups;

public:
	KMeans();
	~KMeans() {}

	void add(cv::MatND& histLength, cv::MatND& histCurvature, int x, int y);
	float distance(cv::MatND& hist1, cv::MatND& hist2);
	void cluster();
	cv::Mat_<uchar> getSegmentation();
};

