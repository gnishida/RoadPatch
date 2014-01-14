/**
 * このプログラムは、Binary Coherent Edge Descriptorsの論文を参考に、
 * 当該論文の内容を実装したものです。
 *
 * @author Gen Nishida
 * @version 1.0
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <QVector2D>
#include <QHash>
#include "GraphUtil.h"
#include "BBox.h"
#include "KMeans.h"

int main(int argc, char *argv[]) {
	int patch_size = 60;
	
	// Load the road graph (except highways) from GSM file.
	RoadGraph r;
	//GraphUtil::loadRoads(r, "osm/3x3_simplified/london_3.gsm");
	GraphUtil::loadRoads(r, "osm/1x1/canberra.gsm");

	// K-Means
	KMeans km;
	
	// Divide the road graph into 250x250 patches
	int count = 0;
	for (int y = -500; y < 500; y += 250) {
		for (int x = -500; x < 500; x += 250) {
			RoadGraph patch;
			GraphUtil::copyRoads(r, patch);

			BBox box(QVector2D(x, y));
			box.addPoint(QVector2D(x + 250, y + 250));
			GraphUtil::extractRoads(patch, box, false);

			// extract some features from the patch
			cv::MatND histLength = GraphUtil::computeEdgeLengthHistogram(patch, 10);
			cv::MatND histCurvature = GraphUtil::computeEdgeCurvatureHistogram(patch, 10);

			int c = histLength.depth();

			float hoge = compareHist(histLength, histCurvature, 1);

			km.add(histLength, histCurvature, x, y);

			count++;
		}
	}

	// Convert the road graph to a matrix
	km.cluster();
	cv::Mat_<uchar> result = km.getSegmentation();

	// Display
	cv::namedWindow("segmentation", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
	cv::imshow("segmentation", result);

	cv::waitKey(0);
}