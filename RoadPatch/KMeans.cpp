#include "KMeans.h"

KMeans::KMeans() {
	cv::MatND meanLength = cv::MatND::zeros(1, 10, CV_64FC1);
	cv::MatND meanCurvature = cv::MatND::zeros(1, 10, CV_64FC1);

	// Dense & Grid
	meanLength.at<double>(0, 0) = 1.0f;
	meanCurvature.at<double>(0, 0) = 1.0f;

	meanLengths.push_back(meanLength.clone());
	meanCurvatures.push_back(meanCurvature.clone());

	// Sparse & Grid
	meanLength = cv::MatND::zeros(1, 10, CV_64FC1);
	meanCurvature = cv::MatND::zeros(1, 10, CV_64FC1);
	meanLength.at<double>(0, 4) = 1.0f;
	meanCurvature.at<double>(0, 0) = 1.0f;

	meanLengths.push_back(meanLength.clone());
	meanCurvatures.push_back(meanCurvature.clone());
}

void KMeans::add(cv::MatND& histLength, cv::MatND& histCurvature, int x, int y) {
	histLengths.push_back(histLength);
	histCurvatures.push_back(histCurvature);
	locations.push_back(QVector2D(x, y));
}

float KMeans::distance(cv::MatND& hist1, cv::MatND& hist2) {
	return cv::compareHist(hist1, hist2, 1);
}

void KMeans::cluster() {
	groups.clear();

	for (int i = 0; i < histLengths.size(); i++) {
		groups.push_back(-1);
	}

	bool updated = true;
	while (updated) {
		updated = false;
		for (int i = 0; i < histLengths.size(); i++) {
			// find the closest mean
			float min_dist = std::numeric_limits<float>::max();
			int mean_id;
			for (int j = 0; j < meanLengths.size(); j++) {
				float dist1 = distance(histLengths[i], meanLengths[j]);
				float dist2 = distance(histCurvatures[i], meanCurvatures[j]);
				float dist = sqrtf(dist1 * dist1 + dist2 * dist2);
				if (dist < min_dist) {
					min_dist = dist;
					mean_id = j;
				}
			}

			// update the group id
			if (groups[i] != mean_id) {
				groups[i] = mean_id;
				updated = true;
			}
		}

		// clear the means
		for (int i = 0; i < meanLengths.size(); i++) {
			meanLengths[i] = cv::MatND(meanLengths[i].rows, meanLengths[i].cols, CV_32F);
			meanCurvatures[i] = cv::MatND(meanCurvatures[i].rows, meanCurvatures[i].cols, CV_32F);
		}

		// update the means
		std::vector<int> count;
		for (int i = 0; i < meanLengths.size(); i++) {
			count.push_back(0);
		}
		for (int i = 0; i < histLengths.size(); i++) {
			meanLengths[groups[i]] += histLengths[i];
			meanCurvatures[groups[i]] += histCurvatures[i];
			count[groups[i]]++;
		}
		for (int i = 0; i < meanLengths.size(); i++) {
			meanLengths[i] /= count[i];
			meanCurvatures[i] /= count[i];
		}
	}
}

cv::Mat_<uchar> KMeans::getSegmentation() {
	cv::Mat_<uchar> ret(4, 4);

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			ret(i, j) = groups[i * 4 + j];
		}
	}

	return ret;
}
