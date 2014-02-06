/**
 * 道路網をパッチに分割し、画像、GSMファイル、特徴量ファイルとして保存する。
 *
 * @author Gen Nishida
 * @version 1.0
 */

#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "Polygon2D.h"
#include "GraphUtil.h"
#include "RoadSegmentationUtil.h"

int main(int argc, char *argv[]) {
	if (argc < 4) {
		std::cerr << "Usage: RoadPatch <road GSM file> <original size> <patch size>" << std::endl;
		return 1;
	}
	
	QString filename = argv[1];
	int originalSize = atoi(argv[2]);
	int patchSize = atoi(argv[3]);

	// ファイル名のベースパートを取得
	QString basename = filename.split("\\").last().split(".").at(0);

	// 道路網をGSMファイルから読み込む
	RoadGraph r;
	GraphUtil::loadRoads(r, filename);

	// 指定されたセルサイズに分割する
	int count = 0;
	for (int y = -originalSize / 2; y <= originalSize / 2 - patchSize; y += patchSize / 2) {
		for (int x = -originalSize / 2; x <= originalSize / 2 - patchSize; x += patchSize / 2) {
			Polygon2D area = Polygon2D::createRectangle(QVector2D(x, y), QVector2D(x + patchSize, y + patchSize));

			// パッチの範囲の道路網を抽出
			RoadGraph patch;
			GraphUtil::copyRoads(r, patch, area, false);

			// 道路網が、直交座標系の第一象限に位置するよう、移動する
			QVector2D offset(-x, -y);
			GraphUtil::translate(patch, offset);

			// cv::Matを作成
			cv::Mat_<uchar> mat(patchSize, patchSize);
			GraphUtil::convertToMat(patch, mat, mat.size());

			// 1/5に縮小
			cv::resize(mat, mat, cv::Size(), 0.2, 0.2, cv::INTER_CUBIC);

			// 画像として保存
			cv::imwrite((basename + "_%1.jpg").arg(count).toUtf8().data(), mat);

			// GSMファイルとして保存
			GraphUtil::saveRoads(patch, (basename + "_%1.gsm").arg(count));

			// 特徴量を取得
			std::vector<RadialFeature> radialFeatures;
			RoadSegmentationUtil::detectRadial(patch, area, 3, radialFeatures, 2, 0.05, 0.1, 80, 0.4, 0.2, 80, 0.2, 150, 0.7, 80, 6, 0.1);
			if (radialFeatures.size() > 0) {
				radialFeatures[0].save((basename + "_%1_radial_feature.xml").arg(count));
			}
			std::vector<GridFeature> gridFeatures;
			RoadSegmentationUtil::detectGrid(patch, area, 2, gridFeatures, 10, 9, 3000, 0.5, 0.1, 0.7, 20, 300);
			if (gridFeatures.size() > 0) {
				gridFeatures[0].save((basename + "_%1_grid_feature.xml").arg(count));
			}
			std::vector<GenericFeature> genericFeatures;
			RoadSegmentationUtil::extractGenericFeature(patch, area, genericFeatures);
			if (genericFeatures.size() > 0) {
				genericFeatures[0].save((basename + "_%1_generic_feature.xml").arg(count));
			}

			count++;
		}
	}
}