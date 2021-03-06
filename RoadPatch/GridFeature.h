﻿#pragma once

#include "Polygon2D.h"
#include <QMap>
#include <QDomNode>
#include <QVector2D>
#include <QColor>
#include <opencv/cv.h>
#include <opencv/highgui.h>

class GridFeature {
public:
	int group_id;
	float angle1;				// 横方向の角度（第一象限）
	float angle2;				// 縦方向の角度（第二象限）
	QMap<float, float> length1;	// 横方向の長さのヒストグラム
	QMap<float, float> length2;	// 縦方向の長さのヒストグラム

	int accmDirCount;			// 累積ベクトル用カウンタ
	QVector2D accmDir1;			// 横方向の累積ベクトル
	QVector2D accmDir2;			// 縦方向の累積ベクトル

	int accmLenCount1;			// 横方向のエッジ長の累積カウンタ
	int accmLenCount2;			// 縦方向のエッジ長の累積カウンタ

	Polygon2D _polygon;			// 領域を表すポリゴン（時計回り）

public:
	GridFeature(int group_id) : group_id(group_id) {}
	~GridFeature() {}

	void setAngle(float angle);
	void addEdge(const QVector2D& edge_vec, float threshold);
	void computeFeature();
	bool isClose(const QVector2D& dir, float threshold);

	void load(QString filename);
	void load(QDomNode& node);
	void save(QString filename);

	QColor color();
	Polygon2D polygon();
};

