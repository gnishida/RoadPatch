﻿#pragma once

#include <QVector3D>
#include <QVector2D>

class Util {
	static const float MTC_FLOAT_TOL;

protected:
	Util();

public:
	static float pointSegmentDistanceXY(const QVector3D &a, const QVector3D &b, const QVector3D &c);
	//static QVector3D projLatLonToMeter(const QVector3D &latLon, const QVector3D &centerLatLon);
	static QVector2D projLatLonToMeter(const QVector2D &latLon, const QVector2D &centerLatLon);

	static bool segmentSegmentIntersectXY(const QVector2D& a, const QVector2D& b, const QVector2D& c, const QVector2D& d, float *tab, float *tcd, bool segmentOnly, QVector2D &intPoint);
	static float pointSegmentDistanceXY(const QVector2D& a, const QVector2D& b, const QVector2D& c, QVector2D& closestPtInAB);

	// 角度関係
	static float rad2deg(float rad);
	static float normalizeAngle(float angle);
	static float diffAngle(const QVector2D& dir1, const QVector2D& dir2, bool absolute = true);
	static float diffAngle(float angle1, float angle2, bool absolute = true);

	// 乱数関係
	static float uniform_rand();
	static float uniform_rand(float a, float b);
};

