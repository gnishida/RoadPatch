﻿#pragma once

#include "Polygon2D.h"
#include <QDomNode>
#include <QColor>
#include <QVector2D>

class RadialFeature {
public:
	int group_id;
	QVector2D center;
	std::vector<float> radii;
	int numDirections;
	
	Polygon2D _polygon;			// 領域を表すポリゴン（時計回り）

public:
	RadialFeature(int group_id) : group_id(group_id), numDirections(0) {}
	~RadialFeature() {}

	void load(QString filename);
	void load(QDomNode& node);
	void save(QString filename);

	QColor color();
	Polygon2D polygon();
};

