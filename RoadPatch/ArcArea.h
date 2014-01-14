#pragma once

#include "AbstractArea.h"
#include "BBox.h"
#include <boost/shared_ptr.hpp>

class ArcArea : public AbstractArea {
public:
	QVector2D leftPt;
	QVector2D rightPt;
	float radius;
	float arc_len;

public:
	ArcArea(const QVector2D& leftPt, const QVector2D& rightPt, float radius, float arc_len);
	ArcArea(const AbstractArea& ref, int mode);
	~ArcArea();

	bool contains(const QVector2D& pt) const;
	QVector2D midPt() const;
	float dx() const;
	float dy() const;
	void translate(float x, float y);
	void resize(const QVector2D& pt);
	bool hitTest(const QVector2D& pt) const;
	bool hitTestResizingPoint(const QVector2D& pt) const;
	std::vector<QVector2D> polyline() const;
	QVector2D deform(const QVector2D& pt) const;
};

typedef boost::shared_ptr<ArcArea> ArcAreaPtr;