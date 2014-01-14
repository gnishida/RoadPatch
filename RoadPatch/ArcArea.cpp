#include "ArcArea.h"
//#include "RoadGraphEditor.h"

#ifndef M_PI
#define M_PI	3.14159265359
#endif

ArcArea::ArcArea(const QVector2D& leftPt, const QVector2D& rightPt, float radius, float arc_len) {
	this->leftPt = leftPt;
	this->rightPt = rightPt;
	this->radius = radius;
	this->arc_len = arc_len;
}

ArcArea::ArcArea(const AbstractArea& ref, int mode) {
	leftPt.setX(ref.midPt().x() - ref.dx() / 2.0f);
	leftPt.setY(ref.midPt().y());
	rightPt.setX(ref.midPt().x() + ref.dx() / 2.0f);
	rightPt.setY(ref.midPt().y());

	arc_len = ref.dy();

	/*
	switch (mode) {
	case RoadGraphEditor::MODE_BASIC_AREA_DISTORTING_TL:
		radius = 10000.0f;
		resizingType = AbstractArea::RESIZING_TOP_LEFT;
		break;
	case RoadGraphEditor::MODE_BASIC_AREA_DISTORTING_TR:
		radius = -10000.0f;
		resizingType = AbstractArea::RESIZING_TOP_RIGHT;
		break;
	case RoadGraphEditor::MODE_BASIC_AREA_DISTORTING_BL:
		radius = 10000.0f;
		resizingType = AbstractArea::RESIZING_BOTTOM_LEFT;
		break;
	case RoadGraphEditor::MODE_BASIC_AREA_DISTORTING_BR:
		radius = -10000.0f;
		resizingType = AbstractArea::RESIZING_BOTTOM_RIGHT;
		break;
	}*/
}

ArcArea::~ArcArea() {
}

bool ArcArea::contains(const QVector2D& pt) const {
	// compute the center of the arc
	QVector2D center = (leftPt + rightPt) / 2.0f;
	center.setX(center.x() + radius);

	// compute the width
	float width = rightPt.x() - leftPt.x();

	// compute the radian
	float rad = arc_len / fabs(radius);

	QVector2D vec = pt - center;

	if (vec.length() < fabs(radius) - width / 2.0f) return false;
	if (vec.length() > fabs(radius) + width / 2.0f) return false;

	float angle = atan2f(vec.y(), vec.x());
	if (radius >= 0) {
		if (angle >= M_PI - rad / 2.0f || angle <= -M_PI + rad / 2.0f) return true;
	} else {
		if (angle >= -rad / 2.0f && angle <= rad / 2.0f) return true;
	}

	return false;
}

QVector2D ArcArea::midPt() const {
	return (leftPt + rightPt) / 2.0f;
}

float ArcArea::dx() const {
	return rightPt.x() - leftPt.x();
}

float ArcArea::dy() const {
	return arc_len;
}

void ArcArea::translate(float x, float y) {
	leftPt.setX(leftPt.x() + x);
	leftPt.setY(leftPt.y() + y);
	rightPt.setX(rightPt.x() + x);
	rightPt.setY(rightPt.y() + y);
}

void ArcArea::resize(const QVector2D& pt) {
	// compute the width
	float width = rightPt.x() - leftPt.x();

	switch (resizingType) {
	case RESIZING_TOP_LEFT:
		radius = width * arc_len / ((pt.y() - leftPt.y()) * 2.0f - arc_len) + width * 0.5f;
		break;
	case RESIZING_TOP_RIGHT:
		radius = -width * arc_len / ((pt.y() - rightPt.y()) * 2.0f - arc_len) - width * 0.5f;
		break;
	case RESIZING_BOTTOM_LEFT:
		radius = width * arc_len / ((leftPt.y() - pt.y()) * 2.0f - arc_len) + width * 0.5f;
		break;
	case RESIZING_BOTTOM_RIGHT:
		radius = -width * arc_len / ((rightPt.y() - pt.y()) * 2.0f - arc_len) - width * 0.5f;
		break;
	}
}

bool ArcArea::hitTest(const QVector2D& pt) const {
	// compute the center of the arc
	QVector2D center = (leftPt + rightPt) / 2.0f;
	center.setX(center.x() + radius);

	// compute the width
	float width = rightPt.x() - leftPt.x();

	// compute the radian
	float rad = arc_len / fabs(radius);

	if ((pt - center).length() < radius - width / 2.0f * 1.1f) return false;
	if ((pt - center).length() > radius + width / 2.0f * 1.1f) return false;

	float angle = atan2f((pt - center).y(), (pt - center).x());
	if (angle < -rad / 2.0f * 1.1f || angle > rad / 2.0f * 1.1f) return false;

	return true;
}

bool ArcArea::hitTestResizingPoint(const QVector2D& pt) const {
	// compute the radian
	float rad = arc_len / fabs(radius);

	// compute the width
	float width = rightPt.x() - leftPt.x();

	QVector2D resizingPt = (leftPt + rightPt) / 2.0f;
	resizingPt.setX(resizingPt.x() + radius);
	resizingPt.setX(resizingPt.x() + cosf(-rad / 2.0f) * (radius + width / 2.0f));
	resizingPt.setY(resizingPt.y() + sinf(-rad / 2.0f) * (radius + width / 2.0f));

	if (fabs(pt.x() - resizingPt.x()) < dx() * 0.1f && fabs(pt.y() - resizingPt.y()) < dy() * 0.1f) return true;
	else return false;
}

std::vector<QVector2D> ArcArea::polyline() const {
	std::vector<QVector2D> ret;

	// compute the center of the arc
	QVector2D center = (leftPt + rightPt) / 2.0f;
	center.setX(center.x() + radius);

	// compute the width
	float width = rightPt.x() - leftPt.x();

	// compute the radian
	float rad = arc_len / fabs(radius);

	int n = 8;
	if (radius > 0) {
		for (int i = 0; i <= n; i++) {
			float theta = rad * (float)i / (float)n;
			ret.push_back(QVector2D(center.x() + cosf(rad / 2.0f - theta) * (-radius - width / 2.0f), center.y() + sinf(rad / 2.0f - theta) * (radius + width / 2.0f)));
		}
		for (int i = 0; i <= n; i++) {
			float theta = rad * (float)i / (float)n;
			ret.push_back(QVector2D(center.x() + cosf(-rad / 2.0f + theta) * (-radius + width / 2.0f), center.y() + sinf(-rad / 2.0f + theta) * (radius - width / 2.0f)));
		}
	} else {
		for (int i = 0; i <= n; i++) {
			float theta = rad * (float)i / (float)n;
			ret.push_back(QVector2D(center.x() + cosf(rad / 2.0f - theta) * (-radius - width / 2.0f), center.y() + sinf(rad / 2.0f - theta) * (-radius - width / 2.0f)));
		}
		for (int i = 0; i <= n; i++) {
			float theta = rad * (float)i / (float)n;
			ret.push_back(QVector2D(center.x() + cosf(-rad / 2.0f + theta) * (-radius + width / 2.0f), center.y() + sinf(-rad / 2.0f + theta) * (-radius + width / 2.0f)));
		}
	}

	return ret;
}

/**
 * 与えられた点を、このArcAreaに基づいて変形する。
 */
QVector2D ArcArea::deform(const QVector2D& pt) const {
	// compute the center of the arc
	QVector2D center = (leftPt + rightPt) / 2.0f;
	QVector2D vec = pt - center;
	center.setX(center.x() + radius);

	float theta0 = arc_len / fabs(radius);

	float theta = vec.y() / arc_len * theta0;
	float r = radius - vec.x();

	QVector2D ret;
	ret.setX(center.x() - r * cos(theta));
	ret.setY(center.y() + fabs(r) * sin(theta));

	return ret;
}

