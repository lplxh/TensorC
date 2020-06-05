#ifndef SHAPE_SHAPE_H
#define SHAPE_SHAPE_H

#include <stdarg.h>
#include <stdbool.h>

typedef struct Shape Shape;
struct Shape {
	int ndim;
	int nelem;
	int dims[4];
	int strides[4];
};

Shape allocShape(int n, ...);
Shape allocShapeFromValist(int n, va_list vl);
void computeStride(Shape *shape);
Shape cloneShapeExceptStride(Shape shape);
bool isSameShape(Shape a, Shape b);

void dumpShape(Shape shape);

#endif
