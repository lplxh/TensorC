#include "shape.h"
#include <stdlib.h>
#include <stdio.h>

void computeStride(Shape *s) {
	int dim = s->ndim;
	if (dim == 0) return;
	s->strides[dim-1] = 1;
	for (int i = dim-2; i >= 0; --i) {
		s->strides[i] = s->strides[i+1] * s->dims[i+1];
	}
}

Shape allocShapeFromValist(int n, va_list vl) {
	Shape res;
	res.ndim = n;
	if (n < 0) {
		printf("The number of valist should not be negative\n");
		exit(0);
	}
	int count = 1;
	for (int i = 0; i < n; ++i) {
		int dim = va_arg(vl, int);
		res.dims[i] = dim;
		count *= dim;
	}
	res.nelem = count;
	va_end(vl);
	computeStride(&res);
	return res;
}

Shape allocShape(int n, ...) {
	va_list vl;
	va_start(vl, n);
	Shape res = allocShapeFromValist(n, vl);
	va_end(vl);
	return res;
}

Shape cloneShapeExceptStride(Shape shape) {
	Shape res = shape;
	computeStride(&res);
	return res;
}

void ignoreOneLenDim(Shape *shape) {
	if (shape->nelem == 0 || shape->ndim == 0) return;
	int ndim = shape->ndim;
	Shape oldshape = *shape;
	int j = 0;
	for (int i = 0; i < ndim; ++i) {
		if (oldshape.dims[i] == 1) continue;
		shape->dims[j] = oldshape.dims[i];
		shape->strides[j++] = oldshape.strides[i];
	}
	shape->ndim = j;
}

bool isSameShape(Shape a, Shape b) {
	if (a.ndim != b.ndim || a.nelem != b.nelem) return false;
	for (int i = 0, n = a.ndim; i < n; ++i) {
		if (a.dims[i] != b.dims[i]) return false;
	}
	return true;
}

void dumpShape(Shape shape) {
	printf("ndim: %d\n", shape.ndim);
	printf("nelem: %d\n", shape.nelem);
	for (int i = 0; i < shape.ndim; ++i) {
		printf("dims: %d, stride: %d\n", shape.dims[i], shape.strides[i]);
	}
}
