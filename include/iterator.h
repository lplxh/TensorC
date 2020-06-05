#ifndef ITERATOR_H
#define ITERATOR_H

#include "tensor.h"
#include "shape.h"

typedef struct Iterator Iterator;
struct Iterator {
	Shape shape;
	int curidx[4];
	void *data;
	int typesize;
};

Iterator getIterator(Tensor tensor);
void *begin(Iterator *it);
void *end(Iterator *it);
void *next(Iterator *it);

#endif
