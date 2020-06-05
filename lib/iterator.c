#include "iterator.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

static int getOffset(int n, int *idx, int *strides) {
	int res = 0;
	for (int i = 0; i < n; ++i) {
		res += idx[i] * strides[i];
	}
	return res;
}

Iterator getIterator(Tensor tensor) {
	assert(tensor.valid);
	Iterator it;
	it.shape = tensor.shape;
	it.data = tensor.data;
	it.typesize = getTypeSize(tensor.dtype);
	memset(it.curidx, 0, it.shape.ndim * sizeof(int));
	return it;
}

void *begin(Iterator *it) {
	memset(it->curidx, 0, it->shape.ndim * sizeof(int));
	return it->data;
}
void *end(Iterator *it) {
	int ndim = it->shape.ndim;
	if (ndim == 0) return it->data + it->typesize;
	
	int idxs[4];
	memset(idxs, 0, it->shape.ndim * sizeof(int));
	idxs[0] = it->shape.dims[0];
	return it->data + getOffset(ndim, idxs, it->shape.strides) * it->typesize;
}
void *next(Iterator *it) {
	int ndim = it->shape.ndim;
	if (ndim == 0) return it->data + it->typesize;

	int i = ndim - 1;
	int *dims = it->shape.dims;
	++it->curidx[i];
	while (i > 0 && it->curidx[i] >= dims[i]) {
		it->curidx[i] = 0;
		++(it->curidx[--i]); 	
	}
	return it->data + getOffset(ndim, it->curidx, it->shape.strides) * it->typesize;
}
