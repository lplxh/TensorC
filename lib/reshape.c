#include "tensor.h"
#include "iterator.h"
#include <stdbool.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

bool isContiguous(Shape shape) {
	int dim = shape.ndim;
	if (dim == 0) return true;
	if (shape.strides[dim-1] != 1) return false;
	for (int i = dim-2; i >= 0; --i) {
		if (shape.strides[i] != shape.strides[i+1] * shape.dims[i+1]) return false;
	}
	return true;
}

Tensor ravel(Tensor *this) {
	assert(this->valid);
	Shape shape = this->shape;
	if (isContiguous(shape)) return *this;
	int elesize = getTypeSize(this->dtype);
	Tensor res;
	res.valid = true;
	res.dtype = this->dtype;
	res.shape.ndim = 1;
	res.shape.nelem = shape.nelem;
	res.shape.dims[0] = shape.nelem;
	res.shape.strides[0] = 1;
	res.data = malloc(shape.nelem * elesize);
	Iterator it = getIterator(*this);
	void *ib = begin(&it);
	for (int i = 0, n = shape.nelem; i < n; ++i, ib = next(&it)) {
		memcpy(res.data+i*elesize, ib, elesize);
	}
	return res;
}

Tensor ravel_out(Tensor *output,Tensor*this) {
	assert(this->valid);
	Shape shape = this->shape;
	if (isContiguous(shape)) return *this;
	int elesize = getTypeSize(this->dtype);
	output->valid = true;
	output->dtype = this->dtype;
	output->shape.ndim = 1;
	output->shape.nelem = shape.nelem;
	output->shape.dims[0] = shape.nelem;
	output->shape.strides[0] = 1;
	output->data = malloc(shape.nelem * elesize);
	Iterator it = getIterator(*this);
	void *ib = begin(&it);
	for (int i = 0, n = shape.nelem; i < n; ++i, ib = next(&it)) {
		memcpy(output->data+i*elesize, ib, elesize);
	}
	return *output;
}

Tensor reshapeFromShape(Tensor *this, Shape newshape) {
	assert(this->valid);
	Tensor ret;
	ret.valid = true;
	ret.shape = newshape;
	ret.dtype = this->dtype;

	// If data is stored contiguously, then reuse data. Otherwise, reallocate space and copy data
	if (isContiguous(this->shape)) {
		ret.data = this->data;
	} else {
		Tensor tmp = ravel(this);
		ret.data = tmp.data;
	}
	return ret;
}
Tensor reshape_internel(Tensor *this, int n, int *dims) {
	assert(this->valid);
	Shape newshape;
	newshape.ndim = n;
	if (n == 0) return *this;
	memcpy(newshape.dims, dims, n * sizeof(int));
	int count = 1;
	for (int i = 0; i < n; ++i) count *= dims[i];
	newshape.nelem = count;
	if (count != this->shape.nelem) {
		printf("Number of elements should be same after reshape\n");
		exit(0);
	}
	computeStride(&newshape);
	return reshapeFromShape(this, newshape);
}
Tensor reshape(Tensor *this, int n, ...) {
	assert(this->valid);
	va_list vl;
	va_start(vl, n);
	Shape newshape = allocShapeFromValist(n, vl);
	va_end(vl);
	if (newshape.nelem != this->shape.nelem) {
		printf("Number of elements should be same after reshape\n");
		exit(0);
	}
	
	return reshapeFromShape(this, newshape);
}

Tensor reshape_out(Tensor*output,Tensor *this, int n, ...) {
	assert(this->valid);
	va_list vl;
	va_start(vl, n);
	Shape newshape = allocShapeFromValist(n, vl);
	va_end(vl);
	if (newshape.nelem != this->shape.nelem) {
		printf("Number of elements should be same after reshape\n");
		exit(0);
	}
	*output = reshapeFromShape(this, newshape);
	return *output;
}

Tensor transpose(Tensor *this) {
	assert(this->valid);
	int dim = this->shape.ndim;
	if (dim < 2) {
		printf("Error: Rank of tensor should not be less than 2\n");
		exit(0);
	}
	
	// share same data
	Tensor res;
	res.valid = true;
	res.data = this->data;
	res.dtype = this->dtype;
	res.shape = this->shape;

	// set size and stride
	for (int i = dim-1; i >= 0; --i) {
		res.shape.dims[i] = this->shape.dims[dim-i-1];
		res.shape.strides[i] = this->shape.strides[dim-i-1];
	}
	
	return res; 
}

Tensor transpose_out(Tensor*output,Tensor *this) {
	assert(this->valid);
	int dim = this->shape.ndim;
	if (dim < 2) {
		printf("Error: Rank of tensor should not be less than 2\n");
		exit(0);
	}
	
	// share same data
	output->valid = true;
	output->data = this->data;
	output->dtype = this->dtype;
	output->shape = this->shape;

	// set size and stride
	for (int i = dim-1; i >= 0; --i) {
		output->shape.dims[i] = this->shape.dims[dim-i-1];
		output->shape.strides[i] = this->shape.strides[dim-i-1];
	}
	
	return *output; 
}

static int *getMissingAxes(int ndim, int *order, int n) {
	int *res = (int *)malloc(ndim * sizeof(int));
	if (n >= ndim) {
		memcpy(res, order, ndim * sizeof(int));
		return res;
	}
	int *axes = (int *)malloc(ndim * sizeof(int));
	memset(axes, 0, ndim * sizeof(int));
	for (int i = 0; i < n; ++i) axes[order[i]] = -1;
	memcpy(res, order, n * sizeof(int));
	for (int i = 0; i < ndim; ++i) {
		if (axes[i] == -1) continue;
		res[n++] = i;
	}
	free(axes);
	return res;
}
Tensor permute_internel(Tensor *this, int *order) {
	assert(this->valid);
	Tensor res;
	res.valid = true;
	res.data = this->data;
	res.dtype = this->dtype;
	res.shape.ndim = this->shape.ndim;
	res.shape.nelem = this->shape.nelem;

	for (int i = 0, n = this->shape.ndim; i < n; ++i) {
		res.shape.dims[i] = this->shape.dims[order[i]];
		res.shape.strides[i] = this->shape.strides[order[i]];
	}
	return res;
}

Tensor permute(Tensor *this, int n, ...) {
	assert(this->valid);
	int dim = this->shape.ndim;
	va_list vl;
	va_start(vl, n);
	int *axes = (int *)malloc(dim * sizeof(int));
	for (int i = 0; i < n; ++i) {
		int axis = va_arg(vl, int);
		if (axis < 0 || axis >= dim) {
			printf("Axis arg error\n");
			exit(0);
		}
		axes[i] = axis;
	}
	int *order = getMissingAxes(dim, axes, n);
	Tensor res = permute_internel(this, order);
	free(axes);
	free(order);
	return res;
}

Tensor permute_out(Tensor*output,Tensor *this, int n, ...) {
	assert(this->valid);
	int dim = this->shape.ndim;
	va_list vl;
	va_start(vl, n);
	int *axes = (int *)malloc(dim * sizeof(int));
	for (int i = 0; i < n; ++i) {
		int axis = va_arg(vl, int);
		if (axis < 0 || axis >= dim) {
			printf("Axis arg error\n");
			exit(0);
		}
		axes[i] = axis;
	}
	int *order = getMissingAxes(dim, axes, n);
	*output = permute_internel(this, order);
	free(axes);
	free(order);
	return *output;
}

Tensor ipermute_internel(Tensor *this, int *order) {
	assert(this->valid);
	int dim = this->shape.ndim;
	int *inverseOrder = (int *)malloc(dim * sizeof(int));
	for (int i = 0; i < dim; ++i) {
		inverseOrder[order[i]] = i;
	}
	return permute_internel(this, inverseOrder);
}
