#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

Tensor select(Tensor *this, unsigned dim, int64_t index) {
	assert(this->valid);
	Tensor ret;
	ret.valid = true;
	ret.dtype = this->dtype;
	int ndim = this->shape.ndim;
	if (dim >= ndim) {
		printf("Error: The input dim is illegal\n");
		exit(0);
	}
	if (index >= this->shape.dims[dim] || index < -this->shape.dims[dim]) {
		printf("Error: The input index is illegal\n");
		exit(0);
	} 
	if (index < 0) index += this->shape.dims[dim];
	ret.data = this->data + index * this->shape.strides[dim] * getTypeSize(this->dtype);
	ret.shape.ndim = ndim - 1;
	int i = 0;
	if (dim > 0) {
		memcpy(ret.shape.dims, this->shape.dims, dim*sizeof(int));
		memcpy(ret.shape.strides, this->shape.strides, dim*sizeof(int));
		i += dim;
	}
	if (dim < ndim-1) {
		memcpy(ret.shape.dims+i, this->shape.dims+dim+1, (ndim-dim-1)*sizeof(int));
		memcpy(ret.shape.strides+i, this->shape.strides+dim+1, (ndim-dim-1)*sizeof(int));
	}
	ret.shape.nelem = this->shape.nelem / this->shape.dims[dim];
	return ret;
}
Tensor slice(Tensor *this, unsigned dim, int64_t start, int64_t step, int64_t end) {
	assert(this->valid);
	Tensor ret;
	ret.valid = true;
	ret.dtype = this->dtype;
	ret.shape = this->shape;
	int ndim = this->shape.ndim;
	ret.shape.ndim = this->shape.ndim;
	if (dim >= ndim) {
		printf("Error: The input dim is illegal\n");
		exit(0);
	}

	// compute shape and stride info
	int len = this->shape.dims[dim];
	if (start < 0) start += len;
	if (end < 0) end += len;
	if (end < start || start >= len) {
		ret.shape.nelem = 0;
		return ret;
	}
	end = (end >= len) ? len-1 : end;
	ret.shape.dims[dim] = (end - start + step) / step;
	ret.shape.strides[dim] = this->shape.strides[dim] * step;

	// compute element count
	int count = 1, offset = 0;
	for (int i = 0; i < ndim; ++i) {
		count *= ret.shape.dims[i];
	}
	ret.shape.nelem = count;
	ret.data = this->data + this->shape.strides[dim] * start * getTypeSize(this->dtype);
	return ret;
}
