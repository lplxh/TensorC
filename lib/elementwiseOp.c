#include "tensor.h"
#include "iterator.h"
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <complex.h>

void computeStrideForBroadcast(Shape *a, Shape *b, Shape res) {
	int dim = res.ndim, dimA = a->ndim, dimB = b->ndim;
	for (int i = dim-1; i >= 0; --i) {
		int j = i-dim+dimA, k = i-dim+dimB;
		a->strides[i] = (j < 0 || a->dims[j] == 1) ? 0 : a->strides[j];
		b->strides[i] = (k < 0 || b->dims[k] == 1) ? 0 : b->strides[k];
	}
	memcpy(a->dims, res.dims, dim * sizeof(int));
	memcpy(b->dims, res.dims, dim * sizeof(int));
	a->ndim = b->ndim = dim;
	a->nelem = b->nelem = res.nelem;
}

bool computeBroadCastShape(Shape *a, Shape *b, Shape *res) {
	int dimA = a->ndim, dimB = b->ndim;
	int dim = dimA > dimB ? dimA : dimB;
	res->ndim = dim;
	if (dim == 0) {
		res->nelem = 0;
		return true;
	}
	// compute shape after broadcast
	int count = 1;
	for (int i = dimA-1, j = dimB-1, k = dim-1; k >= 0; --k) {
		if (i < 0) res->dims[k] = b->dims[j--];
		else if (j < 0) res->dims[k] = a->dims[i--];
		else {
			if (a->dims[i] == b->dims[j])
				res->dims[k] = a->dims[i];
			else if (a->dims[i] == 1 || b->dims[j] == 1)
				res->dims[k] = (a->dims[i] > b->dims[j]) ? a->dims[i] : b->dims[j];
			else 
				return false;
			--i;
			--j;
		}
		count *= res->dims[k];
	}
	res->nelem = count;
	computeStrideForBroadcast(a, b, *res);
	return true;
}

Tensor EWAdd(const Tensor a, const Tensor b) {
	assert(a.valid && b.valid);
	// compute common type (promote)
	// If common type is different from original type, do cast
	Tensor lhs = a, rhs = b;
	ScalarType type = castToCommonType(&lhs, &rhs);
	
	// Get shape after broadcasting
	Shape shapeRes;
	bool validShape = computeBroadCastShape(&(lhs.shape), &(rhs.shape), &shapeRes);
	if (!validShape) { // invalid shape
		printf("Error: Shapes of two tensor don't match\n");
		exit(0);
	}
	
	// do element-wise operation
	Tensor res = zerosFromShape(type, shapeRes);

	Iterator itA = getIterator(lhs), itB = getIterator(rhs);
	switch (type) {
		case DOUBLE: {
			double *ptr = (double *)res.data, *ibA = (double *)begin(&itA), *ibB = (double *)begin(&itB);
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ibA = next(&itA), ibB = next(&itB)) {
				*ptr++ = (*ibA) + (*ibB);
			}
			break;
		}
		case FLOAT: {
			float *ptr = (float *)res.data, *ibA = (float *)begin(&itA), *ibB = (float *)begin(&itB);
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ibA = next(&itA), ibB = next(&itB)) {
				*ptr++ = (*ibA) + (*ibB);
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			double complex *ptr = (double complex *)res.data, *ibA = (double complex *)begin(&itA), *ibB = (double complex *)begin(&itB);
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ibA = next(&itA), ibB = next(&itB)) {
				*ptr++ = (*ibA) + (*ibB);
			}
			break;
		}
		case INT: {
			int *ptr = (int *)res.data, *ibA = (int *)begin(&itA), *ibB = (int *)begin(&itB);
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ibA = next(&itA), ibB = next(&itB)) {
				*ptr++ = (*ibA) + (*ibB);
			}
			break;
		}
		default: {
			printf("Unsupported data type\n");
			break;
		}
	}
	return res;
}

Tensor EWAdd_out(Tensor*output,const Tensor a, const Tensor b) {
	assert(a.valid && b.valid);
	// compute common type (promote)
	// If common type is different from original type, do cast
	Tensor lhs = a, rhs = b;
	ScalarType type = castToCommonType(&lhs, &rhs);
	
	// Get shape after broadcasting
	Shape shapeRes;
	bool validShape = computeBroadCastShape(&(lhs.shape), &(rhs.shape), &shapeRes);
	if (!validShape) { // invalid shape
		printf("Error: Shapes of two tensor don't match\n");
		exit(0);
	}
	
	// do element-wise operation

	Iterator itA = getIterator(lhs), itB = getIterator(rhs);
	switch (type) {
		case DOUBLE: {
			double *ptr = (double *)output->data, *ibA = (double *)begin(&itA), *ibB = (double *)begin(&itB);
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ibA = next(&itA), ibB = next(&itB)) {
				*ptr++ = (*ibA) + (*ibB);
			}
			break;
		}
		case FLOAT: {
			float *ptr = (float *)output->data, *ibA = (float *)begin(&itA), *ibB = (float *)begin(&itB);
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ibA = next(&itA), ibB = next(&itB)) {
				*ptr++ = (*ibA) + (*ibB);
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			double complex *ptr = (double complex *)output->data, *ibA = (double complex *)begin(&itA), *ibB = (double complex *)begin(&itB);
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ibA = next(&itA), ibB = next(&itB)) {
				*ptr++ = (*ibA) + (*ibB);
			}
			break;
		}
		case INT: {
			int *ptr = (int *)output->data, *ibA = (int *)begin(&itA), *ibB = (int *)begin(&itB);
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ibA = next(&itA), ibB = next(&itB)) {
				*ptr++ = (*ibA) + (*ibB);
			}
			break;
		}
		default: {
			printf("Unsupported data type\n");
			break;
		}
	}
	return *output;
}

Tensor EWAddTensorScalar(const Tensor a, Scalar num) {
	assert(a.valid);
	Tensor res = zerosFromShape(a.dtype, a.shape);
	Iterator it = getIterator(a);
	switch (a.dtype) {
		case DOUBLE: {
			double *ptr = (double *)res.data, *ib = (double *)begin(&it);
			double num2 = (double)num;
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = *ib + num2;
			}
			break;
		}
		case FLOAT: {
			float *ptr = (float *)res.data, *ib = (float *)begin(&it);
			float num2 = (float)num;
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = *ib + num2;
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			double complex *ptr = (double complex *)res.data, *ib = (double complex *)begin(&it);
			double complex num2 = (double complex)num;
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = *ib + num2;
			}
			break;
		}
		case INT: {
			int *ptr = (int *)res.data, *ib = (int *)begin(&it);
			int num2 = (int)num;
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = *ib + num2;
			}
			break;
		}
		default: {
			printf("Unsupported data type\n");
			break;
		}
	}
	return res;
}

Tensor EWAddTensorScalar_out(Tensor* output,const Tensor a, Scalar num) {
	assert(a.valid);
	Iterator it = getIterator(a);
	switch (a.dtype) {
		case DOUBLE: {
			double *ptr = (double *)output->data, *ib = (double *)begin(&it);
			double num2 = (double)num;
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = *ib + num2;
			}
			break;
		}
		case FLOAT: {
			float *ptr = (float *)output->data, *ib = (float *)begin(&it);
			float num2 = (float)num;
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = *ib + num2;
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			double complex *ptr = (double complex *)output->data, *ib = (double complex *)begin(&it);
			double complex num2 = (double complex)num;
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = *ib + num2;
			}
			break;
		}
		case INT: {
			int *ptr = (int *)output->data, *ib = (int *)begin(&it);
			int num2 = (int)num;
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = *ib + num2;
			}
			break;
		}
		default: {
			printf("Unsupported data type\n");
			break;
		}
	}
	return *output;
}

//sub
Tensor EWSub(const Tensor a, const Tensor b) {
	assert(a.valid && b.valid);
	// compute common type (promote)
	// If common type is different from original type, do cast
	Tensor lhs = a, rhs = b;
	ScalarType type = castToCommonType(&lhs, &rhs);
	
	// Get shape after broadcasting
	Shape shapeRes;
	bool validShape = computeBroadCastShape(&(lhs.shape), &(rhs.shape), &shapeRes);
	if (!validShape) { // invalid shape
		printf("Error: Shapes of two tensor don't match\n");
		exit(0);
	}
	
	// do element-wise operation
	Tensor res = zerosFromShape(type, shapeRes);

	Iterator itA = getIterator(lhs), itB = getIterator(rhs);
	switch (type) {
		case DOUBLE: {
			double *ptr = (double *)res.data, *ibA = (double *)begin(&itA), *ibB = (double *)begin(&itB);
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ibA = next(&itA), ibB = next(&itB)) {
				*ptr++ = (*ibA) - (*ibB);
			}
			break;
		}
		case FLOAT: {
			float *ptr = (float *)res.data, *ibA = (float *)begin(&itA), *ibB = (float *)begin(&itB);
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ibA = next(&itA), ibB = next(&itB)) {
				*ptr++ = (*ibA) - (*ibB);
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			double complex *ptr = (double complex *)res.data, *ibA = (double complex *)begin(&itA), *ibB = (double complex *)begin(&itB);
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ibA = next(&itA), ibB = next(&itB)) {
				*ptr++ = (*ibA) - (*ibB);
			}
			break;
		}
		case INT: {
			int *ptr = (int *)res.data, *ibA = (int *)begin(&itA), *ibB = (int *)begin(&itB);
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ibA = next(&itA), ibB = next(&itB)) {
				*ptr++ = (*ibA) - (*ibB);
			}
			break;
		}
		default: {
			printf("Unsupported data type\n");
			break;
		}
	}
	return res;
}

Tensor EWSub_out(Tensor*output,const Tensor a, const Tensor b) {
	assert(a.valid && b.valid);
	// compute common type (promote)
	// If common type is different from original type, do cast
	Tensor lhs = a, rhs = b;
	ScalarType type = castToCommonType(&lhs, &rhs);
	
	// Get shape after broadcasting
	Shape shapeRes;
	bool validShape = computeBroadCastShape(&(lhs.shape), &(rhs.shape), &shapeRes);
	if (!validShape) { // invalid shape
		printf("Error: Shapes of two tensor don't match\n");
		exit(0);
	}
	
	// do element-wise operation

	Iterator itA = getIterator(lhs), itB = getIterator(rhs);
	switch (type) {
		case DOUBLE: {
			double *ptr = (double *)output->data, *ibA = (double *)begin(&itA), *ibB = (double *)begin(&itB);
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ibA = next(&itA), ibB = next(&itB)) {
				*ptr++ = (*ibA) - (*ibB);
			}
			break;
		}
		case FLOAT: {
			float *ptr = (float *)output->data, *ibA = (float *)begin(&itA), *ibB = (float *)begin(&itB);
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ibA = next(&itA), ibB = next(&itB)) {
				*ptr++ = (*ibA) - (*ibB);
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			double complex *ptr = (double complex *)output->data, *ibA = (double complex *)begin(&itA), *ibB = (double complex *)begin(&itB);
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ibA = next(&itA), ibB = next(&itB)) {
				*ptr++ = (*ibA) - (*ibB);
			}
			break;
		}
		case INT: {
			int *ptr = (int *)output->data, *ibA = (int *)begin(&itA), *ibB = (int *)begin(&itB);
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ibA = next(&itA), ibB = next(&itB)) {
				*ptr++ = (*ibA) - (*ibB);
			}
			break;
		}
		default: {
			printf("Unsupported data type\n");
			break;
		}
	}
	return *output;
}

Tensor EWSubTensorScalar(const Tensor lhs, Scalar num) {
	assert(lhs.valid);
	Tensor res = zerosFromShape(lhs.dtype, lhs.shape);
	Iterator it = getIterator(lhs);
	switch (lhs.dtype) {
		case DOUBLE: {
			double *ptr = (double *)res.data, *ib = (double *)begin(&it);
			double num2 = (double)num;
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = *ib - num2;
			}
			break;
		}
		case FLOAT: {
			float *ptr = (float *)res.data, *ib = (float *)begin(&it);
			float num2 = (float)num;
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = *ib - num2;
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			double complex *ptr = (double complex *)res.data, *ib = (double complex *)begin(&it);
			double complex num2 = (double complex)num;
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = *ib - num2;
			}
			break;
		}
		case INT: {
			int *ptr = (int *)res.data, *ib = (int *)begin(&it);
			int num2 = (int)num;
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = *ib - num2;
			}
			break;
		}
		default: {
			printf("Unsupported data type\n");
			break;
		}
	}
	return res;
}

Tensor EWSubTensorScalar_out(Tensor*output,const Tensor lhs, Scalar num) {
	assert(lhs.valid);
	Iterator it = getIterator(lhs);
	switch (lhs.dtype) {
		case DOUBLE: {
			double *ptr = (double *)output->data, *ib = (double *)begin(&it);
			double num2 = (double)num;
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = *ib - num2;
			}
			break;
		}
		case FLOAT: {
			float *ptr = (float *)output->data, *ib = (float *)begin(&it);
			float num2 = (float)num;
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = *ib - num2;
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			double complex *ptr = (double complex *)output->data, *ib = (double complex *)begin(&it);
			double complex num2 = (double complex)num;
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = *ib - num2;
			}
			break;
		}
		case INT: {
			int *ptr = (int *)output->data, *ib = (int *)begin(&it);
			int num2 = (int)num;
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = *ib - num2;
			}
			break;
		}
		default: {
			printf("Unsupported data type\n");
			break;
		}
	}
	return *output;
}
Tensor EWSubScalarTensor(Scalar num, const Tensor rhs) {
	assert(rhs.valid);
	return EWAddTensorScalar(EWNegative(rhs), num);
}


Tensor EWAddScalarTensor(Scalar num, const Tensor a) {
	return EWAddTensorScalar(a, num);
}

//mul scalar
Tensor EWMulTensorScalar(const Tensor a, Scalar num) {
	assert(a.valid);
	Tensor res = zerosFromShape(a.dtype, a.shape);
	Iterator it = getIterator(a);
	switch (a.dtype) {
		case DOUBLE: {
			double *ptr = (double *)res.data, *ib = (double *)begin(&it);
			double num2 = (double)num;
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = (*ib) * num2;
			}
			break;
		}
		case FLOAT: {
			float *ptr = (float *)res.data, *ib = (float *)begin(&it);
			float num2 = (float)num;
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = (*ib) * num2;
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			double complex *ptr = (double complex *)res.data, *ib = (double complex *)begin(&it);
			double complex num2 = (double complex)num;
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = (*ib) * num2;
			}
			break;
		}
		case INT: {
			int *ptr = (int *)res.data, *ib = (int *)begin(&it);
			int num2 = (int)num;
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = (*ib) * num2;
			}
			break;
		}
		default: {
			printf("Unsupported data type\n");
			break;
		}
	}
	return res;
}

Tensor EWMulTensorScalar_out(Tensor*output,const Tensor a, Scalar num) {
	assert(a.valid);
	Iterator it = getIterator(a);
	switch (a.dtype) {
		case DOUBLE: {
			double *ptr = (double *)output->data, *ib = (double *)begin(&it);
			double num2 = (double)num;
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = (*ib) * num2;
			}
			break;
		}
		case FLOAT: {
			float *ptr = (float *)output->data, *ib = (float *)begin(&it);
			float num2 = (float)num;
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = (*ib) * num2;
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			double complex *ptr = (double complex *)output->data, *ib = (double complex *)begin(&it);
			double complex num2 = (double complex)num;
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = (*ib) * num2;
			}
			break;
		}
		case INT: {
			int *ptr = (int *)output->data, *ib = (int *)begin(&it);
			int num2 = (int)num;
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = (*ib) * num2;
			}
			break;
		}
		default: {
			printf("Unsupported data type\n");
			break;
		}
	}
	return *output;
}

//mul scalar*tensor
Tensor EWMulScalarTensor(Scalar num, const Tensor a) {
	return EWMulTensorScalar(a, num);
}

Tensor EWMul(const Tensor a, const Tensor b) {
	assert(a.valid && b.valid);
	// compute common type (promote)
	// If common type is different from original type, do cast
	Tensor lhs = a, rhs = b;
	ScalarType type = castToCommonType(&lhs, &rhs);

	// Get shape after broadcasting
	Shape shapeRes;
	bool validShape = computeBroadCastShape(&(lhs.shape), &(rhs.shape), &shapeRes);
	if (!validShape) { // invalid shape
		printf("Error: Shapes of two tensor don't match\n");
		exit(0);
	}

	// do element-wise operation
	Tensor res = zerosFromShape(type, shapeRes);

	Iterator itA = getIterator(lhs), itB = getIterator(rhs);
	switch (type) {
		case DOUBLE: {
			double *ptr = (double *)res.data, *ibA = (double *)begin(&itA), *ibB = (double *)begin(&itB);
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ibA = next(&itA), ibB = next(&itB)) {
				*ptr++ = (*ibA) * (*ibB);
			}
			break;
		}
		case FLOAT: {
			float *ptr = (float *)res.data, *ibA = (float *)begin(&itA), *ibB = (float *)begin(&itB);
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ibA = next(&itA), ibB = next(&itB)) {
				*ptr++ = (*ibA) * (*ibB);
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			double complex *ptr = (double complex *)res.data, *ibA = (double complex *)begin(&itA), *ibB = (double complex *)begin(&itB);
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ibA = next(&itA), ibB = next(&itB)) {
				*ptr++ = (*ibA) * (*ibB);
			}
			break;
		}
		case INT: {
			int *ptr = (int *)res.data, *ibA = (int *)begin(&itA), *ibB = (int *)begin(&itB);
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ibA = next(&itA), ibB = next(&itB)) {
				*ptr++ = (*ibA) * (*ibB);
			}
			break;
		}
		default: {
			printf("Unsupported data type\n");
			break;
		}
	}
	return res;
}

Tensor EWMul_out(Tensor*output,const Tensor a, const Tensor b) {
	assert(a.valid && b.valid);
	// compute common type (promote)
	// If common type is different from original type, do cast
	Tensor lhs = a, rhs = b;
	ScalarType type = castToCommonType(&lhs, &rhs);

	// Get shape after broadcasting
	Shape shapeRes;
	bool validShape = computeBroadCastShape(&(lhs.shape), &(rhs.shape), &shapeRes);
	if (!validShape) { // invalid shape
		printf("Error: Shapes of two tensor don't match\n");
		exit(0);
	}

	// do element-wise operation


	Iterator itA = getIterator(lhs), itB = getIterator(rhs);
	switch (type) {
		case DOUBLE: {
			double *ptr = (double *)output->data, *ibA = (double *)begin(&itA), *ibB = (double *)begin(&itB);
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ibA = next(&itA), ibB = next(&itB)) {
				*ptr++ = (*ibA) * (*ibB);
			}
			break;
		}
		case FLOAT: {
			float *ptr = (float *)output->data, *ibA = (float *)begin(&itA), *ibB = (float *)begin(&itB);
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ibA = next(&itA), ibB = next(&itB)) {
				*ptr++ = (*ibA) * (*ibB);
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			double complex *ptr = (double complex *)output->data, *ibA = (double complex *)begin(&itA), *ibB = (double complex *)begin(&itB);
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ibA = next(&itA), ibB = next(&itB)) {
				*ptr++ = (*ibA) * (*ibB);
			}
			break;
		}
		case INT: {
			int *ptr = (int *)output->data, *ibA = (int *)begin(&itA), *ibB = (int *)begin(&itB);
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ibA = next(&itA), ibB = next(&itB)) {
				*ptr++ = (*ibA) * (*ibB);
			}
			break;
		}
		default: {
			printf("Unsupported data type\n");
			break;
		}
	}
	return *output;
}

Tensor EWDiv(const Tensor a, const Tensor b) {
	assert(a.valid && b.valid);
	// compute common type (promote)
	// If common type is different from original type, do cast
	Tensor lhs = a, rhs = b;
	ScalarType type = castToCommonType(&lhs, &rhs);

	// Get shape after broadcasting
	Shape shapeRes;
	bool validShape = computeBroadCastShape(&(lhs.shape), &(rhs.shape), &shapeRes);
	if (!validShape) { // invalid shape
		printf("Error: Shapes of two tensor don't match\n");
		exit(0);
	}

	// do element-wise operation
	Tensor res = zerosFromShape(type, shapeRes);

	Iterator itA = getIterator(lhs), itB = getIterator(rhs);
	switch (type) {
		case DOUBLE: {
			double *ptr = (double *)res.data, *ibA = (double *)begin(&itA), *ibB = (double *)begin(&itB);
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ibA = next(&itA), ibB = next(&itB)) {
				*ptr++ = (*ibA) / (*ibB);
			}
			break;
		}
		case FLOAT: {
			float *ptr = (float *)res.data, *ibA = (float *)begin(&itA), *ibB = (float *)begin(&itB);
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ibA = next(&itA), ibB = next(&itB)) {
				*ptr++ = (*ibA) / (*ibB);
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			double complex *ptr = (double complex *)res.data, *ibA = (double complex *)begin(&itA), *ibB = (double complex *)begin(&itB);
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ibA = next(&itA), ibB = next(&itB)) {
				*ptr++ = (*ibA) / (*ibB);
			}
			break;
		}
		case INT: {
			int *ptr = (int *)res.data, *ibA = (int *)begin(&itA), *ibB = (int *)begin(&itB);
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ibA = next(&itA), ibB = next(&itB)) {
				*ptr++ = (*ibA) / (*ibB);
			}
			break;
		}
		default: {
			printf("Unsupported data type\n");
			break;
		}
	}
	return res;
}

Tensor EWDiv_out(Tensor* output,const Tensor a, const Tensor b) {
	assert(a.valid && b.valid);
	// compute common type (promote)
	// If common type is different from original type, do cast
	Tensor lhs = a, rhs = b;
	ScalarType type = castToCommonType(&lhs, &rhs);

	// Get shape after broadcasting
	Shape shapeRes;
	bool validShape = computeBroadCastShape(&(lhs.shape), &(rhs.shape), &shapeRes);
	if (!validShape) { // invalid shape
		printf("Error: Shapes of two tensor don't match\n");
		exit(0);
	}

	// do element-wise operation

	Iterator itA = getIterator(lhs), itB = getIterator(rhs);
	switch (type) {
		case DOUBLE: {
			double *ptr = (double *)output->data, *ibA = (double *)begin(&itA), *ibB = (double *)begin(&itB);
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ibA = next(&itA), ibB = next(&itB)) {
				*ptr++ = (*ibA) / (*ibB);
			}
			break;
		}
		case FLOAT: {
			float *ptr = (float *)output->data, *ibA = (float *)begin(&itA), *ibB = (float *)begin(&itB);
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ibA = next(&itA), ibB = next(&itB)) {
				*ptr++ = (*ibA) / (*ibB);
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			double complex *ptr = (double complex *)output->data, *ibA = (double complex *)begin(&itA), *ibB = (double complex *)begin(&itB);
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ibA = next(&itA), ibB = next(&itB)) {
				*ptr++ = (*ibA) / (*ibB);
			}
			break;
		}
		case INT: {
			int *ptr = (int *)output->data, *ibA = (int *)begin(&itA), *ibB = (int *)begin(&itB);
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ibA = next(&itA), ibB = next(&itB)) {
				*ptr++ = (*ibA) / (*ibB);
			}
			break;
		}
		default: {
			printf("Unsupported data type\n");
			break;
		}
	}
	return *output;
}

Tensor EWDivTensorScalar(const Tensor a, Scalar num) {
	assert(a.valid);
	Tensor res = zerosFromShape(a.dtype, a.shape);
	Iterator it = getIterator(a);
	switch (a.dtype) {
		case DOUBLE: {
			double *ptr = (double *)res.data, *ib = (double *)begin(&it);
			double num2 = (double)num;
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = (*ib) / num2;
			}
			break;
		}
		case FLOAT: {
			float *ptr = (float *)res.data, *ib = (float *)begin(&it);
			float num2 = (float)num;
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = (*ib) / num2;
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			double complex *ptr = (double complex *)res.data, *ib = (double complex *)begin(&it);
			double complex num2 = (double complex)num;
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = (*ib) / num2;
			}
			break;
		}
		case INT: {
			int *ptr = (int *)res.data, *ib = (int *)begin(&it);
			int num2 = (int)num;
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = (*ib) / num2;
			}
			break;
		}
		default: {
			printf("Unsupported data type\n");
			break;
		}
	}
	return res;
}

Tensor EWDivTensorScalar_out(Tensor*output,const Tensor a, Scalar num) {
	assert(a.valid);
	Iterator it = getIterator(a);
	switch (a.dtype) {
		case DOUBLE: {
			double *ptr = (double *)output->data, *ib = (double *)begin(&it);
			double num2 = (double)num;
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = (*ib) / num2;
			}
			break;
		}
		case FLOAT: {
			float *ptr = (float *)output->data, *ib = (float *)begin(&it);
			float num2 = (float)num;
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = (*ib) / num2;
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			double complex *ptr = (double complex *)output->data, *ib = (double complex *)begin(&it);
			double complex num2 = (double complex)num;
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = (*ib) / num2;
			}
			break;
		}
		case INT: {
			int *ptr = (int *)output->data, *ib = (int *)begin(&it);
			int num2 = (int)num;
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = (*ib) / num2;
			}
			break;
		}
		default: {
			printf("Unsupported data type\n");
			break;
		}
	}
	return *output;
}

Tensor EWAbs(Tensor *this) {
	assert(this->valid);
	ScalarType type = (this->dtype == DOUBLE_COMPLEX) ? DOUBLE : this->dtype;
	Tensor ret = zerosFromShape(type, this->shape);
	Iterator it = getIterator(*this);
	switch(this->dtype) {
		case INT: {
			int *data = (int *)ret.data, *ib = (int *)begin(&it);
			for (int i = 0, n = ret.shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = abs(*ib);
			}
			break;
		}
		case FLOAT: {
			float *data = (float *)ret.data, *ib = (float *)begin(&it);
			for (int i = 0, n = ret.shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = fabsf(*ib);
			}
			break;
		}
		case DOUBLE: {
			double *data = (double *)ret.data, *ib = (double *)begin(&it);
			for (int i = 0, n = ret.shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = fabs(*ib);
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			double *data = (double *)ret.data;
			for (double complex *ib = begin(&it), *ie = end(&it); ib != ie; ib = next(&it)) {
				double real = creal(*ib), imag = cimag(*ib);
				*data++ = sqrt(real*real+imag*imag);
			}
			break;
		}
		default: {
			printf("Unsupported data type\n");
			exit(0);
		}
	}
	return ret;
}

Tensor EWAbs_(Tensor *this) {
	assert(this->valid);
	ScalarType type = (this->dtype == DOUBLE_COMPLEX) ? DOUBLE : this->dtype;
	Iterator it = getIterator(*this);
	switch(this->dtype) {
		case INT: {
			int *data = (int *)this->data, *ib = (int *)begin(&it);
			for (int i = 0, n = this->shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = abs(*ib);
			}
			break;
		}
		case FLOAT: {
			float *data = (float *)this->data, *ib = (float *)begin(&it);
			for (int i = 0, n = this->shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = fabsf(*ib);
			}
			break;
		}
		case DOUBLE: {
			double *data = (double *)this->data, *ib = (double *)begin(&it);
			for (int i = 0, n = this->shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = fabs(*ib);
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			double *data = (double *)this->data;
			for (double complex *ib = begin(&it), *ie = end(&it); ib != ie; ib = next(&it)) {
				double real = creal(*ib), imag = cimag(*ib);
				*data++ = sqrt(real*real+imag*imag);
			}
			break;
		}
		default: {
			printf("Unsupported data type\n");
			exit(0);
		}
	}
	return *this;
}

Tensor EWAbs_out(Tensor*output,Tensor *this) {
	assert(this->valid);
	ScalarType type = (this->dtype == DOUBLE_COMPLEX) ? DOUBLE : this->dtype;
	Iterator it = getIterator(*this);
	switch(this->dtype) {
		case INT: {
			int *data = (int *)output->data, *ib = (int *)begin(&it);
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = abs(*ib);
			}
			break;
		}
		case FLOAT: {
			float *data = (float *)output->data, *ib = (float *)begin(&it);
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = fabsf(*ib);
			}
			break;
		}
		case DOUBLE: {
			double *data = (double *)output->data, *ib = (double *)begin(&it);
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = fabs(*ib);
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			double *data = (double *)output->data;
			for (double complex *ib = begin(&it), *ie = end(&it); ib != ie; ib = next(&it)) {
				double real = creal(*ib), imag = cimag(*ib);
				*data++ = sqrt(real*real+imag*imag);
			}
			break;
		}
		default: {
			printf("Unsupported data type\n");
			exit(0);
		}
	}
	return *output;
}

Tensor EWNegative(const Tensor tensor) {
	assert(tensor.valid);
	Tensor ret = zerosFromShape(tensor.dtype, tensor.shape);
	Iterator it = getIterator(tensor);
	switch(tensor.dtype) {
		case INT: {
			int *data = (int *)ret.data, *ib = (int *)begin(&it);
			for (int i = 0, n = ret.shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = -(*ib);
			}
			break;
		}
		case FLOAT: {
			float *data = (float *)ret.data, *ib = (float *)begin(&it);
			for (int i = 0, n = ret.shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = -(*ib);
			}
			break;
		}
		case DOUBLE: {
			double *data = (double *)ret.data, *ib = (double *)begin(&it);
			for (int i = 0, n = ret.shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = -(*ib);
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			double complex *data = (double complex *)ret.data;
			for (double complex *ib = begin(&it), *ie = end(&it); ib != ie; ib = next(&it)) {
				*data++ = -(*ib);
			}
			break;
		}
		default: {
			printf("Unsupported data type\n");
			exit(0);
		}
	}
	return ret;
}

Tensor EWNegative_(Tensor*tensor) {
	assert(tensor->valid);
	Iterator it = getIterator(*tensor);
	switch(tensor->dtype) {
		case INT: {
			int *data = (int *)tensor->data, *ib = (int *)begin(&it);
			for (int i = 0, n = tensor->shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = -(*ib);
			}
			break;
		}
		case FLOAT: {
			float *data = (float *)tensor->data, *ib = (float *)begin(&it);
			for (int i = 0, n = tensor->shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = -(*ib);
			}
			break;
		}
		case DOUBLE: {
			double *data = (double *)tensor->data, *ib = (double *)begin(&it);
			for (int i = 0, n = tensor->shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = -(*ib);
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			double complex *data = (double complex *)tensor->data;
			for (double complex *ib = begin(&it), *ie = end(&it); ib != ie; ib = next(&it)) {
				*data++ = -(*ib);
			}
			break;
		}
		default: {
			printf("Unsupported data type\n");
			exit(0);
		}
	}
	return *tensor;
}

Tensor EWNegative_out(Tensor*output,const Tensor tensor) {
	assert(tensor.valid);
	Iterator it = getIterator(tensor);
	switch(tensor.dtype) {
		case INT: {
			int *data = (int *)output->data, *ib = (int *)begin(&it);
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = -(*ib);
			}
			break;
		}
		case FLOAT: {
			float *data = (float *)output->data, *ib = (float *)begin(&it);
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = -(*ib);
			}
			break;
		}
		case DOUBLE: {
			double *data = (double *)output->data, *ib = (double *)begin(&it);
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = -(*ib);
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			double complex *data = (double complex *)output->data;
			for (double complex *ib = begin(&it), *ie = end(&it); ib != ie; ib = next(&it)) {
				*data++ = -(*ib);
			}
			break;
		}
		default: {
			printf("Unsupported data type\n");
			exit(0);
		}
	}
	return *output;
}

//func square()
Tensor EWSquare(Tensor *this) {
	assert(this->valid);
	ScalarType type = (this->dtype == DOUBLE_COMPLEX) ? DOUBLE : this->dtype;
	Tensor ret = zerosFromShape(type, this->shape);
	Iterator it = getIterator(*this);
	switch(this->dtype) {
		case INT: {
			int *data = (int *)ret.data, *ib = (int *)begin(&it);
			for (int i = 0, n = ret.shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = (*ib)*(*ib);
			}
			break;
		}
		case FLOAT: {
			float *data = (float *)ret.data, *ib = (float *)begin(&it);
			for (int i = 0, n = ret.shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = (*ib)*(*ib);
			}
			break;
		}
		case DOUBLE: {
			double *data = (double *)ret.data, *ib = (double *)begin(&it);
			for (int i = 0, n = ret.shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = (*ib)*(*ib);
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			double *data = (double *)ret.data;
			for (double complex *ib = begin(&it), *ie = end(&it); ib != ie; ib = next(&it)) {
				double real = creal(*ib), imag = cimag(*ib); //creal , cimag函数为double complex类型的自带的函数
				*data++ = (*ib)*(*ib);;
			}
			break;
		}
		default: {
			printf("Unsupported data type\n");
			exit(0);
		}
	}
	return ret;
}

Tensor EWSquare_(Tensor *this) {
	assert(this->valid);
	ScalarType type = (this->dtype == DOUBLE_COMPLEX) ? DOUBLE : this->dtype;
	Iterator it = getIterator(*this);
	switch(this->dtype) {
		case INT: {
			int *data = (int *)this->data, *ib = (int *)begin(&it);
			for (int i = 0, n = this->shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = (*ib)*(*ib);
			}
			break;
		}
		case FLOAT: {
			float *data = (float *)this->data, *ib = (float *)begin(&it);
			for (int i = 0, n = this->shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = (*ib)*(*ib);
			}
			break;
		}
		case DOUBLE: {
			double *data = (double *)this->data, *ib = (double *)begin(&it);
			for (int i = 0, n = this->shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = (*ib)*(*ib);
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			double *data = (double *)this->data;
			for (double complex *ib = begin(&it), *ie = end(&it); ib != ie; ib = next(&it)) {
				double real = creal(*ib), imag = cimag(*ib);
				*data++ = (*ib)*(*ib);;
			}
			break;
		}
		default: {
			printf("Unsupported data type\n");
			exit(0);
		}
	}
	return *this;
}

Tensor EWSquare_out(Tensor*output,Tensor *this) {
	assert(this->valid);
	ScalarType type = (this->dtype == DOUBLE_COMPLEX) ? DOUBLE : this->dtype;
	Iterator it = getIterator(*this);
	switch(this->dtype) {
		case INT: {
			int *data = (int *)output->data, *ib = (int *)begin(&it);
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = (*ib)*(*ib);
			}
			break;
		}
		case FLOAT: {
			float *data = (float *)output->data, *ib = (float *)begin(&it);
			for (int i = 0, n = output->shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = (*ib)*(*ib);
			}
			break;
		}
		case DOUBLE: {
			double *data = (double *)output->data, *ib = (double *)begin(&it);
			for (int i = 0, n =output->shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = (*ib)*(*ib);
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			double *data = (double *)output->data;
			for (double complex *ib = begin(&it), *ie = end(&it); ib != ie; ib = next(&it)) {
				double real = creal(*ib), imag = cimag(*ib);
				*data++ = (*ib)*(*ib);;
			}
			break;
		}
		default: {
			printf("Unsupported data type\n");
			exit(0);
		}
	}
	return *output;
}

//EWGEScalar
Tensor EWGEScalar(const Tensor a, Scalar num) {
	assert(a.valid);
	Tensor res = zerosFromShape(INT, a.shape);
	Iterator it = getIterator(a);
	switch (a.dtype) {
		case DOUBLE: {
			int *ptr = (int *)res.data;
			double *ib = (double *)begin(&it);
			double num2 = (double)num;
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ib = next(&it)) {
				if((*ib)>= num2){
					*ptr++ = 1;
				} else {
					*ptr++ = 0;
				}
			}
			break;
		}
		case FLOAT: {
			int *ptr = (int *)res.data;
			float *ib = (float *)begin(&it);
			float num2 = (float)num;
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ib = next(&it)) {
				if((*ib)>= num2){
					*ptr++ = 1;
				} else {
					*ptr++ = 0;
				}
			}
			break;
		}
		/*
		case DOUBLE_COMPLEX: {
			int *ptr = (int *)res.data;
			double complex *ib = (double complex *)begin(&it);
			double complex num2 = (double complex)num;
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ib = next(&it)) {
				if((*ib)>= num2){
					*ptr++ = 1;
				} else {
					*ptr++ = 0;
				}
			}
			break;
		}*/
		case INT: {
			int *ptr = (int *)res.data;
			int *ib = (int *)begin(&it);
			int num2 = (int)num;
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ib = next(&it)) {
				if((*ib)>= num2){
					*ptr++ = 1;
				} else {
					*ptr++ = 0;
				}
			}
			break;
		}
		default: {
			printf("Unsupported data type\n");
			break;
		}
	}
	return res;
}

Tensor EWGTScalar(const Tensor a, Scalar num) {
	assert(a.valid);
	Tensor res = zerosFromShape(INT, a.shape);
	Iterator it = getIterator(a);
	switch (a.dtype) {
		case DOUBLE: {
			double *ptr = (double *)res.data, *ib = (double *)begin(&it);
			double num2 = (double)num;
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = *ib > num2;
			}
			break;
		}
		case FLOAT: {
			float *ptr = (float *)res.data, *ib = (float *)begin(&it);
			float num2 = (float)num;
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = *ib > num2;
			}
			break;
		}
		case INT: {
			int *ptr = (int *)res.data, *ib = (int *)begin(&it);
			int num2 = (int)num;
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ib = next(&it)) {
				*ptr++ = *ib > num2;
			}
			break;
		}
		default: {
			printf("Unsupported data type\n");
			break;
		}
	}
	return res;		
}

Tensor EWExp(Tensor *this) {
	assert(this->valid);
	ScalarType type = (this->dtype == DOUBLE_COMPLEX) ? DOUBLE : this->dtype;
	Tensor ret = zerosFromShape(type, this->shape);
	Iterator it = getIterator(*this);
	switch(this->dtype) {
		case INT: {
			int *data = (int *)ret.data, *ib = (int *)begin(&it);
			for (int i = 0, n = ret.shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = exp(*ib);
			}
			break;
		}
		case FLOAT: {
			float *data = (float *)ret.data, *ib = (float *)begin(&it);
			for (int i = 0, n = ret.shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = exp(*ib);
			}
			break;
		}
		case DOUBLE: {
			double *data = (double *)ret.data, *ib = (double *)begin(&it);
			for (int i = 0, n = ret.shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = exp(*ib);
			}
			break;
		}
		default: {
			printf("Unsupported data type for exp10\n");
			exit(0);
		}
	}
	return ret;
}

Tensor EWExp10(Tensor *this) {
	assert(this->valid);
	ScalarType type = (this->dtype == DOUBLE_COMPLEX) ? DOUBLE : this->dtype;
	Tensor ret = zerosFromShape(type, this->shape);
	Iterator it = getIterator(*this);
	switch(this->dtype) {
		case INT: {
			int *data = (int *)ret.data, *ib = (int *)begin(&it);
			for (int i = 0, n = ret.shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = pow(10, *ib);
			}
			break;
		}
		case FLOAT: {
			float *data = (float *)ret.data, *ib = (float *)begin(&it);
			for (int i = 0, n = ret.shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = pow(10, *ib);
			}
			break;
		}
		case DOUBLE: {
			double *data = (double *)ret.data, *ib = (double *)begin(&it);
			for (int i = 0, n = ret.shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = pow(10, *ib);
			}
			break;
		}
		default: {
			printf("Unsupported data type for exp10\n");
			exit(0);
		}
	}
	return ret;
}
