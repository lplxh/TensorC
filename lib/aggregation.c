#include "tensor.h"
#include "iterator.h"
#include <stdlib.h>
#include <assert.h>

Scalar sumOfTensor(Tensor *tensor) {
	assert(tensor->valid);
	Iterator it = getIterator(*tensor);
	Scalar res = 0;
	switch(tensor->dtype) {
		case INT: {
			int *ib = begin(&it);
			for (int i = 0, n = tensor->shape.nelem; i < n; ++i, ib = next(&it)) {
				res += *ib;
			}
			break;
		}
		case FLOAT: {
			float *ib = begin(&it);
			for (int i = 0, n = tensor->shape.nelem; i < n; ++i, ib = next(&it)) {
				res += *ib;
			}
			break;
		}
		case DOUBLE: {
			double *ib = begin(&it);
			for (int i = 0, n = tensor->shape.nelem; i < n; ++i, ib = next(&it)) {
				res += *ib;
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			double complex *ib = begin(&it);
			for (int i = 0, n = tensor->shape.nelem; i < n; ++i, ib = next(&it)) {
				res += *ib;
			}
			break;
		}
		default: {
			printf("Unsupported data type\n");
			exit(0);
		}
	}
	return res;
}

Scalar meanOfTensor(Tensor *tensor) {
	Scalar res=sumOfTensor(tensor)/tensor->shape.nelem;
                return res;
}

Scalar maxOfTensor(Tensor *tensor) {
	assert(tensor->valid);
	Iterator it = getIterator(*tensor);
	double res;
	switch(tensor->dtype) {
		case INT: {
			int *ib = begin(&it);
			res = *ib;
			for (int i = 0, n = tensor->shape.nelem; i < n; ++i, ib = next(&it)) {
				res = (*ib > res) ? *ib : res;
			}
			break;
		}
		case FLOAT: {
			float *ib = begin(&it);
			res = *ib;
			for (int i = 0, n = tensor->shape.nelem; i < n; ++i, ib = next(&it)) {
				res = (*ib > res) ? *ib : res;
			}
			break;
		}
		case DOUBLE: {
			double *ib = begin(&it);
			res = *ib;
			for (int i = 0, n = tensor->shape.nelem; i < n; ++i, ib = next(&it)) {
				res = (*ib > res) ? *ib : res;
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			printf("Complex cannot be compared with each other\n");
			exit(0);
		}
		default: {
			printf("Unsupported data type\n");
			exit(0);
		}
	}
	return res;
}

Scalar minOfTensor(Tensor *tensor) {
	assert(tensor->valid);
	Iterator it = getIterator(*tensor);
	double res;
	switch(tensor->dtype) {
		case INT: {
			int *ib = begin(&it);
			res = *ib;
			for (int i = 0, n = tensor->shape.nelem; i < n; ++i, ib = next(&it)) {
				res = (*ib < res) ? *ib : res;
			}
			break;
		}
		case FLOAT: {
			float *ib = begin(&it);
			res = *ib;
			for (int i = 0, n = tensor->shape.nelem; i < n; ++i, ib = next(&it)) {
				res = (*ib < res) ? *ib : res;
			}
			break;
		}
		case DOUBLE: {
			double *ib = begin(&it);
			res = *ib;
			for (int i = 0, n = tensor->shape.nelem; i < n; ++i, ib = next(&it)) {
				res = (*ib < res) ? *ib : res;
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			printf("Complex cannot be compared with each other\n");
			exit(0);
		}
		default: {
			printf("Unsupported data type\n");
			exit(0);
		}
	}
	return res;
}

Scalar stdOfTensor(Tensor *tensor) {
	assert(tensor->valid);
	Iterator it = getIterator(*tensor);
	Scalar res = 0;
                Scalar mean = meanOfTensor(tensor);
	switch(tensor->dtype) {
		case INT: {
			int *ib = begin(&it);
			for (int i = 0, n = tensor->shape.nelem; i < n; ++i, ib = next(&it)) {
				res += (*ib-mean)*(*ib-mean);
			}
			break;
		}
		case FLOAT: {
			float *ib = begin(&it);
			for (int i = 0, n = tensor->shape.nelem; i < n; ++i, ib = next(&it)) {
				res +=  (*ib-mean)*(*ib-mean);
			}
			break;
		}
		case DOUBLE: {
			double *ib = begin(&it);
			for (int i = 0, n = tensor->shape.nelem; i < n; ++i, ib = next(&it)) {
				res +=  (*ib-mean)*(*ib-mean);
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			double complex *ib = begin(&it);
			for (int i = 0, n = tensor->shape.nelem; i < n; ++i, ib = next(&it)) {
				res +=  (*ib-mean)*(*ib-mean);
			}
			break;
		}
		default: {
			printf("Unsupported data type\n");
			exit(0);
		}
	}
	return res;
}