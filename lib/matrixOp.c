#include "tensor.h"
#include "iterator.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <assert.h>

Tensor matmul(Tensor *a, const Tensor b) {
	assert(a->valid && b.valid);
	if (a->shape.ndim != 2 || b.shape.ndim != 2 || a->shape.dims[1] != b.shape.dims[0]) {
		printf("Shapes of two matrix to multiply don't match\n");
		exit(0);
	}

	// get common type
	Tensor lhs = *a, rhs = b;
	ScalarType type = castToCommonType(&lhs, &rhs);

	int m = a->shape.dims[0], k = a->shape.dims[1], n = b.shape.dims[1];
	Tensor res = zeros(type, 2, m, n);

	int *strideA = lhs.shape.strides, *strideB = rhs.shape.strides;
	switch(type) {
		case DOUBLE: {
			double *dataA = (double *)a->data, *dataB = (double *)b.data, *dataRes = (double *)res.data;
			for (int i = 0; i < m; ++i) {
				for (int j = 0; j < n; ++j) {
					double tmp = 0;
					for (int l = 0; l < k; ++l) {
						tmp += *(dataA + i *strideA[0] + l * strideA[1]) * *(dataB + l * strideB[0] + j * strideB[1]);
					}
					*(dataRes + i * n + j) = tmp;
				}
			}
			break;
		}
		case FLOAT: {
			float *dataA = (float *)a->data, *dataB = (float *)b.data, *dataRes = (float *)res.data;
			for (int i = 0; i < m; ++i) {
				for (int j = 0; j < n; ++j) {
					float tmp = 0;
					for (int l = 0; l < k; ++l) {
						tmp += *(dataA + i *strideA[0] + l * strideA[1]) * *(dataB + l * strideB[0] + j * strideB[1]);
					}
					*(dataRes + i * n + j) = tmp;
				}
			}
			break;
		}
		case INT: {
			int *dataA = (int *)a->data, *dataB = (int *)b.data, *dataRes = (int *)res.data;
			for (int i = 0; i < m; ++i) {
				for (int j = 0; j < n; ++j) {
					int tmp = 0;
					for (int l = 0; l < k; ++l) {
						tmp += *(dataA + i *strideA[0] + l * strideA[1]) * *(dataB + l * strideB[0] + j * strideB[1]);
					}
					*(dataRes + i * n + j) = tmp;
				}
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			double complex *dataA = (double complex *)a->data, *dataB = (double complex *)b.data, *dataRes = (double complex *)res.data;
			for (int i = 0; i < m; ++i) {
				for (int j = 0; j < n; ++j) {
					double complex tmp = 0;
					for (int l = 0; l < k; ++l) {
						tmp += *(dataA + i *strideA[0] + l * strideA[1]) * *(dataB + l * strideB[0] + j * strideB[1]);
					}
					*(dataRes + i * n + j) = tmp;
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

Tensor diag(Tensor *tensor) {
	assert(tensor->valid);
	int typesize = getTypeSize(tensor->dtype);
	if (tensor->shape.ndim == 1) {
		int n = tensor->shape.dims[0];
		Tensor res = zeros(tensor->dtype, 2, n, n);
		int stride = tensor->shape.strides[0];
		for (int i = 0; i < n; ++i) {
			memcpy(res.data + (i * n + i) * typesize, tensor->data + i * stride * typesize, typesize);
		}
		return res;
	} else if (tensor->shape.ndim == 2) {
		int m = tensor->shape.dims[0], n = tensor->shape.dims[1];
		int len = m < n ? m : n;
		Tensor res = zeros(tensor->dtype, 1, len);
		int *stride = tensor->shape.strides;
		for (int i = 0; i < len; ++i) {
			memcpy(res.data + i * typesize, tensor->data + (i * stride[0] + i * stride[1]) * typesize, typesize);
		}
		return res;
	} else {
		printf("The input of diag function should be 1-D or 2-D array\n");
		exit(0);
	}
}


double func(double num, void *input) {
	double tol = *(double *)input;
	return num < tol ? 0 : num;
}
Tensor pinv(Tensor *matrix) {
	assert(matrix->valid);
	if (matrix->shape.ndim != 2) {
		printf("The input of pinv should be 2-D matrix\n");
		exit(0);
	}
	if (matrix->dtype != DOUBLE) {
		printf("Unsupported data type for svd\n");
		exit(0);
	}

	int m = matrix->shape.dims[0], n = matrix->shape.dims[1];
	int minmn = m < n ? m : n;
	Tensor *res = dsvd(matrix); // return u, s, v
	Tensor u = res[0], s = res[1], v = res[2];

	// [U,S,V] = svd(A);   %A = U*S*V'
	// pinv(A) = V * S-1 * U';
	//double tol = 1e-6;
	//Tensor s1 = map(&s, func, &tol); // s[i][i] < tol -> s[i][i] = 0
	Tensor tmps = zeros(matrix->dtype, 2, n, m);
	double *dataS = (double *)s.data, *dataSInv = (double *)tmps.data;
	for (int i = 0; i < minmn; ++i) {
		double num = *(dataS + i * s.shape.strides[0] + i * s.shape.strides[1]);
		if (num > 1e-6)
			*(dataSInv + i * tmps.shape.strides[0] + i * tmps.shape.strides[1]) = 1.0 / num;
	}
	Tensor tmp = matmul(&v, tmps);
	return matmul(&tmp, transpose(&u));
}
