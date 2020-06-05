#include "tensor.h"
#include "iterator.h"
#include <complex.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

int dim(Tensor *this) {
	return this->shape.ndim;
}
int size(Tensor *this, int d) {
	if (d < 0 || d >= this->shape.ndim) {
		printf("The dim is out of range\n");
		return -1;
	}
	return this->shape.dims[d];
}
int nelem(Tensor *this) {
	return this->shape.nelem;
}
ScalarType getDataType(Tensor *this) {
	return this->dtype;
}
void *data_ptr(Tensor *this) {
	return this->data;
}

Tensor fromdata(void*data, ScalarType type, int n,...) {
                Tensor res;
                va_list vl;
	va_start(vl, n);
	res.shape = allocShapeFromValist(n, vl);
	va_end(vl);
	res.valid = true;
	switch(type) {
		case FLOAT: {
			res.dtype = FLOAT;
			res.data = data; 
			break;
		}
		case DOUBLE: {
			res.dtype = DOUBLE;
			res.data = data;
			break;
		}
		case DOUBLE_COMPLEX: {
			res.dtype = DOUBLE_COMPLEX;
			res.data = data;
			break;
		}
		case INT: {
			res.dtype = INT;
			res.data = data;
			break;
		}
		default: {
			printf("Unsupported data type\n");
			exit(0);
		}
	}
	return res;
}

Tensor fromfile(FILE *fd, ScalarType type, int n) {
	Tensor res;
	res.valid = true;
	res.shape.ndim = 1;
	res.shape.dims[0] = n;
	res.shape.strides[0] = 1;
	res.shape.nelem = n;
	switch(type) {
		case FLOAT: {
			res.dtype = FLOAT;
			res.data = malloc(n * sizeof(float));
			fread(res.data, sizeof(float), n, fd);
			break;
		}
		case DOUBLE: {
			res.dtype = DOUBLE;
			res.data = malloc(n * sizeof(double));
			fread(res.data, sizeof(double), n, fd);
			break;
		}
		case DOUBLE_COMPLEX: {
			res.dtype = DOUBLE_COMPLEX;
			res.data = malloc(n * sizeof(double complex));
			fread(res.data, sizeof(double complex), n, fd);
			break;
		}
		case INT: {
			res.dtype = INT;
			res.data = malloc(n * sizeof(int));
			fread(res.data, sizeof(int), n, fd);
			break;
		}
		default: {
			printf("Unsupported data type\n");
			exit(0);
		}
	}
	return res;
}

void fillData(void *data, ScalarType type, Scalar num, int count) {
	switch(type) {
		case FLOAT: {
			float *ptr = (float *)data;
			float number = (float)num;
			for (int i = 0; i < count; ++i) {
				*ptr++ = number;
			}
			break;
		}
		case DOUBLE: {
			double *ptr = (double *)data;
			double number = (double)num;
			for (int i = 0; i < count; ++i) {
				*ptr++ = number;
			}
			break;
		}
		case INT: {
			int *ptr = (int *)data;
			int number = (int)num;
			for (int i = 0; i < count; ++i) {
				*ptr++ = number;
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			double complex *ptr = (double complex *)data;
			double complex number = (double complex)num;
			for (int i = 0; i < count; ++i) {
				*ptr++ = number;
			}
			break;
		}
		default: {
			printf("Unsupported data type\n");
			exit(0);
		}
	}
}
Tensor full(ScalarType type, Scalar num, int n, ...) {
	Tensor res;
	va_list vl;
	va_start(vl, n);
	Shape shape = allocShapeFromValist(n, vl);
	res.valid = true;
	res.shape = shape;
	res.dtype = type;
	res.data = malloc(getTypeSize(type) * shape.nelem);
	fillData(res.data, type, num, shape.nelem);
	return res;
}
Tensor ones(ScalarType type, int n, ...) {
	Tensor res;
	va_list vl;
	va_start(vl, n);
	Shape shape = allocShapeFromValist(n, vl);
	res.valid = true;
	res.shape = shape;
	res.dtype = type;
	res.data = malloc(getTypeSize(type) * shape.nelem);
	fillData(res.data, type, 1, shape.nelem);
	return res;
}

Tensor zeros(ScalarType type, int n, ...) {
	Tensor res;
	va_list vl;
	va_start(vl, n);
	Shape shape = allocShapeFromValist(n, vl);
	res.valid = true;
	res.shape = shape;
	res.dtype = type;
	res.data = malloc(getTypeSize(type) * shape.nelem);
	fillData(res.data, type, 0, shape.nelem);
	return res;
}

Tensor arange(ScalarType type, Scalar start, Scalar end, Scalar step) {
	Tensor res;
	res.valid = true;
	res.dtype = type;
	res.shape.ndim = 1;
	int count = (end - start + step - 1) / step;
	res.shape.dims[0] = count;
	res.shape.strides[0] = 1;
	res.shape.nelem = count;
	switch(type) {
		case INT: {
			int startnum = (int)start, endnum = (int)end, stepnum = (int)step;
			int *data = (int *)malloc(count * sizeof(int));
			res.data = data;
			for (int i = 0; i < count; ++i) {
				data[i] = startnum + i * stepnum;
			}
			break;
		}
		case FLOAT: {
			float startnum = (float)start, endnum = (float)end, stepnum = (float)step;
			float *data = (float *)malloc(count * sizeof(float));
			res.data = data;
			for (int i = 0; i < count; ++i) {
				data[i] = startnum + i * stepnum;
			}
			break;
		}
		case DOUBLE: {
			double startnum = (double)start, endnum = (double)end, stepnum = (double)step;
			double *data = (double *)malloc(count * sizeof(double));
			res.data = data;
			for (int i = 0; i < count; ++i) {
				data[i] = startnum + i * stepnum;
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			double complex startnum = (double complex)start, endnum = (double complex)end, stepnum = (double complex)step;
			double complex *data = (double complex *)malloc(count * sizeof(double complex));
			res.data = data;
			for (int i = 0; i < count; ++i) {
				data[i] = startnum + i * stepnum;
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

Tensor onesFromShape(ScalarType type, Shape shape) {
	Tensor res;
	res.valid = true;
	res.dtype = type;
	res.shape = cloneShapeExceptStride(shape);
	res.data = malloc(getTypeSize(type) * shape.nelem);
	fillData(res.data, type, 1, shape.nelem);
	return res;
}

Tensor zerosFromShape(ScalarType type, Shape shape) {
	Tensor res;
	res.valid = true;
	res.dtype = type;
	res.shape = cloneShapeExceptStride(shape);
	res.data = malloc(getTypeSize(type) * shape.nelem);
	fillData(res.data, type, 0, shape.nelem);
	return res;
}

Tensor castTo(Tensor tensor, ScalarType type) {
	if (tensor.dtype == type) return tensor;
	Tensor res = zerosFromShape(type, tensor.shape);
	Iterator it = getIterator(tensor);
	switch(tensor.dtype) {
		case INT: {
			switch(type) {
				case FLOAT: {
					float *data = (float *)res.data;
					int *ib = begin(&it);
					for (int i = 0, n = tensor.shape.nelem; i < n; ++i, ib = next(&it)) {
						*data++ = *ib;
					}
					break;
				}
				case DOUBLE: {
					double *data = (double *)res.data;
					int *ib = begin(&it);
					for (int i = 0, n = tensor.shape.nelem; i < n; ++i, ib = next(&it)) {
						*data++ = *ib;
					}
					break;
				}
				case DOUBLE_COMPLEX: {
					double complex *data = (double complex *)res.data;
					int *ib = begin(&it);
					for (int i = 0, n = tensor.shape.nelem; i < n; ++i, ib = next(&it)) {
						*data++ = *ib;
					}
					break;
				}
				default: {
					printf("Unsupported data type, or cast cannot be done\n");
					return tensor;
				}
			}
			break;
		}
		case FLOAT: {
			switch(type) {
				case DOUBLE: {
					double *data = (double *)res.data;
					float *ib = begin(&it);
					for (int i = 0, n = tensor.shape.nelem; i < n; ++i, ib = next(&it)) {
						*data++ = *ib;
					}
					break;
				}
				case DOUBLE_COMPLEX: {
					double complex *data = (double complex *)res.data;
					float *ib = begin(&it);
					for (int i = 0, n = tensor.shape.nelem; i < n; ++i, ib = next(&it)) {
						*data++ = *ib;
					}
					break;
				}
				default: {
					printf("Unsupported data type, or cast cannot be done\n");
					return tensor;
				}
			}
			break;
		}
		case DOUBLE: {
			switch(type) {
				case DOUBLE_COMPLEX: {
					double complex *data = (double complex *)res.data;
					double *ib = begin(&it);
					for (int i = 0, n = tensor.shape.nelem; i < n; ++i, ib = next(&it)) {
						*data++ = *ib;
					}
					break;
				}
				default: {
					printf("Unsupported data type, or cast cannot be done\n");
					return tensor;
				}
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			printf("Unsupported data type, or cast cannot be done\n");
			return tensor;
		}
	}
	return res;
}

static Tensor forceCastTo(Tensor tensor, ScalarType type) {
	if (tensor.dtype == type) return tensor;
	Tensor res = zerosFromShape(type, tensor.shape);
	Iterator it = getIterator(tensor);
	switch(tensor.dtype) {
		case INT: {
			switch(type) {
				case FLOAT: {
					float *data = (float *)res.data;
					int *ib = begin(&it);
					for (int i = 0, n = tensor.shape.nelem; i < n; ++i, ib = next(&it)) {
						*data++ = *ib;
					}
					break;
				}
				case DOUBLE: {
					double *data = (double *)res.data;
					int *ib = begin(&it);
					for (int i = 0, n = tensor.shape.nelem; i < n; ++i, ib = next(&it)) {
						*data++ = *ib;
					}
					break;
				}
				case DOUBLE_COMPLEX: {
					double complex *data = (double complex *)res.data;
					int *ib = begin(&it);
					for (int i = 0, n = tensor.shape.nelem; i < n; ++i, ib = next(&it)) {
						*data++ = *ib;
					}
					break;
				}
				default: {
					printf("Unsupported data type\n");
					exit(0);
				}
			}
			break;
		}
		case FLOAT: {
			switch(type) {
				case INT: {
					int *data = (int *)res.data;
					float *ib = begin(&it);
					for (int i = 0, n = tensor.shape.nelem; i < n; ++i, ib = next(&it)) {
						*data++ = (int)*ib;
					}
					break;
				}
				case DOUBLE: {
					double *data = (double *)res.data;
					float *ib = begin(&it);
					for (int i = 0, n = tensor.shape.nelem; i < n; ++i, ib = next(&it)) {
						*data++ = *ib;
					}
					break;
				}
				case DOUBLE_COMPLEX: {
					double complex *data = (double complex *)res.data;
					float *ib = begin(&it);
					for (int i = 0, n = tensor.shape.nelem; i < n; ++i, ib = next(&it)) {
						*data++ = *ib;
					}
					break;
				}
				default: {
					printf("Unsupported data type\n");
					exit(0);
				}
			}
			break;
		}
		case DOUBLE: {
			switch(type) {
				case INT: {
					int *data = (int *)res.data;
					double *ib = begin(&it);
					for (int i = 0, n = tensor.shape.nelem; i < n; ++i, ib = next(&it)) {
						*data++ = (int)*ib;
					}
					break;
				}
				case FLOAT: {
					float *data = (float *)res.data;
					double *ib = begin(&it);
					for (int i = 0, n = tensor.shape.nelem; i < n; ++i, ib = next(&it)) {
						*data++ = (float)*ib;
					}
					break;
				}
				case DOUBLE_COMPLEX: {
					double complex *data = (double complex *)res.data;
					double *ib = begin(&it);
					for (int i = 0, n = tensor.shape.nelem; i < n; ++i, ib = next(&it)) {
						*data++ = *ib;
					}
					break;
				}
				default: {
					printf("Unsupported data type\n");
					exit(0);
				}
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			switch(type) {
				case INT: {
					int *data = (int *)res.data;
					double complex *ib = begin(&it);
					for (int i = 0, n = tensor.shape.nelem; i < n; ++i, ib = next(&it)) {
						*data++ = (int)*ib;
					}
					break;
				}
				case FLOAT: {
					float *data = (float *)res.data;
					double complex *ib = begin(&it);
					for (int i = 0, n = tensor.shape.nelem; i < n; ++i, ib = next(&it)) {
						*data++ = (float)*ib;
					}
					break;
				}
				case DOUBLE: {
					double *data = (double *)res.data;
					double complex *ib = begin(&it);
					for (int i = 0, n = tensor.shape.nelem; i < n; ++i, ib = next(&it)) {
						*data++ = (double)*ib;
					}
					break;
				}
				default: {
					printf("Unsupported data type\n");
					exit(0);
				}
			}
			break;
		}
	}
	return res;
}

ScalarType castToCommonType(Tensor *lhs, Tensor *rhs) {
	ScalarType type = getCommonType(lhs->dtype, rhs->dtype);
	*lhs = castTo(*lhs, type);
	*rhs = castTo(*rhs, type);
	return type;
}

Tensor copy(Tensor *dst, const Tensor src) {
	if (!dst->valid) *dst = zerosFromShape(src.dtype, src.shape);
	if (!isSameShape(dst->shape, src.shape)) {
		printf("Shapes of two tensors to copy should match\n");
		exit(0);
	}
	Tensor src_copy = forceCastTo(src, dst->dtype);
	Iterator it1 = getIterator(*dst), it2 = getIterator(src_copy);
	int typesize = getTypeSize(dst->dtype);
	void *ib1 = begin(&it1), *ib2 = begin(&it2);
	for (int i = 0, n = dst->shape.nelem; i < n; ++i, ib1 = next(&it1), ib2 = next(&it2)) {
		memcpy(ib1, ib2, typesize);
	}
	return *dst;
}
Tensor copyScalar(Tensor *dst, Scalar src) {
	assert(dst->valid);
	Iterator it = getIterator(*dst);
	switch(dst->dtype) {
		case INT: {
			int num = (int)src;
			for (int *ib = (int *)begin(&it), *ie = (int *)end(&it); ib != ie; ib = (int *)next(&it)) {
				*ib = num;
			}
			break;
		}
		case FLOAT: {
			float num = (float)src;
			for (float *ib = (float *)begin(&it), *ie = (float *)end(&it); ib != ie; ib = (float *)next(&it)) {
				*ib = num;
			}
			break;
		}
		case DOUBLE: {
			double num = (double)src;
			for (double *ib = (double *)begin(&it), *ie = (double *)end(&it); ib != ie; ib = (double *)next(&it)) {
				*ib = num;
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			double complex num = (double complex)src;
			for (double complex *ib = (double complex *)begin(&it), *ie = (double complex *)end(&it); ib != ie; ib = (double complex *)next(&it)) {
				*ib = num;
			}
			break;
		}
		default: {
			printf("Unsupported data type\n");
			exit(0);
		}
	}
	return *dst;
}

Tensor allocComplexTensor(Tensor real, Tensor imag) {
	Tensor res;
	if (!isSameShape(real.shape, imag.shape)) {
		printf("Shapes of real and imag should match\n");
		return res;
	}
	res.dtype = DOUBLE_COMPLEX;
	res.valid = true;
	ScalarType type = getCommonType(real.dtype, imag.dtype);
	Tensor real_copy = castTo(real, type), imag_copy = castTo(imag, type);
	res.shape = cloneShapeExceptStride(real.shape);
	res.data = malloc(res.shape.nelem * sizeof(double complex));
	double complex *data = (double complex *)res.data;
	Iterator it1 = getIterator(real_copy), it2 = getIterator(imag_copy);
	switch(type) {
		case INT: {
			int *ib1 = begin(&it1), *ib2 = begin(&it2);
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ib1 = next(&it1), ib2 = next(&it2)) {
				*data++ = *ib1 + 1j * (*ib2);
			}
			break;
		}
		case FLOAT: {
			float *ib1 = begin(&it1), *ib2 = begin(&it2);
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ib1 = next(&it1), ib2 = next(&it2)) {
				*data++ = *ib1 + 1j * (*ib2);
			}
			break;
		}
		case DOUBLE: {
			double *ib1 = begin(&it1), *ib2 = begin(&it2);
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ib1 = next(&it1), ib2 = next(&it2)) {
				*data++ = *ib1 + 1j * (*ib2);
			}
			break;
		}
		default: {
			printf("Unsupported data type\n");
		}
	}
	return res;
}

void dump(const Tensor *tensor) {
	if (!tensor->valid) {
		printf("Tensor is invalid\n");
		return;
	}
	Shape s = tensor->shape;
	dumpShape(s);
	Iterator it = getIterator(*tensor);
	switch(tensor->dtype) {
	case INT: {
		printf("dtype: int\n");
		int *data = tensor->data;
		switch(s.ndim) {
		case 0:
			printf("%d\n", *data);
			break;
		case 1:
			for (int i = 0; i < s.dims[0]; ++i)
				printf("%d,  ", *(data + i * s.strides[0]));
			printf("\n");
			break;
		case 2:
			printf("[");
			for (int i = 0; i < s.dims[0]; ++i) {
				printf("[");
				for (int j = 0; j < s.dims[1]; ++j) {
					printf("%d,  ", *(data + i * s.strides[0] + j * s.strides[1]));
				}
				printf("]\n");
			}
			printf("]\n");
			break;
		case 3:
			printf("[");
			for (int i = 0; i < s.dims[0]; ++i) {
				printf("[");
				for (int j = 0; j < s.dims[1]; ++j) {
					printf("[");
					for (int k = 0; k < s.dims[2]; ++k) {
						printf("%d,  ", *(data + i * s.strides[0] + j * s.strides[1] + k * s.strides[2]));
					}
					printf("]\n");
				}
				printf("]\n");
			}
			printf("]\n");
			break;
		case 4:
			printf("[");
			for (int i = 0; i < s.dims[0]; ++i) {
				printf("[");
				for (int j = 0; j < s.dims[1]; ++j) {
					printf("[");
					for (int k = 0; k < s.dims[2]; ++k) {
						printf("[");
						for (int l = 0; l < s.dims[3]; ++l) {
							printf("%d,  ", *(data + i * s.strides[0] + j * s.strides[1] +
									          k * s.strides[2] + l * s.strides[3]));
						}
						printf("]\n");
					}
					printf("]\n");
				}
				printf("]\n");
			}
			printf("]\n");
			break;
		}
		break;
	}
	case FLOAT: {
		printf("dtype: float\n");
		float *data = tensor->data;
		switch(s.ndim) {
		case 0:
			printf("%g\n", *data);
			break;
		case 1:
			for (int i = 0; i < s.dims[0]; ++i)
				printf("%g,  ", *(data + i * s.strides[0]));
			printf("\n");
			break;
		case 2:
			printf("[");
			for (int i = 0; i < s.dims[0]; ++i) {
				printf("[");
				for (int j = 0; j < s.dims[1]; ++j) {
					printf("%g,  ", *(data + i * s.strides[0] + j * s.strides[1]));
				}
				printf("]\n");
			}
			printf("]\n");
			break;
		case 3:
			printf("[");
			for (int i = 0; i < s.dims[0]; ++i) {
				printf("[");
				for (int j = 0; j < s.dims[1]; ++j) {
					printf("[");
					for (int k = 0; k < s.dims[2]; ++k) {
						printf("%g,  ", *(data + i * s.strides[0] + j * s.strides[1] + k * s.strides[2]));
					}
					printf("]\n");
				}
				printf("]\n");
			}
			printf("]\n");
			break;
		case 4:
			printf("[");
			for (int i = 0; i < s.dims[0]; ++i) {
				printf("[");
				for (int j = 0; j < s.dims[1]; ++j) {
					printf("[");
					for (int k = 0; k < s.dims[2]; ++k) {
						printf("[");
						for (int l = 0; l < s.dims[3]; ++l) {
							printf("%g,  ", *(data + i * s.strides[0] + j * s.strides[1] +
									          k * s.strides[2] + l * s.strides[3]));
						}
						printf("]\n");
					}
					printf("]\n");
				}
				printf("]\n");
			}
			printf("]\n");
			break;
		}
		break;
	}
	case DOUBLE: {
		printf("dtype: double\n");
		double *data = tensor->data;
		switch(s.ndim) {
		case 0:
			printf("%lg\n", *data);
			break;
		case 1:
			for (int i = 0; i < s.dims[0]; ++i)
				printf("%lg,  ", *(data + i * s.strides[0]));
			printf("\n");
			break;
		case 2:
			printf("[");
			for (int i = 0; i < s.dims[0]; ++i) {
				printf("[");
				for (int j = 0; j < s.dims[1]; ++j) {
					printf("%lg,  ", *(data + i * s.strides[0] + j * s.strides[1]));
				}
				printf("]\n");
			}
			printf("]");
			break;
		case 3:
			printf("[");
			for (int i = 0; i < s.dims[0]; ++i) {
				printf("[");
				for (int j = 0; j < s.dims[1]; ++j) {
					printf("[");
					for (int k = 0; k < s.dims[2]; ++k) {
						printf("%lg,  ", *(data + i * s.strides[0] + j * s.strides[1] + k * s.strides[2]));
					}
					printf("]\n");
				}
				printf("]\n");
			}
			printf("]\n");
			break;
		case 4:
			printf("[");
			for (int i = 0; i < s.dims[0]; ++i) {
				printf("[");
				for (int j = 0; j < s.dims[1]; ++j) {
					printf("[");
					for (int k = 0; k < s.dims[2]; ++k) {
						printf("[");
						for (int l = 0; l < s.dims[3]; ++l) {
							printf("%lg,  ", *(data + i * s.strides[0] + j * s.strides[1] +
									          k * s.strides[2] + l * s.strides[3]));
						}
						printf("]\n");
					}
					printf("]\n");
				}
				printf("]\n");
			}
			printf("]\n");
			break;
		}
		break;
	}
	case DOUBLE_COMPLEX: {
		printf("dtype: double complex\n");
		double complex *data = tensor->data;
		switch(s.ndim) {
		case 0:
			printf("%lg + %lgi\n", creal(*data), cimag(*data));
			break;
		case 1:
			for (int i = 0; i < s.dims[0]; ++i) {
				double complex tmp = *(data + i * s.strides[0]);
				printf("%lg + %lgi,  ", creal(tmp), cimag(tmp));
			}
			printf("\n");
			break;
		case 2:
			printf("[");
			for (int i = 0; i < s.dims[0]; ++i) {
				printf("[");
				for (int j = 0; j < s.dims[1]; ++j) {
					double complex tmp = *(data + i * s.strides[0] + j * s.strides[1]);
					printf("%lg + %lgi,  ", creal(tmp), cimag(tmp));
				}
				printf("]\n");
			}
			printf("]\n");
			break;
		case 3:
			printf("[");
			for (int i = 0; i < s.dims[0]; ++i) {
				printf("[");
				for (int j = 0; j < s.dims[1]; ++j) {
					printf("[");
					for (int k = 0; k < s.dims[2]; ++k) {
						double complex tmp = *(data + i * s.strides[0] + j * s.strides[1] + k * s.strides[2]);
						printf("%lg + %lgi,  ", creal(tmp), cimag(tmp));
					}
					printf("]\n");
				}
				printf("]\n");
			}
			printf("]\n");
			break;
		case 4:
			printf("[");
			for (int i = 0; i < s.dims[0]; ++i) {
				printf("[");
				for (int j = 0; j < s.dims[1]; ++j) {
					printf("[");
					for (int k = 0; k < s.dims[2]; ++k) {
						printf("[");
						for (int l = 0; l < s.dims[3]; ++l) {
							double complex tmp = *(data + i * s.strides[0] + j * s.strides[1] +
							          	  	  	   k * s.strides[2] + l * s.strides[3]);
							printf("%lg + %lgi,  ", creal(tmp), cimag(tmp));
						}
						printf("]\n");
					}
					printf("]\n");
				}
				printf("]\n");
			}
			printf("]\n");
			break;
		}
		break;
	}
	}
}
