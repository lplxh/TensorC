#include "tensor.h"
#include "iterator.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>

Scalar innerproduct(Tensor *this, const Tensor other) {
	if (!isSameShape(this->shape, other.shape)) {
		printf("Shapes of two tensor don't match\n");
		exit(0);
	}
	Tensor res = EWMul(*this, other);
	return sumOfTensor(&res);
}

Tensor outerproduct(Tensor*this,const Tensor other){
               assert(this->valid&&other.valid);
               Tensor lhs = *this, rhs = other;
               ScalarType type = castToCommonType(&lhs,&rhs);
               int ndimA = lhs.shape.ndim;
               int ndimB = rhs.shape.ndim;
               int*dimsA = lhs.shape.dims, *dimsB = rhs.shape.dims;
               if(ndimA+ndimB>4 || ndimA<0 || ndimB<0){
                        printf("The dim is out of range\n");
                        exit(0);
               }
               Shape*shape=(Shape*)malloc(sizeof(Shape));
               shape->ndim = ndimA+ndimB;
               shape->nelem = lhs.shape.nelem*rhs.shape.nelem;
               for(int i =0;i<ndimA;++i){
                    shape->dims[i]= dimsA[i];
               }
               for(int i=0;i<ndimB;++i){
                     shape->dims[i+ndimA]=dimsB[i];
               }
               computeStride(shape);
               Tensor res = onesFromShape(type,*shape);
               Iterator itA = getIterator(lhs),itB = getIterator(rhs);
               switch(type){
                                 case DOUBLE:{
                                                  double *ibA=(double*)begin(&itA),*ibB=(double*)begin(&itB),*dataRes=(double*)res.data;
                                                  for(int i = 0; i< lhs.shape.nelem ;++i,ibA = next(&itA),ibB = (double*)begin(&itB)){
                                                             for(int j=0;j<rhs.shape.nelem;++j,ibB = next(&itB)){
                                                                         *dataRes++ = (*ibA)*(*ibB);
                                                            }
                                                  }
                                                  break;
                                }
                                case FLOAT:{
                                                  float *ibA=(float*)begin(&itA),*ibB=(float*)begin(&itB),*dataRes=(float*)res.data;
                                                  for(int i = 0; i< lhs.shape.nelem ;++i,ibA = next(&itA),ibB = (float*)begin(&itB)){
                                                             for(int j=0;j<rhs.shape.nelem;++j,ibB = next(&itB)){
                                                                         *dataRes++ = (*ibA)*(*ibB);
                                                            }
                                                  }
                                                  break;
                                }
                                case INT:{
                                                  int *ibA=(int*)begin(&itA),*ibB=(int*)begin(&itB),*dataRes=(int*)res.data;
                                                  for(int i = 0; i< lhs.shape.nelem ;++i,ibA = next(&itA),ibB = (int*)begin(&itB)){
                                                             for(int j=0;j<rhs.shape.nelem;++j,ibB = next(&itB)){
                                                                         *dataRes++ = (*ibA)*(*ibB);
                                                            }
                                                  }
                                                  break;
                                }
                            case DOUBLE_COMPLEX:{
                                                  double complex*ibA=(double complex*)begin(&itA),*ibB=(double complex*)begin(&itB),*dataRes=(double complex*)res.data;
                                                  for(int i = 0; i< lhs.shape.nelem ;++i,ibA = next(&itA),ibB = (double complex*)begin(&itB)){
                                                             for(int j=0;j<rhs.shape.nelem;++j,ibB = next(&itB)){
                                                                         *dataRes++ = (*ibA)*(*ibB);
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
double norm(Tensor *this) {
	Scalar s = innerproduct(this, *this);
	if (this->dtype != DOUBLE_COMPLEX) {
		return sqrt((double)s);
	} else {
		printf("Norm of double complex cannot be computed\n");
		exit(0);
	}
}

Tensor unfold(Tensor*this,int mode){
               int dim = this->shape.ndim;
	int *order = (int *)malloc(dim);
	order[0] = mode;
	for (int i = 1; i <= mode; ++i) order[i] = i-1;
	for (int i = mode+1; i < dim; ++i) order[i] = i;
	Tensor afterPermute = permute_internel(this, order);
	Tensor afterReshape = reshape(&afterPermute, 2, this->shape.dims[mode], this->shape.nelem / this->shape.dims[mode]);
               return afterReshape;
}

Tensor ttm(Tensor *this, int mode, const Tensor matrix) {
	int dim = this->shape.ndim;
	int *order = (int *)malloc(dim);
	order[0] = mode;
	for (int i = 1; i <= mode; ++i) order[i] = i-1;
	for (int i = mode+1; i < dim; ++i) order[i] = i;
	Tensor afterPermute = permute_internel(this, order);
	Tensor afterReshape = reshape(&afterPermute, 2, this->shape.dims[mode], this->shape.nelem / this->shape.dims[mode]);
	Tensor mulRes = matmul(&matrix, afterReshape);
	int *newdims = (int *)malloc(dim * sizeof(int));
	newdims[0] = matrix.shape.dims[0];
	for (int i = 1; i <= mode; ++i) newdims[i] = this->shape.dims[i-1];
	for (int i = mode+1; i < dim; ++i) newdims[i] = this->shape.dims[i];
	Tensor iReshape = reshape_internel(&mulRes, dim, newdims);
	return ipermute_internel(&iReshape, order);
}
Tensor scale(Tensor *this, const Tensor other, int n, ...) {
	// check shape info
	if (n != other.shape.ndim) {
		printf("Shapes of input don't match\n");
		exit(0);
	}

	int *axes = (int *)malloc(n * sizeof(int));
	va_list vl;
	va_start(vl, n);
	int i;
	for (i = 0; i < n; ++i) {
		int axis = va_arg(vl, int);
		axes[i] = axis;
		if (this->shape.dims[axis] != other.shape.dims[i]) break;
	}
	va_end(vl);
	if (i < n) {
		printf("Shapes of input don't match\n");
		exit(0);
	}

	if (n == this->shape.ndim) {
		int i;
		for (i = 0; i < n; ++i) {
			if (axes[i] != i) break;
		}
		if (i == n) return EWMul(*this, other);
	}

	// TODO: other situation
	return *this;
}

Tensor map(Tensor *this, Scalar(*func)(Scalar, void *), void *input) {
	Tensor res = zerosFromShape(this->dtype, this->shape);
	Iterator it = getIterator(*this);
	switch(this->dtype) {
		case INT: {
			int *data = (int *)res.data, *ib = (int *)begin(&it);
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = (int)func(*ib, input);
			}
			break;
		}
		case FLOAT: {
			float *data = (float *)res.data, *ib = (float *)begin(&it);
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = (float)func(*ib, input);
			}
			break;
		}
		case DOUBLE: {
			double *data = (double *)res.data, *ib = (double *)begin(&it);
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = (double)func(*ib, input);
			}
			break;
		}
		case DOUBLE_COMPLEX: {
			double complex *data = (double complex *)res.data, *ib = (double complex *)begin(&it);
			for (int i = 0, n = res.shape.nelem; i < n; ++i, ib = next(&it)) {
				*data++ = (double complex)func(*ib, input);
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
