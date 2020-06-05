#ifndef TENSOR_TENSOR_H
#define TENSOR_TENSOR_H

#include "shape.h"
#include "scalartype.h"
#include <stdio.h>
#include <stdint.h>


typedef struct Tensor Tensor;
struct Tensor {
	bool valid;
	void *data;
	ScalarType dtype;
	Shape shape;
};

// get functions
int dim(Tensor *this);
int size(Tensor *this, int d);
int nelem(Tensor *this);
ScalarType getDataType(Tensor *this);
void *data_ptr(Tensor *this);
	
// element-wise operations
Tensor EWAdd(const Tensor lhs, const Tensor rhs);
Tensor EWAdd_out(Tensor*output,const Tensor lhs, const Tensor rhs);
Tensor EWMul(const Tensor lhs, const Tensor rhs);
Tensor EWMul_out(Tensor*output,const Tensor lhs,const Tensor rhs);
Tensor EWAddTensorScalar(const Tensor lhs, Scalar number);
Tensor EWAddTensorScalar_out(Tensor*output,const Tensor lhs,Scalar number);
Tensor EWAddScalarTensor(Scalar number, const Tensor rhs);
Tensor EWNegative(const Tensor tensor);
Tensor EWNegative_(Tensor* tensor);
Tensor EWNegative_out(Tensor*output,const Tensor tensor);
Tensor EWSquare(Tensor *this);//addupdate
Tensor EWAbs(Tensor *this);
Tensor EWAbs_(Tensor*this);
Tensor EWAbs_out(Tensor*output,Tensor*this);
Tensor EWSquare_(Tensor*this);
Tensor EWSquare_out(Tensor*output,Tensor*this);

/*TODO*/
Tensor EWLog10(Tensor *this);

//Tensor matmulvec(Tensor *this, const Tensor other);

/*updated*/
Tensor EWSub(const Tensor lhs, const Tensor rhs);//addupdate
Tensor EWSub_out(Tensor*output,const Tensor lhs,const Tensor rhs);
Tensor EWSubTensorScalar(const Tensor lhs, Scalar number);
Tensor EWSubTensorScalar_out(Tensor*output,const Tensor lhs,Scalar number);
Tensor EWSubScalarTensor(Scalar num, const Tensor rhs);
Tensor EWMulTensorScalar(const Tensor lhs, Scalar number);//addupdate
Tensor EWMulTensorScalar_out(Tensor*output,const Tensor lhs,Scalar number);
Tensor EWMulScalarTensor(Scalar number, const Tensor rhs);//addupdate
Tensor EWGEScalar(const Tensor lhs, Scalar);//addupdate
Tensor EWGTScalar(const Tensor a, Scalar num);
Tensor EWDiv(const Tensor a, const Tensor b);
Tensor EWDiv_out(Tensor*output,const Tensor a,const Tensor b);
Tensor EWDivTensorScalar(const Tensor a, Scalar num);
Tensor EWDivTensorScalar_out(Tensor*output,const Tensor a ,Scalar num);
Tensor EWExp(Tensor *this);
Tensor EWExp10(Tensor *this);

// permute operations
Tensor ravel(Tensor *this);
Tensor ravel_out(Tensor*output,Tensor*this);
Tensor transpose(Tensor *this);
Tensor transpose_out(Tensor* output,Tensor*this);
Tensor reshape(Tensor *this, int n, ...);
Tensor reshape_out(Tensor* output,Tensor*this,int n,...);
Tensor reshape_internel(Tensor *this, int n, int *dims);
Tensor permute(Tensor *this, int n, ...);
Tensor permute_out(Tensor*output,Tensor*this,int n,...);
Tensor permute_internel(Tensor *this, int *axes);
Tensor ipermute_internel(Tensor *this, int *axes);

// matrix operations
Tensor matmul(Tensor *this, const Tensor other);
Tensor diag(Tensor *this);  // 若输入张量是矩阵，则返回对角线元素的向量；若输入一维向量，则生成对角线元素是该向量的矩阵
Tensor pinv(Tensor *this);
Tensor *dsvd(Tensor *this);

/*updated*/


// tensor operations
double norm(Tensor *this);
Tensor ttm(Tensor *this, int mode, const Tensor matrix);
Tensor scale(Tensor *this, const Tensor other, int n, ...); //scale along specified dimensions of tensor
Tensor map(Tensor *this, Scalar(*func)(Scalar, void *), void *input);
Tensor unfold(Tensor*this,int mode); //new
Scalar innerproduct(Tensor *this, const Tensor other);
Tensor outerproduct(Tensor*this,const Tensor other);  //new

// 聚合统计类函数
Scalar maxOfTensor(Tensor *this);
Scalar sumOfTensor(Tensor *this);
Scalar minOfTensor(Tensor*this);
Scalar meanOfTensor(Tensor*this);
Scalar stdOfTensor(Tensor*this);

// dump
void dump(const Tensor *this);

// slice
Tensor select(Tensor *this, unsigned dim, int64_t index);
Tensor slice(Tensor *this, unsigned dim, int64_t start, int64_t step, int64_t end);

// copy data (assign operator)
Tensor copy(Tensor *dst, const Tensor src);
Tensor copyScalar(Tensor *dst, Scalar src);

// 创建张量
Tensor fromdata(void*data, ScalarType type, int n,...) ; //new
Tensor fromfile(FILE *fd, ScalarType type, int n);
Tensor ones(ScalarType type, int n, ...);
Tensor zeros(ScalarType type, int n, ...);
Tensor arange(ScalarType type, Scalar start, Scalar end, Scalar step);
Tensor onesFromShape(ScalarType type, Shape shape);
Tensor zerosFromShape(ScalarType type, Shape shape);
Tensor allocComplexTensor(Tensor real, Tensor imag);
Tensor castTo(Tensor tensor, ScalarType type);
ScalarType castToCommonType(Tensor *lhs, Tensor *rhs);
Tensor full(ScalarType type,Scalar num,int n,...);

#endif
