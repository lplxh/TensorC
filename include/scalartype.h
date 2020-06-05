#ifndef TENSOR_SCALARTYPE_H
#define TENSOR_SCALARTYPE_H

#include <complex.h>

typedef enum {
	INT,
	FLOAT,
	DOUBLE,
	DOUBLE_COMPLEX
} ScalarType;

typedef double complex Scalar;

int getTypeSize(ScalarType type);
ScalarType getCommonType(ScalarType type1, ScalarType type2);

#endif
