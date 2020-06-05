#include "scalartype.h"
#include <complex.h>
#include <stdio.h>

int getTypeSize(ScalarType type) {
	switch(type) {
		case INT: return sizeof(int);
		case FLOAT: return sizeof(float);
		case DOUBLE: return sizeof(double);
		case DOUBLE_COMPLEX: return sizeof(double complex);
		default: {
			printf("Unsupported data type\n");
			return -1;
		}
	}
}

ScalarType getCommonType(ScalarType type1, ScalarType type2) {
	if (type1 == type2) return type1;
	return type1 > type2 ? type1 : type2;
}
