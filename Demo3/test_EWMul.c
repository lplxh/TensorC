#include<tensor.h>
int main()
{
    Tensor a = ones(INT,2,3,4);
    Tensor b = zeros(INT,2,3,4);
    Tensor c = EWAdd(a,a);
    Tensor c1 = EWMul(c,c);
    Tensor d = EWMulTensorScalar(c,2);
    dump(&d);
    dump(&c1);
    return 0;
}

