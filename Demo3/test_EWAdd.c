#include<tensor.h>
int main()
{
    Tensor a = ones(INT,2,3,4);
    Tensor b = zeros(INT,2,3,4);
    Tensor c = EWAdd(a,b);
    Tensor c1 = EWAdd(c,a);
    Tensor d = EWAddTensorScalar(a,1);
    dump(&d);
    dump(&c1);
    return 0;
}

