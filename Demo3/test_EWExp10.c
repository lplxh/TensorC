#include<tensor.h>
int main()
{
    Tensor a = ones(INT,3,2,2,2);
    Tensor b = ones(INT,3,2,2,2);
    Tensor c = EWAdd(a,b);
    Tensor d = EWExp10(&c);
    dump(&d);
    return 0;
}
