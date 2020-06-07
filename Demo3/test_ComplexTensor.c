#include <tensor.h>
int main()
{
    Tensor real = zeros(FLOAT,2,3,4);
    Tensor imag = ones(FLOAT,2,3,4);
    Tensor ans = allocComplexTensor(real,imag);
    dump(&ans);
    return 0;
}
