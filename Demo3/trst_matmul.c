#include <tensor.h>
int main()
{
    Tensor a = ones(DOUBLE,3,3);
    Tensor b = ones(DOUBLE,3,3);
    Tensor c = matmul(a,b);
    dump(&c);
    return 0;
}
