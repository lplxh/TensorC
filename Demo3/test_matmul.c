#include <tensor.h>
int main()
{
    Tensor a = ones(DOUBLE,2,3,3);
    Tensor b = ones(DOUBLE,2,3,3);
//    dump(&a);
//    dump(&b);
    Tensor c = matmul(&a,b);
    dump(&c);
    return 0;
}
