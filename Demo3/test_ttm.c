#include <tensor.h>
int main()
{
    Tensor a = ones(DOUBLE,3,3,2,3);
//    dump(&a);
    Tensor b = ones(DOUBLE,2,3,3);
//    dump(&b);
    Tensor c = ttm(&a,0,b);
    dump(&c);
    return 0;
}
