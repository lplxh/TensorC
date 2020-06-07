#include <tensor.h>
int main()
{
    Tensor a = ones(FLOAT,2,2,3);
    Tensor b = castTo(a,DOUBLE);
    dump(&a);
    dump(&b);
    return 0;
}
