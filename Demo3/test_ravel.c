#include <tensor.h>
int main()
{
    Tensor a = ones(DOUBLE,2,3,3);
    Tensor b = ravel(&a);
    dump(&b);    
    return 0;
}
