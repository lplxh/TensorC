#include <tensor.h>
int main()
{
    Tensor a = ones(FLOAT,2,3,4);
    Tensor b = ones(DOUBLE,2,3,4);
    ScalarType type = castToCommonType(&a,&b);
    printf("%d\n",type);
    return 0;
}
