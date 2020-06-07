#include<tensor.h>
int main()
{
    Tensor a = ones(FLOAT,2,2,3);
    int d = dim(&a);
    int s = size(&a,1);
    int n = nelem(&a);
    printf("%d %d %d\n",d,s,n);
    return 0;
}
