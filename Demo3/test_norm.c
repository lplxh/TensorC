#include <tensor.h>
int main()
{
    Tensor a = ones(FLOAT,2,3,3);
    Tensor b = ones(FLOAT,2,3,3);
    Tensor c = EWAdd(a,b);
    double ans = norm (&c);
    printf("%f\n",ans);
    return 0;
}
