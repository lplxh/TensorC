#include <tensor.h>

int main(){
	Tensor t = zeros(DOUBLE, 2, 3, 3,4);
        dump(t);
	return 0;
}
