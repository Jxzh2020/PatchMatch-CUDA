#include <iostream>

#include "patch.h"

int main(){
    enum_gpu();
//  gen_cpu();
    while(1)
        gen_gpu();
    return 0;
}

