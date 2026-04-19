#include "prime_num_validator.h"

int test_prime(int value) {
    if (value <= 1) {
        return -value;
    }
    if (value == 2 || value == 3) {
        return value;
    }
    if (value % 2 == 0 || value % 3 == 0) {
        return -value;
    }
    
    int divisor = 5;
    while (divisor * divisor <= value) {
        if (value % divisor == 0 || value % (divisor + 2) == 0) {
            return -value;
        }
        divisor += 6;
    }
    
    return value;
}
