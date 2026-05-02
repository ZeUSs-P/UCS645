#include "perf_num_validator.h"

int test_perfect(int candidate_num) {
    // Numbers less than 2 cannot be perfect numbers
    if (candidate_num <= 1) {
        return -candidate_num;
    }
    
    int accumulated_sum = 1; // 1 is always a proper divisor for n > 1
    int div = 2;
    
    // Check divisors up to the square root of the number
    while (div * div <= candidate_num) {
        if (candidate_num % div == 0) {
            accumulated_sum += div;
            
            int paired_div = candidate_num / div;
            if (paired_div != div) {
                accumulated_sum += paired_div;
            }
        }
        div++;
    }
    
    // Check if the sum of proper divisors equals the original number
    if (accumulated_sum == candidate_num) {
        return candidate_num;
    }
    
    return -candidate_num;
}
