/**
 * @brief Model testing script
 */
#include "model.h"
#include "customexceptions.h"
#include "matrix.h"
#include "testing.h"
#include "stats.h"
//#include <Rcpp.h>

auto main( ) -> int
{
    matrix obj1 = {
        {1,0,2},
        {2,0,3}, 
        {3,2,4},
        {6,5,10},
        {30,12,19}
    };

    auto obj2 = fitlm( obj1, 1, {2,3} );
    matrix obj2_result = { {0.7604167},{3.1354167},{-0.5625000} };
    expectT("Testing beta fit of 5x3 data", obj2.coef( ), obj2_result );
    
    return 0;
}