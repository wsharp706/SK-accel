/**
 * @brief Matrix testing script
 */

#include "customexceptions.h"
#include "matrix.h"
#include "testing.h"
//#include <Rcpp.h>

auto main( ) -> int
{
    //copy assignment checks
    matrix obj1 = {
        {1,2,3},
        {2,3,1},
        {3,2,1}
    };
    matrix obj1_copy = obj1;
    expectT("Testing matrix copy assignment of square matrix", obj1, obj1_copy);

    matrix obj2;
    auto obj2_copy = obj2;
    expectT("Testing matrix copy assignment of empty matrix", obj2, obj2_copy);

    matrix obj3 = { {1},{2},{4} };
    auto obj3_copy = obj3;
    expectT("Testing matrix copy assignment of column matrix", obj3, obj3_copy);

    matrix obj4 = { { 1, 2, 4 } };
    auto obj4_copy = obj4;
    expectT("Testing matrix copy assignment of row matrix", obj4, obj4_copy);

    //matrix multiply checks
    matrix obj5_result = { {6},{6},{6} };
    matrix obj5 = { {1},{1},{1} };
    expectT( "Testing matrix multiply for unequal dimension matrix", obj1 % obj5, obj5_result );

    matrix obj6_mresult = { 
        {14,14,8},
        {11,15,10}, 
        {10,14,12} 
    };
    expectT("Testing matrix multiply of square matrix", obj6_mresult, obj1 % obj1);

    expectT("Testing matrix multiply of empty matrix against empty", obj2, obj2 % obj2);

    //matrix transpose checks
    expectT("Testing matrix transpose of row matrix", obj3, obj4.t( ) );

    expectT("Testing matrix transpose of column matrix", obj3.t( ), obj4 );

    expectT("Testing matrix transpose of empty matrix", obj2.t( ), obj2 );

    matrix obj6_tresult = {
        {14,11,10},
        {14,15,14},
        {8,10,12}
    };
    expectT("Testing matrix transpose of square matrix", obj6_tresult, obj6_mresult.t( ) );

    //matrix init checks
    matrix obj7_result = {
        {4,4},
        {4,4},
        {4,4}
    };
    expectT("Testing matrix init on full matrix", obj7_result, init( 4, 3, 2 ) );

    matrix obj8_result = {
        {1,0,0},
        {0,1,0},
        {0,0,1}
    };
    expectT("Testing identity init of 3 dim", obj8_result, identity( 3 ) );

    matrix obj9_result = {
        {1}
    };
    expectT("Testing identity init of odd dim", obj9_result, identity( 1 ) );

    expectT("Testing identity init of empty matrix", obj2, identity( 0 ) );

    //append checks
    matrix obj10_result = {
        {1,2},
        {2,1},
        {3,3}
    };
    matrix obj10_step = {
        {1},
        {2},
        {3}
    };
    std::vector< long double > obj10_step2 = {2,1,3};
    obj10_step.appendcol( obj10_step2 );
    expectT("Testing append col to full matrix", obj10_result, obj10_step );

    matrix obj11;
    matrix obj11_result = {
        {2},
        {1},
        {3}
    };
    obj11.appendcol( obj10_step2 );
    expectT("Testing append col to empty matrix", obj11_result, obj11 );

    //matrix invert checks
    expectT("Testing matrix invert on odd-dim Identity", invert( identity( 5 ), "spd" ), identity( 5 ) );

    matrix obj12 = {
        {2,1},
        {1,3}
    };
    matrix obj12_result = { 
        {0.6,-0.2},
        {-0.2,0.4}
    };

    auto obj12_inv = invert( obj12, "spd" );
    expectT("Testing matrix invert on full even-dim matrix", invert( obj12 ), obj12_result );

    matrix obj13;
    expectT("Testing matrix invert on empty matrix", obj13, invert( obj13, "spd") );

    expectT("Testing matrix invert on 1-dim matrix", invert( init( 4.2884, 1, 1 ), "spd" ), init( 1.0 / 4.2884, 1, 1 ) );

    expectT("Testing matrix invert invert A to A", invert( invert( obj12 ) ), obj12 );

    return EXIT_SUCCESS;
}