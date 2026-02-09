/**
 * @brief Matrix testing script
 */
#include "vect.h"
#include "matrix.h"
#include "testing.h"
#include <typeinfo>
#include <sycl/sycl.hpp>
#include <chrono>

auto main( ) -> int
{
    using namespace SKAS::matrix;

    //--------------- a. construction
    matrix< double > a1;
    expectT( "a1. testing for empty creation of matrix.", a1.is_empty( ), true );

    matrix< float > a2_1( { 1,2,3,
                            4,5,6,
                            7,8,9 }, 3, 3 );
    matrix< float > a2_2( a2_1 );
    expectT( "a2. testing copy and initializer list constructor.", a2_2, a2_1 );

    expectT( "a3. testing initilizer list construction.", a2_1.is_empty( ), false );

    a2_2.clear( );
    expectT( "a4. testing clear().", a2_2.is_empty( ), true );

    //--------------b. member utility
    expectT( "b1. testing ncol( ).", a2_1.ncol( ), size_t{3} );

    expectT( "b2. testing nrow( ).", a2_2.nrow( ), size_t{0} );

    matrix< double > b3_1( 5, 10, 10 );
    SKAS::vect::vect< double > b3_2 = {5,5,5,5,5,5,5,5,5,5};
    expectT( "b3. testing getrow( ).", b3_1.getrow( 0 ), b3_2 );

    SKAS::vect::vect< float > b4_1 = { 2, 5, 8 };
    expectT( "b4. testing getcol( ).", b4_1, a2_1.getcol( 1 ) );

    expectT( "b5. testing at( ) [getelem()] .", a2_1.at( 1, 2 ), float{6} );

    matrix< double > b6_1( {5,5,5,5,5,5,5,5,5,5,5,5}, 3, 4 );
    matrix< double > b6_2( {5,5,5,5,5,5,5,5,5,6,5,5}, 3, 4 );
    b6_1.setelem( 6, 2, 1 );
    expectT( "b6. testing setelem( ).", b6_1, b6_2 );

    matrix< float > b7_1( {1,1,2,2}, 2, 2 );
    matrix< float > b7_2( {1,1,2,2,3,3}, 3, 2 );
    std::vector< float > b7_4 = {3,3};
    b7_1.appendrow( b7_4 );
    expectT( "b7. testing appendrow( ).", b7_1, b7_2 );

    matrix< double > b8_1( {1,1,2,2,3,3}, 3, 2 ); 
    matrix< double > b13_1( b8_1 );
    matrix< double > b8_2( {1,1,4,4,2,2,3,3}, 4, 2 );
    std::vector< double > b8_3 = {4,4};
    b8_1.insertrow( b8_3, 1 );
    expectT( "b8. testing insertrow( ).", b8_1, b8_2 );

    matrix< float > b9_1( {1,1,1,2,2,2}, 2, 3 );
    matrix< float > b9_2( {1,1,1,1,2,2,2,2}, 2, 4 );
    SKAS::vect::vect< float > b9_3( {1,2} );
    b9_1.insertcol( b9_3, 4 );
    expectT( "b9. testing insertcol( ).", b9_1, b9_2 );

    matrix< double > b10_1( { 1,0,2,3,  0,1,0,2, 2,0,1,0,  3,2,0,1 }, 4, 4 );
    expectT( "b10. testing self transpose( ).", b10_1.t( ), b10_1 );

    matrix< float > b11_1( { 1,2,3 }, 1, 3 );
    matrix< float > b11_2( {1,2,3}, 3, 1 );
    expectT( "b11. testing t() on vector matricies.", b11_1.t( ), b11_2 );

    matrix< double > b12_1( {1,5,0,2,2,2,1,1,1}, 3, 3 );
    matrix< double > b12_2( {1,2,1,5,2,1,0,2,1}, 3, 3 );
    expectT( "b12. testing t() on square matricies.", b12_1.t( ), b12_2 );

    b8_2.droprow( 1 );
    expectT( "b13. testing droprow( ).", b8_2, b13_1 );

    matrix< float > b14_1( {1,2,3,1,2,3,1,2,3,1,2,3}, 4, 3 );
    matrix< float > b14_2( {1,3,1,3,1,3,1,3}, 4, 2 );
    b14_1.dropcol( 1 );
    expectT( "b14. testing dropcol( ).", b14_1, b14_2 );

    //----------------c. helper functions in matrix.h
    matrix< double > c1_1( {1,1,1,1,1,1}, 2, 3 );
    matrix< double > c1_2( {3,3,3,3,3,3}, 2, 3 );
    expectT( "c1. testing scale*( ).", c1_1 * double{3}, c1_2 );

    matrix< double > c2_1( {4,4,4,4,4}, 1, 5 );
    matrix< double > c2_2( {3,3,3,3,3}, 1, 5 );
    matrix< double > c2_3( {7,7,7,7,7}, 1, 5 );
    expectT( "c2. testing matrix +( ).", c2_1 + c2_2, c2_3 );

    matrix< float > c3_1({2,2,2,2,3,3,3,4,4}, 3, 3 );
    matrix< float > c3_2({2,2,2,2,3,3,3,4,4}, 3, 3 );
    matrix< float > c3_3( 0, 3, 3 );
    expectT( "c3. testing matrix -( ).", c3_1 - c3_2, c3_3 );

    matrix< float > c4_1({9,9,9,16},2,2);
    matrix< float > c4_2({3,3,3,4},2,2);
    expectT( "c4. testing sqrt( ).", sqrt(c4_1), c4_2 );

    matrix< double > c5_1({1,0,0,0, 0,1,0,0, 0,0,1,0 }, 3, 4 );
    matrix< double > c5_2({1,1,1},3,1);
    expectT( "c5. testing diag().", diag(c5_1), c5_2 );

    matrix< float > c6_1({1,0,0,0,1,0,0,0,1}, 3,3);
    expectT( "c6. testing identity().", identity< float >( size_t{3} ), c6_1 );

    matrix< double > c7_1({1,1,2, 3,4,0}, 2, 3);
    matrix< double > c7_2({8,9, 8,9, 4,5}, 3, 2);
    matrix< double > c7_3({24,28,56,63},2,2);
    expectT( "c7. testing matrix multiply().", c7_1 % c7_2, c7_3 );

    //-------------d. inversion
    matrix< double > d1_1({2,1,0,1,1,0,0,0,3}, 3, 3 );
    matrix< double > d1_2({1, -1, 0, -1, 2, 0, 0, 0, 1.0/3 }, 3, 3);
    expectT( "d1. testing pd matrix invert.", invert(d1_1,"spd"), d1_2);

    expectT( "d2. testing pd matrix inverse with qr.", invert(d1_1,"qr"), d1_2);

    matrix< double > d3_3({2,-3,0, -3,5,0, 0,0,float{1.0/9.0}},3,3);
    expectT("d3. testing qr matrix invert with qr(2).", invert(d1_1%d1_1,"qr"),d3_3);

    //-------------e. parallelization
    matrix< double > e1_1({1,1,2,1,1},1,5,true);
    matrix< double > e1_2({2,2,4,2,2},1,5,true);
    expectT( "e1. testing PM_scale.", e1_1 * double{2}, e1_2 );

    matrix< double > e2_1({3,3,6,3,3},1,5,true);
    expectT( "e2. testing PM_add.", e1_1 + e1_2, e2_1); 

    expectT( "e3. testing PM_sub.", e1_2 - e1_1, e1_1 );

    matrix< double > e4_1({2,2,34,3,134,213,4,3425,1324,3215,24,3245,129387,123,40987,987}, 4, 4, true );
    matrix< double > e4_2({2,2,34,3,134,213,4,3425,1324,3215,24,3245,129387,123,40987,987}, 4, 4, false );
    expectT( "e4. testing PM_mul.", e4_1 % e4_1, e4_2 % e4_2 );

    return EXIT_SUCCESS;
}