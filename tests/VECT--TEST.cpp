/**
 * @brief Vect testing script
 */
#include "vect.h"
#include "testing.h"
#include <typeinfo>
#include <sycl/sycl.hpp>

auto main( ) -> int
{
    using namespace SKAS::vect;

    //---------- a. construction

    vect< float > a1( {1,2,3,4}, false );
    expectT( "a1. testing for emptyness of created vector.", a1.isEmpty( ), false );

    auto a2 = a1;
    expectT( "a2. testing for equality of copy constructor.", a1, a2 );

    expectF( "a3. testing deep copy of copy constructor.", &a1, &a2 );

    vect< double > a4( 10, false );
    expectT( "a4. testing for correct sizing of dim-initilized vect.", a4.size( ), size_t{10} );

    vect< double > a5( {0,1,1,1,1,1,1,1,1,1}, false );
    double a5_zero = 0.0;
    expectT( "a5. testing for correct element grab on init.", a5[0], a5_zero );

    vect< double > a6_1( true );
    vect< double > a6_2;
    expectT( "a6. testing for parallel init.", a6_1, a6_2 );

    //---------- b. member functions

    vect< float > b1_1;
    vect< float > b1_2( {1,2,3}, false );
    b1_2.clear( );
    expectT( "b1. testing clear().", b1_1, b1_2 );

    vect< double > b2( {1,2}, false );
    expectT( "b2. testing size().", b2.size( ), size_t{2} );

    vect< float > b3_1( {}, false );
    vect< float > b3_2( {4}, false );
    b3_1.insert( b3_1.begin( ), 4 );
    expectT( "b3. testing insertion.", b3_1, b3_2 );

    vect< double > b4_1( {5,4,3,2,1}, false );
    vect< double > b4_2( {5,6,7,8,1}, false );
    expectT( "b4. testing data().", *b4_1.data( ), *b4_2.data( ) );

    expectT( "b5. testing enddata().", *(b4_1.enddata()-1), *(b4_2.enddata()-1) );

    //----------- c. sequential operations

    vect< double > c1_1 = {3,4,2};
    vect< double > c1_2 = {5,5,5};
    vect< double > c1_3 = {8,9,7};
    expectT( "c1. testing +.", c1_1 + c1_2, c1_3 );

    vect< double > c2 = {2,1,3};
    expectT( "c2. testing -.", c1_2 - c1_1, c2 );

    double c3_scale = 3;
    vect< double > c3 = {9,12,6};
    expectT( "c3. testing scalar*.", c1_1 * c3_scale, c3 );

    vect< double > c4_1 = {1,2,5,6};
    double c4_2 = 66;
    expectT( "c4. testing dot*.", c4_1 * c4_1, c4_2 );

    vect< float > c5_1 = {12,5};
    float c5_2 = 13;
    expectT( "c5. testing magnitude mag().", mag(c5_1), c5_2 );

    //std::cout << c5_1.castToLD( );
    //std::cout << typeid( c5_2.castToLD( )[0]).name( ) << std::endl;

    //expectT( "c6. testing type conversion.", typeid( c5_2.castToLD( )[0]).name( ), "double")

    //----------- d. parallel operations
    vect< double > d1_1( {1,2,3,4}, true );
    vect< double > d1_2( {4,3,2,1}, true );
    vect< double > d1_3( {5,5,5,5}, true );
    expectT( "d1. testing parallel +.", d1_1 + d1_2, d1_3 );

    expectT( "d2. testing parallel -.", d1_3 - d1_2, d1_1 );

    double d3 = 20;
    expectT( "d3. testing parallel *.", d1_1 * d1_2, d3 );

    double d4_1 = 1.0/5.0;
    vect< double > d4_2( {1,1,1,1}, true );
    expectT( "d4. testing parallel scale*.", d1_3 * d4_1, d4_2 );

    double d5 = sqrt(4.0);
    expectT( "d5. testing parallel magnitude.", mag( d4_2 ), d5 );

    vect< float > d6_1( {1,5,6,2}, false );
    vect< float > d6_2( {45,4,312,41}, false );
    vect< float > d6_3( {1,5,6,2}, true );
    vect< float > d6_4( {45,4,312,41}, true );
    expectT( "d6. testing parallel cov.", cov( d6_1, d6_2 ), cov( d6_3, d6_4 ) );

    vect< double > d7_1( {1,3,4,5,6}, true );
    double d7_2 = 3.7;
    expectT( "d7. testing parallel s2.", s2( d7_1 ), d7_2 );

    return EXIT_SUCCESS;
}