/**
 * @brief Vect testing script
 */
#include "vect.h"
#include "testing.h"
#include <sycl/sycl.hpp>

auto main( ) -> int
{
    // ----------------sequential ops----------------
    vect< float > flVect( {1,2,3}, true );
    std::cout << flVect << std::endl;

    auto flVect2 = flVect;
    flVect2.toPar( );
    std::cout << (flVect2 + flVect2) << std::endl;

    return EXIT_SUCCESS;
}