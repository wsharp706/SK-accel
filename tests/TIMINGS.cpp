#include "vect.h"
#include "matrix.h"
#include <chrono>
#include <fstream>
#include <iostream>

auto main( ) -> int
{
    /*
    std::ofstream outputData;
    outputData.open("../CPP_PMPlus_TIME.csv");
    unsigned long int sqdim = 20000;

    for ( int t_dim = 1; t_dim <= sqdim; t_dim += 50 )
    {
        SKAS::matrix::matrix< double > W( 0, t_dim, t_dim, true );
        outputData << t_dim << ",";
        //===========
        auto a1start = std::chrono::high_resolution_clock::now();
        //---
        W + W;
        //---
        auto a1end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(a1end - a1start).count();
        //===========
        outputData << duration << std::endl;
    }
    outputData.close( );
    */
    return EXIT_SUCCESS;
}
    

