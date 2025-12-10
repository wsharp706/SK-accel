/**
 * @brief Testing template function for easy tests
 * @author Will Sharpsteen - wisharpsteen@gmail.com
 */
#include <iostream>
#include <string>

#ifndef TESTING_H
#define TESTING_H

template < typename T >
auto expectT( std::string message, const T& obj_1, const T& obj_2 ) -> void
{
    std::cout << "\033[33m[<][>][<][>][<]    " << message << "    [>][<][>][<][>]\033[0m" << std::endl;
    if ( obj_1 == obj_2 )
    {
        std::cout << "\033[32m[<][>][<][>][<]    PASSED    [>][<][>][<][>]\033[0m" << std::endl;
    }
    else
    {
        std::cout << "\033[31m[<][>][<][>][<]    FAILED    [>][<][>][<][>]\033[0m" << std::endl;
    }
}

template < typename T >
auto expectF( std::string message, const T& obj_1, const T& obj_2 ) -> void
{
    std::cout << "\033[33m[<][>][<][>][<]    " << message << "    [>][<][>][<][>]\033[0m" << std::endl;
    if ( obj_1 != obj_2 )
    {
        std::cout << "\033[32m[<][>][<][>][<]    PASSED    [>][<][>][<][>]\033[0m" << std::endl;
    }
    else
    {
        std::cout << "\033[31m[<][>][<][>][<]    FAILED    [>][<][>][<][>]\033[0m" << std::endl;
    }
}

auto expectT( std::string message, const std::vector< long double >& obj_1, const std::vector< long double >& obj_2, const long double error = 0.0000001 ) -> void
{
    std::cout << "\033[33m[<][>][<][>][<]    " << message << "    [>][<][>][<][>]\033[0m" << std::endl;
    if ( vcomp( obj_1, obj_2, error ) )
    {
        std::cout << "\033[32m[<][>][<][>][<]    PASSED    [>][<][>][<][>]\033[0m" << std::endl;
    }
    else
    {
        std::cout << "\033[31m[<][>][<][>][<]    FAILED    [>][<][>][<][>]\033[0m" << std::endl;
    }
}

auto expectF( std::string message, const std::vector< long double >& obj_1, const std::vector< long double >& obj_2, const long double error = 0.0000001 ) -> void
{
    std::cout << "\033[33m[<][>][<][>][<]    " << message << "    [>][<][>][<][>]\033[0m" << std::endl;
    if ( !vcomp( obj_1, obj_2, error ) )
    {
        std::cout << "\033[32m[<][>][<][>][<]    PASSED    [>][<][>][<][>]\033[0m" << std::endl;
    }
    else
    {
        std::cout << "\033[31m[<][>][<][>][<]    FAILED    [>][<][>][<][>]\033[0m" << std::endl;
    }
}


#endif