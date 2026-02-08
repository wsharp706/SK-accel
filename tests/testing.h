/**
 * @brief Testing template function for easy tests
 * @author Will Sharpsteen - wisharpsteen@gmail.com
 */
#include <iostream>
#include <string>
#include <exception>

#ifndef TESTING_H
#define TESTING_H

class TESTFAILURE : public std::exception
{
    private:
    std::string message;

    public:
    TESTFAILURE( const std::string& msg ) : message( msg ) { }

    const char* what() const noexcept override {
        return message.c_str( );
    }
};

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
        std::cout << "\033[91m | obj_1 = \n" << obj_1 << " |\n\n | obj_2 = \n" << obj_2 << " |\033[0m" << std::endl;
        throw TESTFAILURE{ "\033[31m" + message + " -> FAILED: ARE NOT EQUAL.\033[0m" };
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
        std::cout << "\033[91m | obj_1 = \n" << obj_1 << " |\n\n| obj_2 = \n" << obj_2 << " |\033[0m" << std::endl;
        throw TESTFAILURE{ "\033[31m" + message + " -> FAILED: ARE EQUAL.\033[0m" };
    }
}

#endif