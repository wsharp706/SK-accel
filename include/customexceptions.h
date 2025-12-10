/**
 * @brief Tailored exceptions for matrix class and associated errors
 * @author Will Sharpsteen - wisharpsteen@gmail.com
 */
#include <exception>
#include <string>

#ifndef CUSTOMEXCEPTIONS_H
#define CUSTOMEXCEPTIONS_H

class vectorError : public std::exception 
{
    private:
    std::string message;

    public:
    vectorError( const std::string& msg ) : message( msg ) { }

    const char* what() const noexcept override {
        return message.c_str( );
    }
};

class dimSizeError : public std::exception 
{
    private:
    std::string message;

    public:
    dimSizeError( const std::string& msg ) : message( msg ) { }

    const char* what() const noexcept override {
        return message.c_str( );
    }
};

class solutionError : public std::exception 
{
    private:
    std::string message;

    public:
    solutionError( const std::string& msg ) : message( msg ) { }

    const char* what() const noexcept override {
        return message.c_str( );
    }
};

class statsError : public std::exception 
{
    private:
    std::string message;

    public:
    statsError( const std::string& msg ) : message( msg ) { }

    const char* what() const noexcept override {
        return message.c_str( );
    }
};

class realError : public std::exception 
{
    private:
    std::string message;

    public:
    realError( const std::string& msg ) : message( msg ) { }

    const char* what() const noexcept override {
        return message.c_str( );
    }
};

#endif
