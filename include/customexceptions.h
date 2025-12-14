/**
 * @brief Tailored exceptions for matrix class and associated errors
 * @author Will Sharpsteen - wisharpsteen@gmail.com
 */
#include <exception>
#include <string>

#ifndef CUSTOMEXCEPTIONS_H
#define CUSTOMEXCEPTIONS_H

//---------------PARENTS----------------

class mathError : public std::exception
{
    private:
    std::string message;

    public:
    const char* what() const noexcept override {
        return message.c_str( );
    }
};

class dimError : public std::exception
{
    private:
    std::string message;

    public:
    const char* what() const noexcept override {
        return message.c_str( );
    }
};

//---------------CHILD MATH ERRORS----------------

class solutionError : public mathError
{
    private:
    std::string message;

    public:
    solutionError( const std::string& msg ) : message( msg ) { }

    const char* what() const noexcept override {
        return message.c_str( );
    }
};

class statsError : public mathError
{
    private:
    std::string message;

    public:
    statsError( const std::string& msg ) : message( msg ) { }

    const char* what() const noexcept override {
        return message.c_str( );
    }
};

class realError : public mathError
{
    private:
    std::string message;

    public:
    realError( const std::string& msg ) : message( msg ) { }

    const char* what() const noexcept override {
        return message.c_str( );
    }
};

//--------------CHILD DIM ERRORS----------------

class vectDimError : public std::exception 
{
    private:
    std::string message;

    public:
    vectDimError( const std::string& msg ) : message( msg ) { }

    const char* what() const noexcept override {
        return message.c_str( );
    }
};

class matrixDimError : public dimError
{
    private:
    std::string message;

    public:
    matrixDimError( const std::string& msg ) : message( msg ) { }

    const char* what() const noexcept override {
        return message.c_str( );
    }
};


#endif
