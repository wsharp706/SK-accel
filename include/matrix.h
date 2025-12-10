/**
 * @brief Declaration of matrix class and associated member functions
 * @author Will Sharpsteen - wisharpsteen@gmail.com
 */
#include <iostream>
#include <vector>

#ifndef MATRIX_H
#define MATRIX_H

class matrix
{
    private:
    int dim_n; //row size
    int dim_m; //col size
    std::vector< std::vector< long double > > data;

    public:
    matrix( );

    ~matrix( );

    matrix( const matrix &original );

    //template < typename T >
    matrix( std::initializer_list< std::initializer_list< long double > > init ) : dim_n{ 0 }, dim_m{ 0 }
    {
        std::vector< long double > temp;
        for ( const auto& row : init ) 
        {
            for ( const auto& val : row )
            {
                temp.push_back( static_cast< long double >( val ) );
            }
            appendrow( temp );
            temp.clear( );
        }
    }

    /**
     * @brief full memory clear of matrix
     * @return void
     */
    auto clear( ) -> void;

    /**
     * @brief row count (vertical dimension) of matrix
     * @return int
     */
    auto nrow( ) const -> int;

    /**
     * @brief col count (horizontal dimension) of matrix
     * @return int
     */
    auto ncol( ) const -> int;

    /**
     * @brief retrieve row at selected index in matrix
     * @param index int. index to get row
     * @return std::vector< long double > row from matrix
     * @exception dimSizeError for index outside matrix dimension
     */
    auto getrow( int index ) const -> std::vector< long double >;

    /**
     * @brief retrieve col at selected index in matrix
     * @param index int. index to get col
     * @return std::vector< long double > col from matrix
     * @exception dimSizeError for index outside matrix dimension
     */
    auto getcol( int index ) const -> std::vector< long double >;

    /**
     * @brief retrieve element at selected index of in matrix
     * @param row int
     * @param col int
     * @param index bool. index mode or position mode. default = false. index + 1 = position
     * @return long double value at selected index
     * @exception dimSizeError thrown when indicies outside matrix dimensions
     */
    auto getelem( int row, int col, bool index = false ) const -> long double;

    /**
     * @brief set element of matrix at selected index
     * @param value long double. value to set at index
     * @param row int. row index to place value
     * @param col int. col index to place value
     * @param index bool. index mode or position mode. default = false. index + 1 = position
     * @exception dimSizeError thrown when indicies outside matrix dimensions
     */
    auto setelem( long double value, int row, int col, bool index = false ) -> void;

    /**
     * @brief append row to bottom of matrix
     * @param t_row std::vector< long double > row to append
     * @exception dimSizeError thrown when size of t_row is not compatible with dimensions of matrix 
     */
    auto appendrow( std::vector< long double > t_row ) -> void;

    /**
     * @brief insert row into matrix at index
     * @param t_row std::vector< long double > row to insert
     * @param index row index to insert into
     * @exception dimSizeError thrown when either index is outside of matrix or if row size is not compatible with matrix
     */
    auto insertrow( std::vector< long double > t_row, int index, int quantity = 1 ) -> void;

    /**
     * @brief insert col into matrix at index
     * @param t_row std::vector< long double > col to insert
     * @param index col index to insert into
     * @exception dimSizeError thrown when either index is outside of matrix or if col size is not compatible with matrix
     */
    auto insertcol( std::vector< long double > t_col, int index, int quantity = 1 ) -> void;

    /**
     * @brief append col to right of matrix
     * @param t_col std::vector< long double > col to append
     * @exception dimSizeError thrown when size of t_col is not compatible with dimensions of matrix 
     */
    auto appendcol( std::vector< long double > t_col ) -> void;
    
    /**
     * @brief transpose matrix
     * @return matrix post transposed
     */
    auto t( ) const -> matrix;

    auto operator==( const matrix& c_matrix ) const -> bool;

    /**
     * @brief drops row at selected row index
     * @param index index to row to drop
     * @exception dimSizeError thrown when row to drop is outside of matrix
     */
    auto droprow( int index ) -> void;

    /**
     * @brief drops col at selected col index
     * @param index index to drop col
     * @exception dimSizeError thrown when col to drop is outside of matrix
     */
    auto dropcol( int index ) -> void;

    /**
     * @brief "sticks" all elements of o_matrix below current matrix.
     * @param o_matrix matrix to insert as elements below current matrix.
     * @exception dimSizeError thrown for incompatible sizings.
     */
    auto shove_below( const matrix& o_matrix ) -> void;
};

auto operator<<( std::ostream& os, const matrix& t_matrix ) -> std::ostream&;

/**
 * @brief matrix scale support
 * @param t_matrix matrix to scale
 * @param scalar long double
 * @return scaled matrix
 */
auto operator*( const matrix& t_matrix, long double scalar ) -> matrix;

/**
 * @brief compute covariance between two vectors
 * @param a_vec 
 * @param b_vec
 * @exception dimSizeError thrown for incompatible sizings.
 * @return output long double C(a,b)
 */
auto cov( const std::vector< long double >& a_vec, const std::vector< long double >& b_vec ) -> long double;

/**
 * @brief compute correlation between two vectors
 * @param a_vec 
 * @param b_vec
 * @exception dimSizeError thrown for incompatible sizings.
 * @return output long double C(a,b)
 */
auto corr( const std::vector< long double >& a_vec, const std::vector< long double >& b_vec ) -> long double;
/**
 * @brief matrix scale support
 * @param t_matrix matrix to scale
 * @param scalar long double
 * @return scaled matrix
 */
auto operator*( long double scalar, const matrix& t_matrix ) -> matrix;

/**
 * @brief matrix addition support
 * @param a_matrix left matrix to add
 * @param b_matrix right matrix to add
 * @return matrix result
 * @exception dimSizeError thrown when mismatched dimensions
 */
auto operator+( const matrix& a_matrix, const matrix& b_matrix ) -> matrix;

/**
 * @brief matrix subtraction support
 * @param a_matrix left matrix to subtract
 * @param b_matrix right matrix to subtract
 * @return matrix result
 * @exception dimSizeError thrown when mismatched dimensions
 */
auto operator-( const matrix& a_matrix, const matrix& b_matrix ) -> matrix;

/**
 * @brief matrix sqrt support
 * @param a_matrix matrix to root all elements
 * @return matrix result
 * @exception realError thrown when sqrt applied on negative number
 */
auto sqrt( const matrix& a_matrix ) -> matrix;

/**
 * @brief initilized matrix with given initial value and dimensions
 * @param initial_value long double
 * @param rowcount int
 * @param colcount int
 * @return matrix
 */
auto init( long double initial_value, int rowcount, int colcount ) -> matrix;

/**
 * @brief returns an n by 1 matrix of just the diagonials of t_matrix
 * @param t_matrix matrix
 * @return matrix  
 */
auto diag( const matrix& t_matrix ) -> matrix;

/**
 * @brief initilizes identity matrix of given dimension
 * @param dim int
 * @return matrix
 */
auto identity( int dim ) -> matrix;

/**
 * @brief matrix multiplication
 * @param a_matrix left side matrix to multiply. 
 * @param b_matrix right side matrix to multiply.
 * @return matrix post-multiplication
 * @exception dimSizeError for matricies of incompatible dimensions 
 */
auto operator%( const matrix& a_matrix, const matrix& b_matrix ) -> matrix;

/**
 * @brief matrix inversion. as of 11-25-2025 only type symmetric positive definite supported. this function is expensive; save copy if needed in repetition.
 * @param a_matrix matrix to invert
 * @param type std::string. type of matrix. only "spd" supported currently
 * @exception dimSizeError if non-square using "spd" type
 * @exception solutionError if singularity detected
 * @return matrix
 */
auto invert( const matrix& a_matrix, std::string type = "qr" ) -> matrix;

/**
 * @brief addition operator support for std::vector
 * @param first std::vector< long double >
 * @param last std::vector< long double >
 * @return std::vector< long double > result from operation
 * @exception dimSizeError thrown when first.size( ) != last.size( ) 
 */
auto operator+( const std::vector< long double >& first, const std::vector< long double >& last ) -> std::vector< long double >;

/**
 * @brief subtraction operator support for std::vector
 * @param first std::vector< long double >
 * @param last std::vector< long double >
 * @return std::vector< long double > result from operation
 * @exception dimSizeError thrown when first.size( ) != last.size( ) 
 */
auto operator-( const std::vector< long double >& first, const std::vector< long double >& last ) -> std::vector< long double >;

/**
 * @brief dot product support for std::vector
 * @param first std::vector< double >
 * @param last std::vector< double >
 * @return long double
 * @exception vectorError thrown for incompatible sizes
 */
auto operator*( const std::vector< long double >& first, const std::vector< long double >& last ) -> long double;

/**
 * @brief scalar support for std::vector
 * @param t_vec vector of long double to scale
 * @param scalar long double
 * @return std::vector< long double > scaled output
 */
auto operator*( const std::vector< long double>& t_vec, const long double scalar ) -> std::vector< long double >;

/**
 * @brief scalar support for std::vector
 * @param t_vec vector of long double to scale
 * @param scalar long double
 * @return std::vector< long double > scaled output
 */
auto operator*( const long double scalar, const std::vector< long double >& t_vec ) -> std::vector< long double >;

/**
 * @brief Euclidean norm operator
 * @param t_vec vector of type long double
 * @return long double magnitude
 */
auto mag( const std::vector< long double >& t_vec ) -> long double;

/**
 * @brief Creates new vector of same direction with unit length
 * @param t_vec vector to scale
 * @return unit vector of same length
 */
auto makeunit( const std::vector< long double >& t_vec ) -> std::vector< long double >;

/**
 * @brief comparison tool for high-precision vectors
 * @param first std::vector< double >
 * @param last std::vector< double >
 * @param error amount of accepted error between elements. default = 0.00001
 * @return boolean answer of comparison
 */
auto vcomp( const std::vector< long double >& first, const std::vector< long double >& last, float error = 0.00001 ) -> bool;

/**
 * @brief Welford's Method application of single-pass variance
 * @param t_vector std::vector< long double >
 * @return long double s^2
 */
auto s2( const std::vector< long double >& t_vector ) -> long double;

/**
 * @brief Welford's Method application of single-pass standard deviation
 * @param t_vector std::vector< long double >
 * @return long double s
 */
auto s( const std::vector< long double >& t_vector ) -> long double;

/**
 * @brief Compute mean of vector
 * @param t_vec vector of numeric
 */
auto mean( const std::vector< long double >& t_vec ) -> long double;

//template < typename T >
auto operator<<( std::ostream& os, const std::vector< long double >& vec ) -> std::ostream&;

auto operator<<( std::ostream& os, const std::vector< unsigned int >& vec ) -> std::ostream&;

/**
 * @brief solves Ax = b for lower triangular matrix
 * @param lower_matrix triangular matrix to solve. singularity is checked.
 * @param b vector to solve
 * @return the solution x for Ax = b
 * @exception solutionError thrown for singularity
 */
auto forwardsolve( const matrix& lower_matrix, const std::vector< long double >& b ) -> std::vector< long double >;

/**
 * @brief solves Ax = b for upper triangular matrix
 * @param upper_matrix triangular matrix to solve. singularity is checked.
 * @param b vector to solve
 * @return the solution x for Ax = b
 * @exception solutionError thrown for singularity
 */
auto backsolve( const matrix& upper_matrix, const std::vector< long double >& b ) -> std::vector< long double >;

/**
 * @brief inverts triangular matrix
 * @param t_matrix triangular matrix to invert. singularity is checked.
 * @param lower true for lower triangular matrix, false for upper
 * @return the inverse of t_matrix
 * @exception solutionError thrown for singularity
 */
auto triangularinvert( const matrix& t_matrix, bool lower ) -> matrix;

/**
 * @brief QR decomposition of matrix
 * @param t_matrix matrix to decompose
 * @exception solutionError thrown for singularity
 * @return vector of two matricies: Q, R; where t_matrix = QR
 */
auto qr_decomp( const matrix& t_matrix ) -> std::vector< matrix >;

/**
 * @brief Recursive utility for qr_decomp
 */
auto qr_dive( const matrix& t_matrix, unsigned int pivot, std::vector< matrix >& container ) -> void;

#endif