/**
 * @brief Declaration of matrix class and associated member functions
 * @author Will Sharpsteen - wisharpsteen@gmail.com
 */
#include <iostream>
#include <vector>
#include <concepts>
#include <algorithm>
#include <cmath>
#include <sycl/sycl.hpp>
#include <hipSYCL/algorithms/numeric.hpp>
#include <hipSYCL/algorithms/algorithm.hpp>
#include "templates.h"
#include "gpu.h"
#include "customexceptions.h"
#include "vect.h"

#ifndef MATRIX_H
#define MATRIX_H

namespace SKAS::matrix
{
    template < SKAS::FlAd T >                    
    class matrix
    {
        private:
        size_t dim_n; //row size
        size_t dim_m; //col size
        SKAS::vect::vect< T > data;
        bool parallel;

        public:
        matrix( ) : dim_n( 0 ), dim_m( 0 ), parallel( false ) { };

        ~matrix( ) { };

        matrix( const matrix &original )
        {
            clear( );
            data = original.data;
            dim_m = original.dim_m;
            dim_n = original.dim_n;
            parallel = original.parallel;
        }   

        matrix( const vect::vect< T >& t_data, const size_t row_dim, const size_t col_dim, const bool is_parallel = false )
        {
            data = t_data;
            dim_n = row_dim;
            dim_m = col_dim;
            parallel = is_parallel;
        }

        matrix( std::initializer_list< T > init, const size_t row_dim, const size_t col_dim, const bool is_parallel = false )
        {
            vect::vect< T > outdata( init );
            data = outdata;
            dim_n = row_dim;
            dim_m = col_dim;
            parallel = is_parallel;
        }

        matrix( T initial_value, const size_t& rowcount, const size_t& colcount, const bool is_parallel = false )
        {
            vect::vect< T > outdata(rowcount * colcount, initial_value, parallel );
            data = outdata;
            dim_m = colcount;
            dim_n = rowcount;
            parallel = is_parallel;
        }

        /**
         * @brief full memory clear of matrix
         * @return void
         */
        auto clear( ) -> void
        {
            data.clear( );
            dim_n = 0;
            dim_m = 0;
        }   

        /**
         * @brief row count (vertical dimension) of matrix
         * @return int
         */
        auto nrow( ) const -> size_t
        {
            return dim_n;
        }  

        /**
         * @brief col count (horizontal dimension) of matrix
         * @return int
         */

        auto ncol( ) const -> size_t
        {
            return dim_m;
        }

        /**
         * @brief retrieve copy of row at selected index in matrix
         * @param index int. index to get row
         * @return std::vector< long double > row from matrix
         * @exception matrixDimError for index outside matrix dimension
         */

        auto getrow( const size_t& index ) const -> vect::vect< T >
        {
            if ( index >= nrow( ) )
            {
                throw matrixDimError{"CANNOT RETREIVE ROW OUTSIDE MATRIX"};
            }
            vect::vect< T > output( ncol( ) );
            for ( int i = 0; i < ncol( ); ++i )
            {
                output[ i ] = data[ i + ncol( ) * index ];
            }
            return output;
        }

        /**
         * @brief retrieve copy of col at selected index in matrix
         * @param index int. index to get col
         * @return std::vector< long double > col from matrix
         * @exception matrixDimError for index outside matrix dimension
         */
        auto getcol( const size_t& index ) const -> vect::vect< T >
        {
            if ( index >= ncol( ) )
            {
                throw matrixDimError{"CANNOT RETREIVE COL OUTSIDE MATRIX"};
            }
            vect::vect< T > final( nrow( ) );
            for ( int i = 0; i < nrow( ); ++i )
            {
                final[ i ] = data[ index + i * ncol( ) ];
            }
            return final;
        }

        /**
         * @brief retrieve element at selected index of in matrix
         * @param row int
         * @param col int
         * @return data type T returned at index
         * @exception matrixDimError thrown when indicies outside matrix dimensions
         */
        auto getelem( const size_t& row, const size_t& col ) const -> T
        {
            if ( row >= nrow( ) || col >= ncol( ) )
            {
                throw matrixDimError{"CANNOT getelem OUTSIDE OF MATRIX DIMENSIONS"};
            }
            return data[ col + row * ncol( ) ];
        }

        auto at( const size_t& row_index, const size_t& col_index ) const -> T
        {
            return getelem( row_index, col_index );
        }

        /**
         * @brief set element of matrix at selected index
         * @param value data type T set at index
         * @param row int. row index to place value
         * @param col int. col index to place value
         * @exception matrixDimError thrown when indicies outside matrix dimensions
         */

        auto setelem( const T& value, const size_t& row, const size_t& col ) -> void
        {
            if ( row >= nrow( ) || col >= ncol( ) )
            {
                throw matrixDimError{"CANNOT setelem OUTSIDE OF MATRIX DIMENSIONS"};
            }
            data[ col + row * ncol( ) ] = value;
        }

        /**
         * @brief insert row into matrix at index
         * @param t_row std::vector< T > row to insert
         * @param index row index to insert into
         * @exception matrixDimError thrown when either index is outside of matrix or if row size is not compatible with matrix
         */

        auto insertrow( std::vector< T > t_row, const size_t& index ) -> void
        {
            if ( !ncol( ) )
            {
                data = t_row;
                dim_n = 1;
                dim_m = t_row.size( );
            }
            else if ( t_row.size( ) != dim_m )
            {
                throw matrixDimError{"CANNOT APPEND ROW OF SIZE NOT EQUAL TO COL DIMENSION!"};
            }
            else
            {
                vect::vect< T > new_data( data.size( ) + t_row.size( ) );
                int j = 0;
                int i = 0;
                int l = 0;
                for ( int j = 0; j < nrow( ) + 1; ++j )
                {
                    if ( j == index )
                    {
                        for ( int k = 0; k < ncol( ); ++k )
                        {
                            new_data[ i++ ] = t_row[ k ];
                        }
                    }
                    if ( j != index )
                    {
                        for ( int k = 0; k < ncol( ); ++k )
                        {
                            new_data[ i++ ] = data[ l++ ];
                        }
                    }
                }
                data = new_data;
                dim_n++;
            }
        }

        auto insertrow( SKAS::vect::vect< T > t_row, const size_t& index ) -> void
        {
            if ( !ncol( ) )
            {
                data = t_row;
                dim_n = 1;
                dim_m = t_row.size( );
            }
            else if ( t_row.size( ) != dim_m )
            {
                throw matrixDimError{"CANNOT APPEND ROW OF SIZE NOT EQUAL TO COL DIMENSION!"};
            }
            else
            {
                vect::vect< T > new_data( data.size( ) + t_row.size( ) );
                int j = 0;
                int i = 0;
                int l = 0;
                for ( int j = 0; j < nrow( ) + 1; ++j )
                {
                    if ( j == index )
                    {
                        for ( int k = 0; k < ncol( ); ++k )
                        {
                            new_data[ i++ ] = t_row[ k ];
                        }
                    }
                    if ( j != index )
                    {
                        for ( int k = 0; k < ncol( ); ++k )
                        {
                            new_data[ i++ ] = data[ l++ ];
                        }
                    }
                }
                data = new_data;
                dim_n++;
            }
        }

        /**
         * @brief append row to bottom of matrix
         * @param t_row std::vector< T > row to append
         * @exception matrixDimError thrown when size of t_row is not compatible with dimensions of matrix 
         */
        auto appendrow( const std::vector< T >& t_row ) -> void 
        {
            insertrow( t_row, nrow( ) );
        }

        auto appendrow( const SKAS::vect::vect< T >& t_row ) -> void 
        {
            insertrow( t_row, nrow( ) );
        }

        /**
         * @brief insert col into matrix at index
         * @param t_row std::vector< T > col to insert
         * @param index col index to insert into
         * @exception matrixDimError thrown when either index is outside of matrix or if col size is not compatible with matrix
         */

        auto insertcol( std::vector< T > t_col, const size_t& index_t, int quantity = 1 ) -> void
        {
            if ( !dim_n )
            {
                data = t_col;
                dim_m = 1;
                dim_n = t_col.size( );
            }
            else if ( t_col.size( ) != dim_n )
            {
                throw matrixDimError{"CANNOT APPEND COLUMN OF SIZE NOT EQUAL TO ROW DIMENSION"};
            }
            else
            {
                int index = index_t;
                std::vector< T > new_data( data.size( ) + t_col.size( ) );
                int j = 0;
                int k = 0;
                for ( int i = 0; i < new_data.size( ); i++ )
                {
                    if ( ( std::abs( index - i ) % ( ncol( ) + 1 ) ) == 0 )
                    {
                        new_data[ i ] = t_col[ k++ ];
                    }
                    else
                    {
                        new_data[ i ] = data[ j++ ];
                    }
                }
                dim_m++;
                data = new_data;
            }    
        }

        auto insertcol( SKAS::vect::vect< T > t_col, const size_t& index_t, int quantity = 1 ) -> void
        {
            if ( !dim_n )
            {
                data = t_col;
                dim_m = 1;
                dim_n = t_col.size( );
            }
            else if ( t_col.size( ) != dim_n )
            {
                throw matrixDimError{"CANNOT APPEND COLUMN OF SIZE NOT EQUAL TO ROW DIMENSION"};
            }
            else
            {
                std::vector< T > new_data( data.size( ) + t_col.size( ) );
                int j = 0;
                int k = 0;
                int index = index_t;
                for ( int i = 0; i < new_data.size( ); i++ )
                {
                    if ( ( std::abs( index - i ) % ( ncol( ) + 1 ) ) == 0 )
                    {
                        new_data[ i ] = t_col[ k++ ];
                    }
                    else
                    {
                        new_data[ i ] = data[ j++ ];
                    }
                }
                dim_m++;
                data = new_data;
            }    
        }

        /**
         * @brief append col to right of matrix
         * @param t_col std::vector< T > col to append
         * @exception matrixDimError thrown when size of t_col is not compatible with dimensions of matrix 
         */
        
        auto appendcol( const std::vector< T >& t_col ) -> void
        {
            insertcol( t_col, ncol( ) );
        }

        auto appendcol( const SKAS::vect::vect< T >& t_col ) -> void
        {
            insertcol( t_col, ncol( ) );
        }

        /**
         * @brief transpose matrix
         * @return matrix post transposed
         */

        auto t( ) const -> matrix
        {
            if ( is_empty( ) ) return *this;
            vect::vect< T > outdata( data.size( ) );
            for ( int i = 0; i < ncol( ); ++i )
            {
                for ( int j = 0; j < nrow( ); ++j )
                {
                    outdata[ j + i*nrow( ) ] = data[ j * ncol( ) + i ];
                }
            }
            matrix out( outdata, ncol( ), nrow( ), is_parallel( ) );
            return out;
        }
        
        /**
         * @brief drops row at selected row index
         * @param index index to row to drop
         * @exception matrixDimError thrown when row to drop is outside of matrix
         */

        auto droprow( const size_t& index ) -> void
        {
            if ( index < 0 || index >= nrow( ) )
            {
                throw matrixDimError{"CANNOT DROP ROW OUTSIDE MATRIX DIMENSIONS"};
            }
            data.erase(data.begin( ) + ( index * ncol( ) ), data.begin( ) + ( ( index + 1 ) * ( ncol( ) ) ) );
            dim_n--;
        }

        /**
         * @brief drops col at selected col index
         * @param index index to drop col
         * @exception matrixDimError thrown when col to drop is outside of matrix
         */
        auto dropcol( const size_t& index ) -> void
        {
            if ( index < 0 || index >= ncol( ) )
            {
                throw matrixDimError{"CANNOT DROP ROW OUTSIDE MATRIX DIMENSIONS"};
            }
            for ( int i = nrow( ) - 1; i >= 0; i-- )
            {
            data.erase( data.begin( ) + index + i * ncol( ), data.begin( ) + index + i * ncol( ) + 1 );
            }
            dim_m--;
        }

        auto is_parallel( ) const -> bool
        {
            return parallel;
        }

        auto getinterior( ) const -> vect::vect< T >
        {
            return data;
        }

        auto it_at( const size_t rowIndex, const size_t colIndex ) -> std::vector< T >::iterator
        {
            if ( rowIndex >= nrow( ) || colIndex >= ncol( ) ) throw matrixDimError{"CANNOT RETRIEVE ITERATOR TO ELEMENT OUTSIDE OF MATRIX DIM"};
            return data.begin( ) + rowIndex*ncol( ) + colIndex;
        }

        auto it_at( const size_t rowIndex, const size_t colIndex ) const -> std::vector< T >::const_iterator
        {
            if ( rowIndex >= nrow( ) || colIndex >= ncol( ) ) throw matrixDimError{"CANNOT RETRIEVE ITERATOR TO ELEMENT OUTSIDE OF MATRIX DIM"};
            return data.begin( ) + rowIndex*ncol( ) + colIndex;
        }

        auto operator==( const matrix& c_matrix ) const -> bool
        {
            if ( c_matrix.ncol( ) != ncol( ) || c_matrix.nrow( ) != nrow( ) ) return false;
            if ( c_matrix.getinterior( ) != getinterior( ) ) return false;
            else return true;
        }

        auto is_empty( ) const -> bool
        {
            return ( !nrow( ) && !ncol( ) );
        }

    }; // =========================END OF MEMBER FUNCTIONS FOR MATRIX CLASS=========================

}; // namespace SKAS::matrix -end

namespace SKAS::matrix::accel_matr
{
    template < SKAS::FlAd T1 >
    auto PM_scale( const SKAS::matrix::matrix< T1 >& t_matrix, T1 scalar ) -> SKAS::matrix::matrix< T1 >;

    template < SKAS::FlAd T1 >
    auto PM_add( const SKAS::matrix::matrix< T1 >& a_matrix, const SKAS::matrix::matrix< T1 >& b_matrix ) -> SKAS::matrix::matrix< T1 >;

    template < SKAS::FlAd T1 >
    auto PM_sub( const SKAS::matrix::matrix< T1 >& a_matrix, const SKAS::matrix::matrix< T1 >& b_matrix ) -> SKAS::matrix::matrix< T1 >;

    template < SKAS::FlAd T1 >
    auto PM_mul( const SKAS::matrix::matrix< T1 >& a_matrix, const SKAS::matrix::matrix< T1 >& b_matrix ) -> SKAS::matrix::matrix< T1 >;


}; // namespace SKAS::matrix::accel_matr -end

namespace SKAS::matrix
{
    /**
     * @brief matrix stream insert
     * @param os stream
     * @param t_matrix matrix to insert
     */
    template < SKAS::FlAd T > 
    auto operator<<( std::ostream& os, const matrix< T >& t_matrix ) -> std::ostream&
    {
        auto it = t_matrix.it_at( 0, 0 );
        for ( int j = 0; j < t_matrix.nrow( ); ++j )
        {
            std::cout << "[ ";
            for ( int i = 0; i < t_matrix.ncol( ); ++i ) 
            {
                std::cout << *it << " ";
                ++it;
            }
            std::cout << "]\n";
        }
        return os;
    }

    /**
     * @brief matrix scale support
     * @param t_matrix matrix to scale
     * @param scalar long double
     * @return scaled matrix
     */
    template < SKAS::FlAd T > 
    auto operator*( const matrix< T >& t_matrix, T scalar ) -> matrix< T >
    {
        if ( t_matrix.is_parallel( ) ) return accel_matr::PM_scale( t_matrix, scalar );
        matrix< T > output = t_matrix;
        for ( int i = 0; i < t_matrix.nrow( ); ++i )
        {
            for ( int j = 0; j < t_matrix.ncol( ); ++j )
            {
                output.setelem( output.getelem( i, j ) * scalar, i, j );
            }
        }
        return output;
    }

    /**
     * @brief matrix scale support
     * @param t_matrix matrix to scale
     * @param scalar long double
     * @return scaled matrix
     */
    template < SKAS::FlAd T > 
    auto operator*( T scalar, const matrix< T >& t_matrix ) -> matrix< T >
    {
        if ( t_matrix.is_parallel( ) ) return accel_matr::PM_scale( t_matrix, scalar );
        matrix< T > output = t_matrix;
        for ( int i = 0; i < t_matrix.nrow( ); ++i )
        {
            for ( int j = 0; j < t_matrix.ncol( ); ++j )
            {
                output.setelem( output.getelem( i, j ) * scalar, i, j );
            }
        }
        return output;
    }

    /**
     * @brief matrix addition support
     * @param a_matrix left matrix to add
     * @param b_matrix right matrix to add
     * @return matrix result
     * @exception dimSizeError thrown when mismatched dimensions
     */
    template < SKAS::FlAd T > 
    auto operator+( const matrix< T >& a_matrix, const matrix< T >& b_matrix ) -> matrix< T >
    {
        if ( a_matrix.ncol( ) != b_matrix.ncol( ) || a_matrix.nrow( ) != b_matrix.nrow( ) )
        {
            throw matrixDimError{"CANNOT ADD MATRICIES OF INCOMPATIBLE DIMENSIONS"};
        }
        if ( a_matrix.is_parallel() && b_matrix.is_parallel() ) return accel_matr::PM_add( a_matrix, b_matrix );
        matrix< T > output = a_matrix;
        for ( int i = 0; i < a_matrix.nrow( ); ++i )
        {
            for ( int j = 0; j < a_matrix.ncol( ); ++j )
            {
                output.setelem( a_matrix.getelem(i,j) + b_matrix.getelem(i,j), i, j );
            }
        }
        return output;
    }

    /**
     * @brief matrix subtraction support
     * @param a_matrix left matrix to subtract
     * @param b_matrix right matrix to subtract
     * @return matrix result
     * @exception dimSizeError thrown when mismatched dimensions
     */
    template < SKAS::FlAd T > 
    auto operator-( const matrix< T >& a_matrix, const matrix< T >& b_matrix ) -> matrix< T >
    {
        if ( a_matrix.ncol( ) != b_matrix.ncol( ) || a_matrix.nrow( ) != b_matrix.nrow( ) )
        {
            throw matrixDimError{"CANNOT ADD MATRICIES OF INCOMPATIBLE DIMENSIONS"};
        }
        if ( a_matrix.is_parallel() && b_matrix.is_parallel() ) return accel_matr::PM_sub(a_matrix,b_matrix);
        matrix< T > output = a_matrix;
        for ( int i = 0; i < a_matrix.nrow( ); ++i )
        {
            for ( int j = 0; j < a_matrix.ncol( ); ++j )
            {
                output.setelem( a_matrix.getelem(i,j) - b_matrix.getelem(i,j), i, j );
            }
        }
        return output;
    }

    /**
     * @brief matrix sqrt support
     * @param a_matrix matrix to root all elements
     * @return matrix result
     * @exception realError thrown when sqrt applied on negative number
     */
    template < SKAS::FlAd T > 
    auto sqrt( const matrix< T >& a_matrix ) -> matrix< T >
    {
        matrix< T > out = a_matrix;
        T val;
        for ( int i = 0; i < out.nrow( ); ++i )
        {
            for ( int j = 0; j < out.ncol( ); ++j )
            {
                val = out.getelem( i, j );
                if ( std::abs( val ) < 0.0000001 ) out.setelem( 0, i, j );
                if ( val < 0 ) throw realError{"CANNOT SQRT NEGATIVE IN MATRIX ROOT"};
                else
                {
                    out.setelem( std::sqrt(val), i, j );
                }
            }
        }
        return out;
    }

    /**
     * @brief returns an n by 1 matrix of just the diagonials of t_matrix
     * @param t_matrix matrix
     * @return matrix  
     */
    template < SKAS::FlAd T >
    auto diag( const matrix< T >& t_matrix ) -> matrix< T >
    {
        int mindim = std::min( t_matrix.ncol( ), t_matrix.nrow( ) );
        matrix< T > out( 0, mindim, 1 );
        for ( int i = 0; i < mindim; ++i )
        {
            out.setelem( t_matrix.getelem( i, i ), i, 0 );
        }
        return out;
    }

    /**
     * @brief initilizes identity matrix of given dimension
     * @param dim integer dimesion for a square matrix
     * @return returns identity matrix
     */
    template < SKAS::FlAd T >
    auto identity( const size_t& dim ) -> matrix< T >
    {
        matrix< T > output( 0, dim, dim );
        for ( int diag = 0;  diag < dim; ++diag )
        {
            output.setelem( 1, diag, diag );
        }
        return output;
    }

    /**
     * @brief matrix multiplication
     * @param a_matrix left side matrix to multiply. 
     * @param b_matrix right side matrix to multiply.
     * @return matrix post-multiplication
     * @exception dimSizeError for matricies of incompatible dimensions 
     */
    template < SKAS::FlAd T >
    auto operator%( const matrix< T >& a_matrix, const matrix< T >& b_matrix ) -> matrix< T >
    {
        if ( a_matrix.ncol( ) != b_matrix.nrow( ) ) throw matrixDimError{"CANNOT MULTIPLY MATRICIES OF INCOMPATIBLE DIMENSIONS"};
        if ( a_matrix.is_parallel() && b_matrix.is_parallel() ) return accel_matr::PM_mul(a_matrix,b_matrix);
        matrix< T > product( 0, b_matrix.ncol( ), a_matrix.nrow( ) );
        const auto ait = a_matrix.it_at(0,0);
        const auto bit = b_matrix.it_at(0,0);
        T sum = 0;
        for ( int i = 0; i < product.nrow( ); ++i )
        {
            for ( int j = 0; j < product.ncol( ); ++j )
            {
                sum = 0;
                for ( int k = 0; k < b_matrix.nrow( ); ++k )
                {
                    sum += *( ait + a_matrix.ncol( )*i + k ) * *( bit + j + k*b_matrix.ncol( ) );
                }
                product.setelem( sum, i, j );
            }
        }
        return product;
    }


    // spd inversion
    template < SKAS::FlAd T >
    auto spd( const matrix< T >& a_matrix ) -> matrix< T >
    {
        //checking for user issues or edge cases
        if ( a_matrix.ncol( ) != a_matrix.nrow( ) )
        {
            throw matrixDimError{"CANNOT INVERT NON-SQUARE MATRICIES UNDER SPD PARAMETER"};
        }
        if ( !a_matrix.ncol( ) )
        {
            return a_matrix;
        }
        if ( a_matrix.nrow( ) == 1 && a_matrix.ncol( ) == 1 )
        {
            if ( !a_matrix.getelem( 1, 1 ) )
            {
                throw solutionError{"NON-INVERTIBLE MATRIX CANNOT BE SOLVED"};
            }
            else
            {
                matrix< T > output( 1.0 / a_matrix.getelem( 1, 1 ), 1, 1 );
                return output;
            }
        }
        auto dim = a_matrix.ncol( );
        long double sum = 0;
        matrix< T > L( 0, dim, dim );

        //cholesky decomposition
        for ( int i = 0; i < dim; i++ )
        {
            for ( int j = 0; j <= i; j++ )
            {
                sum = 0;
                if ( i == j )
                {
                    for ( int k = 0; k <= j - 1; ++k )
                    {
                        sum += pow( L.getelem( j, k ), 2 );
                    }
                    L.setelem( ( std::sqrt( a_matrix.getelem( i, j ) - sum ) ), i, j );
                }
                else
                {
                    for ( int k = 0; k <= j - 1; k++ )
                    {
                        sum += L.getelem( i, k ) * L.getelem( j, k );
                    }
                    L.setelem( ( 1.0 / L.getelem( j, j ) * ( a_matrix.getelem( i, j ) - sum ) ), i, j );
                }
            } 
        }
        //forward substituion
        matrix< T > Linv = triangularinvert( L, true );
        return ( Linv.t( ) % Linv );
    }


    /**
     * @brief matrix inversion. as of 11-25-2025 only type symmetric positive definite supported. this function is expensive; save copy if needed in repetition.
     * @param a_matrix matrix to invert
     * @param type std::string. type of matrix. only "spd" supported currently
     * @exception dimSizeError if non-square using "spd" type
     * @exception solutionError if singularity detected
     * @return matrix
     */
    template < SKAS::FlAd T >
    auto invert( const matrix< T >& a_matrix, std::string type = "qr" ) -> matrix< T >
    {
        if ( type == "spd" )
        {
            return spd( a_matrix );
        }
        if ( type == "qr" )
        {
            auto QR = qr_decomp( a_matrix );
            return triangularinvert( QR[ 1 ], false ) % QR[ 0 ].t( );
        }
        else
        {
            throw solutionError{"INCORRECT TYPE PARAMETER"};
        }
    }

    /**
     * @brief solves Ax = b for lower triangular matrix
     * @param lower_matrix triangular matrix to solve. singularity is checked.
     * @param b vector to solve
     * @return the solution x for Ax = b
     * @exception solutionError thrown for singularity
     */
    template < SKAS::FlAd T >
    auto forwardsolve( const matrix< T >& lower_matrix, const vect::vect< T >& b ) -> vect::vect< T >
    {
        vect::vect< T > x;
        T sum = 0;
        for ( int m = 0; m < b.size( ); ++m )
        {
            sum = 0;
            for ( int i = 0; i < m; ++i )
            {
                sum += lower_matrix.getelem( m, i ) * x[ i ];
            } 
            if ( !lower_matrix.getelem( m, m ) )
            {
                throw solutionError{"CANNOT SOLVE LOWER TRIANGULAR MATRIX"};
            }
            x.push_back( ( b[ m ] - sum ) / lower_matrix.getelem( m, m ) );
        }
        return x;
    }

    /**
     * @brief solves Ax = b for upper triangular matrix
     * @param upper_matrix triangular matrix to solve. singularity is checked.
     * @param b vector to solve
     * @return the solution x for Ax = b
     * @exception solutionError thrown for singularity
     */
    template < SKAS::FlAd T >
    auto backsolve( const matrix< T >& upper_matrix, const vect::vect< T >& b ) -> vect::vect< T >
    {
        vect::vect< T > x(b.size());
        T sum = 0;
        for ( int m = b.size( ) - 1; m >= 0; --m )
        {
            sum = 0;
            for ( int i = b.size( ) - 1; i > m; --i )
            {
                sum += upper_matrix.getelem( m, i ) * x[ i ];
            }
            if ( !upper_matrix.getelem( m, m ) ) throw solutionError{"CANNOT SOLVE LOWER TRIANGULAR MATRIX"};
            x[m]=( ( b[ m ] - sum ) / upper_matrix.getelem( m, m ) );
        }
        return x;
    }

    /**
     * @brief inverts triangular matrix
     * @param t_matrix triangular matrix to invert. singularity is checked.
     * @param lower true for lower triangular matrix, false for upper
     * @return the inverse of t_matrix
     * @exception solutionError thrown for singularity
     */

    // ====================================================== unchanged
    template < SKAS::FlAd T >
    auto triangularinvert( const matrix< T >& t_matrix, bool lower ) -> matrix< T >
    {
        matrix< T > output;
        vect::vect< T > e( size_t{t_matrix.nrow( )}, T{0} );
        if ( lower )
        {
            for ( int j = 0; j < t_matrix.ncol( ); ++j )
            {
                e[ j ] = 1;
                output.appendcol( forwardsolve( t_matrix, e ) );
                e[ j ] = 0;
            }
        }
        if ( !lower )
        {
            for ( int j = t_matrix.ncol( ) - 1; j >= 0; --j )
            {
                e[ j ] = 1;
                auto next = backsolve( t_matrix, e );
                output.insertcol( next, 0 );
                e[ j ] = 0;
            }
        }
        return output;
    }

    /**
     * @brief QR decomposition of matrix
     * @param t_matrix matrix to decompose
     * @exception solutionError thrown for singularity
     * @return vector of two matricies: Q, R; where t_matrix = QR
     */
    template < SKAS::FlAd T >
    auto qr_decomp( const matrix< T >& t_matrix ) -> std::vector< matrix< T > >
    {
        std::vector< matrix< T > > container;
        qr_dive( t_matrix, 0, container );
        std::vector< matrix< T > > QR( 2 );
        matrix R = container[ container.size( ) - 1 ];
        for ( int prod_i = container.size( ) - 2; prod_i >= 0; --prod_i )
        {
            R = R % container[ prod_i ];
        }
        matrix Qt = R;
        R = R % t_matrix;
        QR[ 0 ] = Qt.t( );
        QR[ 1 ] = R;
        return QR;
    }

    /**
     * @brief Recursive utility for qr_decomp
     */
    template < SKAS::FlAd T >
    auto qr_dive( const matrix< T >& t_matrix, unsigned int pivot, std::vector< matrix< T > >& container ) -> void
    {
        vect::vect< T > ae( t_matrix.nrow( ), T{0}, false );
        int sign = !std::signbit( t_matrix.getelem( 0, 0 ) );
        ae[ 0 ] = ( -2*sign + 1 ) * mag( t_matrix.getcol( 0 ) );
        matrix< T > v;
        v.appendcol( makeunit( t_matrix.getcol( 0 ) - ae ) );
        matrix< T > Q = identity< T >( v.nrow( ) ) - ( T{2.0} * ( v % v.t( ) ) );    
        matrix< T > Q_p( 0, pivot + Q.nrow( ), pivot + Q.ncol( ) );
        for ( int i_index = 0; i_index < pivot; ++i_index )
        {
            Q_p.setelem( 1, i_index, i_index );
        }
        for ( int q_nindex = 0; q_nindex < Q.nrow( ); ++q_nindex )
        {
            for ( int q_mindex = 0; q_mindex < Q.ncol( ); ++q_mindex )
            {
                Q_p.setelem( Q.getelem( q_nindex, q_mindex ), q_nindex + pivot, q_mindex + pivot );
            }
        }
        Q = Q % t_matrix;
        Q.dropcol( 0 );
        if ( !Q.ncol( ) )
        {
            container.push_back( Q_p );
        }
        else
        {
            Q.droprow( 0 );
            container.push_back( Q_p );
            qr_dive( Q, ++pivot, container );
        }
    }

}; //namespace SKAS::matrix

namespace SKAS::matrix::accel_matr
{
    /*
    template < SKAS::FlAd T1 >
    auto PM_scale( const SKAS::matrix::matrix< T1 >& t_matrix, T1 scalar ) -> SKAS::matrix::matrix< T1 >;

    template < SKAS::FlAd T1 >
    auto PM_add( const SKAS::matrix::matrix< T1 >& a_matrix, const SKAS::matrix::matrix< T1 >& b_matrix ) -> SKAS::matrix::matrix< T1 >;

    template < SKAS::FlAd T1 >
    auto PM_sub( const SKAS::matrix::matrix< T1 >& a_matrix, const SKAS::matrix::matrix< T1 >& b_matrix ) -> SKAS::matrix::matrix< T1 >;

    template < SKAS::FlAd T1 >
    auto PM_mul( const SKAS::matrix::matrix< T1 >& a_matrix, const SKAS::matrix::matrix< T1 >& b_matrix ) -> SKAS::matrix::matrix< T1 >;
    */

    template < SKAS::FlAd T1 >
    auto PM_scale( const SKAS::matrix::matrix< T1 >& t_matrix, T1 scalar ) -> SKAS::matrix::matrix< T1 >
    {
        matrix< T1 > out(SKAS::vect::accel_vect::PV_scale( t_matrix.getinterior(), scalar ), t_matrix.nrow( ), t_matrix.ncol( ), true ); 
        return out;
    }

    template < SKAS::FlAd T1 >
    auto PM_add( const SKAS::matrix::matrix< T1 >& a_matrix, const SKAS::matrix::matrix< T1 >& b_matrix ) -> SKAS::matrix::matrix< T1 >
    {
        matrix< T1 > out(
            SKAS::vect::accel_vect::PV_add( a_matrix.getinterior(), b_matrix.getinterior() ),
            a_matrix.nrow(),
            b_matrix.ncol(),
            true
        );
        return out;
    }

    template < SKAS::FlAd T1 >
    auto PM_sub( const SKAS::matrix::matrix< T1 >& a_matrix, const SKAS::matrix::matrix< T1 >& b_matrix ) -> SKAS::matrix::matrix< T1 >
    {
        matrix< T1 > out(
            SKAS::vect::accel_vect::PV_sub( a_matrix.getinterior(), b_matrix.getinterior() ),
            a_matrix.nrow(),
            b_matrix.ncol(),
            true
        );
        return out;
    }

    template < SKAS::FlAd T1 >
    auto PM_mul( const SKAS::matrix::matrix< T1 >& a_matrix, const SKAS::matrix::matrix< T1 >& b_matrix ) -> SKAS::matrix::matrix< T1 >
    {
        auto bt = b_matrix.t();

        sycl::queue& q = gpu::ctx( ).q;

        int adimn = a_matrix.nrow();
        int adimm = a_matrix.ncol();
        int bdimn = b_matrix.nrow();
        int bdimm = b_matrix.ncol();

        T1* dev_a = sycl::malloc_device< T1 >( adimm*adimn, q );
        T1* dev_b = sycl::malloc_device< T1 >( bdimm*bdimn, q );
        T1* dev_c = sycl::malloc_device< T1 >( adimn*bdimm, q );


        q.memcpy( dev_a, a_matrix.getinterior().data( ), sizeof( T1 ) * adimm*adimn );
        q.memcpy( dev_b, bt.getinterior().data( ), sizeof( T1 ) * bdimm*bdimn );

        q.parallel_for(
            sycl::range<2>(adimn, bdimm),
            [=](sycl::id<2> idx){

            int i = idx[0];
            int j = idx[1];

            T1 sum = 0;

            for(int k=0; k<adimm; ++k)
                sum += *( i*adimm + dev_a + k ) * *( j*bdimn + dev_b + k );

            *(i*bdimm + dev_c + j) = sum;
            
        });

        std::vector< T1 > out( adimn*bdimm );

        q.memcpy( out.data(), dev_c, sizeof( T1 ) * adimn*bdimm );
        q.wait( );

        sycl::free( dev_a, q );
        sycl::free( dev_b, q );
        sycl::free( dev_c, q );

        SKAS::matrix::matrix< T1 > final( out, adimm, bdimn, true );
        return final;
    }
};

#endif