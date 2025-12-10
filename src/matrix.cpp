/**
 * @brief Implementation of matrix functionality
 * @author Will Sharpsteen - wisharpsteen@gmail.com
 */
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include "matrix.h"
#include "customexceptions.h"


// =========================START OF MEMBER FUNCTIONS FOR MATRIX CLASS=========================
matrix::matrix( ) : dim_n{ 0 }, dim_m{ 0 }
{
}

matrix::matrix( const matrix &original ) : dim_n{ 0 }, dim_m{ 0 }
{
    clear( );
    for ( int row = 0; row < original.dim_n; row++ )
    {
        appendrow( original.getrow( row ) );
    }
}

auto matrix::clear( ) -> void
{
    data.clear( );
    dim_n = 0;
    dim_m = 0;
}

auto matrix::nrow( ) const -> int
{
    return dim_n;
}

auto matrix::ncol( ) const -> int
{
    return dim_m;
}

auto matrix::getrow( int index ) const -> std::vector< long double >
{
    if ( index < 0 || index >= nrow( ) )
    {
        throw dimSizeError{"CANNOT RETREIVE ROW OUTSIDE MATRIX"};
    }
    return data[ index ];
}

auto matrix::getcol( int index ) const -> std::vector< long double >
{
    if ( index < 0 || index >= ncol( ) )
    {
        throw dimSizeError{"CANNOT RETREIVE COL OUTSIDE MATRIX"};
    }
    std::vector< long double > final( nrow( ) );
    for ( int i = 0; i < nrow( ); ++i )
    {
        final[ i ] = data[ i ][ index ];
    }
    return final;
}

auto matrix::getelem( int row, int col, bool index ) const -> long double
{
    if ( index )
    {
        if ( row < 0 || row >= nrow( ) || col < 0 || col >= ncol( ) )
        {
            throw dimSizeError{"CANNOT getelem OUTSIDE OF MATRIX DIMENSIONS"};
        }
        return data[ row ][ col ];
    }
    if ( row < 1 || row > nrow( ) || col < 1 || col > ncol( ) )
    {
        throw dimSizeError{"CANNOT getelem OUTSIDE OF MATRIX DIMENSIONS"};
    }
    return data[ row - 1 ][ col - 1 ];
}

auto matrix::setelem( long double value, int row, int col, bool index ) -> void
{
    if ( index )
    {
        if ( row < 0 || row >= nrow( ) || col < 0 || col >= ncol( ) )
        {
            throw dimSizeError{"CANNOT setelem OUTSIDE OF MATRIX DIMENSIONS"};
        }
        data[ row ][ col ] = value;
    }
    if ( !index )
    {
        if ( row < 1 || row > nrow( ) || col < 1 || col > ncol( ) )
        {
            throw dimSizeError{"CANNOT setelem OUTSIDE OF MATRIX DIMENSIONS"};
        }
        data[ row - 1 ][ col - 1 ] = value;
    }
}

auto matrix::appendrow( std::vector< long double > t_row ) -> void
{
    if ( !dim_m )
    {
        dim_m = t_row.size( );
    }
    if ( t_row.size( ) != dim_m )
    {
        throw dimSizeError{"CANNOT APPEND ROW OF SIZE NOT EQUAL TO COL DIMENSION!"};
    }
    data.push_back( t_row );
    dim_n++;
}

auto matrix::appendcol( std::vector< long double > t_col ) -> void
{
    if ( !dim_n )
    {
        dim_n = t_col.size( );
        std::vector< long double > empty;
        for ( int container = 0; container < dim_n; ++container )
        {
            data.push_back( empty );
        }
        dim_m++;
        for ( int row = 0; row < t_col.size( ); row++ )
        {
            data[ row ].push_back( t_col[ row ] );
        }
    }
    else if ( t_col.size( ) != dim_n )
    {
        throw dimSizeError{"CANNOT APPEND COLUMN OF SIZE NOT EQUAL TO ROW DIMENSION"};
    }
    else
    {
        for ( int row = 0; row < nrow( ); row++ )
        {
            data[ row ].push_back( t_col[ row ] );
        }
        dim_m++;
    }
}

auto matrix::t( ) const -> matrix
{
    matrix final;
    for ( int col = 0; col < ncol( ); ++col )
    {
        final.appendrow( getcol( col ) );
    }
    return final;
}

auto matrix::operator==( const matrix& c_matrix ) const -> bool
{
    if ( c_matrix.nrow( ) != nrow( ) )
    {
        return false;
    }
    for ( int row = 0; row < c_matrix.dim_n; ++row )
    {
        if ( !vcomp( getrow( row ), c_matrix.getrow( row ) ) )
        {
            return false;
        }
    }
    return true;
}

auto matrix::shove_below( const matrix& o_matrix ) -> void
{
    if ( ncol( ) != o_matrix.ncol( ) )
    {
        throw dimSizeError{"CANNOT SHOVE_BELOW MATRIX OF INCOMPATIBLE DIMENSIONS"};
    }
    for ( int n_row = 0; n_row < o_matrix.nrow( ); ++n_row )
    {
        appendrow( o_matrix.getrow( n_row ) );
    }
}

auto matrix::insertrow( std::vector< long double > t_row, int index, int quantity ) -> void
{
    if ( !ncol( ) )
    {
        dim_m = t_row.size( );
    }
    if ( t_row.size( ) != ncol( ) || index > nrow( ) )
    {
        throw dimSizeError{"CANNOT INSERT ROW OF INCOMPATIBLE DIMENSIONS"};
    }
    if ( index == nrow( ) )
    {
        appendrow( t_row );
    }
    else
    {
        data.insert( data.begin( ) + index, quantity, t_row );
        dim_n++;
    }
}

auto matrix::insertcol( std::vector< long double > t_col, int index, int quantity ) -> void
{
    if ( !index && !ncol( ) )
    {
        for ( int i = 0; i < quantity; ++i )
        {
            appendcol( t_col );
        }
        return;
    }
    if ( t_col.size( ) != nrow( ) || index > ncol( ) )
    {
        throw dimSizeError{"CANNOT INSERT COL OF INCOMPATIBLE DIMENSIONS"};
    }
    if ( index == ncol( ) )
    {
        for ( int i = 0; i < quantity; ++i )
        {
            appendcol( t_col );
        }
        return;
    }
    else
    {
        int elem_t = 0;
        for ( auto &row : data )
        {
            row.insert( row.begin( ) + index, quantity, t_col[ elem_t++ ] );
        }
        dim_m += quantity;
        return;
    }
}

auto matrix::droprow( int index ) -> void
{
    if ( index < 0 || index >= ncol( ) )
    {
        throw dimSizeError{"CANNOT DROP ROW OUTSIDE MATRIX DIMENSIONS"};
    }
    data.erase( data.begin( ) + index );
    dim_n--;
}

auto matrix::dropcol( int index ) -> void
{
    if ( index < 0 || index >= nrow( ) )
    {
        throw dimSizeError{"CANNOT DROP COL OUTSIDE MATRIX DIMENSIONS"};
    }
    for ( auto &row : data )
    {
        row.erase( row.begin( ) + index );
    }
    dim_m--;
}

matrix::~matrix( ) { }

// =========================END OF MEMBER FUNCTIONS FOR MATRIX CLASS=========================

auto operator<<( std::ostream& os, const matrix& t_matrix ) -> std::ostream&
{
    for ( int row = 0; row < t_matrix.nrow( ); ++row )
    {
        os << t_matrix.getrow( row ) << '\n';
    }
    return os;
}

auto init( long double initial_value, int rowcount, int colcount ) -> matrix
{
    matrix output;
    if ( !colcount && !rowcount ) 
    {
        return output;
    }
    if ( !colcount && rowcount )
    {
        std::vector< long double > empty;
        for ( int i = 0; i < rowcount; ++i )
        {
            output.appendrow( empty );
        }
        return output;
    }
    if ( colcount && !rowcount )
    {
        std::vector< long double > empty;
        for ( int i = 0; i < colcount; ++i )
        {
            output.appendcol( empty );
        }
        return output;
    }
    std::vector< long double > temp( colcount, initial_value );
    for ( int i = 0; i < rowcount; ++i )
    {
        output.appendrow( temp );
    }
    return output;
}

auto identity( int dim ) -> matrix
{
    auto output = init( 0, dim, dim );
    for ( int diag = 0; diag < dim; ++diag )
    {
        output.setelem( 1, diag, diag, true );
    }
    return output;
}

auto operator+( const matrix& a_matrix, const matrix& b_matrix ) -> matrix
{
    if ( a_matrix.ncol( ) != b_matrix.ncol( ) || a_matrix.nrow( ) != b_matrix.nrow( ) )
    {
        throw dimSizeError{"CANNOT ADD MATRICIES OF INCOMPATIBLE DIMENSIONS"};
    }
    matrix output;
    for ( int row_i = 0; row_i < a_matrix.nrow( ); ++row_i )
    {
        output.appendrow( a_matrix.getrow( row_i ) + b_matrix.getrow( row_i ) );
    }
    return output;
}

auto operator-( const matrix& a_matrix, const matrix& b_matrix ) -> matrix
{
    if ( a_matrix.ncol( ) != b_matrix.ncol( ) || a_matrix.nrow( ) != b_matrix.nrow( ) )
    {
        throw dimSizeError{"CANNOT SUBTRACT MATRICIES OF INCOMPATIBLE DIMENSIONS"};
    }
    matrix output;
    for ( int row_i = 0; row_i < a_matrix.nrow( ); ++row_i )
    {
        output.appendrow( a_matrix.getrow( row_i ) - b_matrix.getrow( row_i ) );
    }
    return output;
}

auto operator*( const matrix& t_matrix, long double scalar ) -> matrix
{
    matrix output;
    for ( int t_row = 0; t_row < t_matrix.nrow( ); ++t_row )
    {
        output.appendrow( scalar * t_matrix.getrow( t_row ) );
    }
    return output;
}

auto operator*( long double scalar, const matrix& t_matrix ) -> matrix
{
    matrix output;
    for ( int t_row = 0; t_row < t_matrix.nrow( ); ++t_row )
    {
        output.appendrow( scalar * t_matrix.getrow( t_row ) );
    }
    return output;
}

auto operator%( const matrix& a_matrix, const matrix& b_matrix ) -> matrix
{
    if ( a_matrix.ncol( ) != b_matrix.nrow( ) )
    {
        throw dimSizeError{"CANNOT MULTIPLY MATRICIES OF INCOMPATIBLE DIMENSIONS"};
    }
    matrix product;
    std::vector< long double > temp;
    for ( int i = 0; i < a_matrix.nrow( ); ++i )
    {
        for ( int j = 0; j < b_matrix.ncol( ); ++j )
        {
            temp.push_back( a_matrix.getrow( i ) * b_matrix.getcol( j ) );
        }
        product.appendrow( temp );
        temp.clear( );
    }
    return product;
}

auto invert( const matrix& a_matrix, std::string type ) -> matrix
{
    if ( type == "spd" )
    {
        //checking for user issues or edge cases
        if ( a_matrix.ncol( ) != a_matrix.nrow( ) )
        {
            throw dimSizeError{"CANNOT INVERT NON-SQUARE MATRICIES UNDER SPD PARAMETER"};
        }
        if ( !a_matrix.ncol( ) )
        {
            return a_matrix;
        }
        if ( a_matrix.ncol( ) > a_matrix.nrow( ) )
        {
            throw solutionError{"NON-INVERTIBLE MATRIX CANNOT BE SOLVED"};
        }
        if ( a_matrix.nrow( ) == 1 && a_matrix.ncol( ) == 1 )
        {
            if ( !a_matrix.getelem( 1, 1 ) )
            {
                throw solutionError{"NON-INVERTIBLE MATRIX CANNOT BE SOLVED"};
            }
            else
            {
                matrix output = { { 1.0 / a_matrix.getelem( 1, 1 ) } };
                return output;
            }
        }

        //setup
        auto dim = a_matrix.ncol( );
        long double sum = 0;
        matrix L = init( 0, dim, dim );

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
                        sum += pow( L.getelem( j, k, true ), 2 );
                    }
                    L.setelem( ( std::sqrt( a_matrix.getelem( i, j, true ) - sum ) ), i, j, true );
                }
                else
                {
                    for ( int k = 0; k <= j - 1; k++ )
                    {
                        sum += L.getelem( i, k, true ) * L.getelem( j, k, true );
                    }
                    L.setelem( ( 1.0 / L.getelem( j, j, true ) * ( a_matrix.getelem( i, j, true ) - sum ) ), i, j, true );
                }
            } 
        }

        //forward substituion
        matrix Linv = triangularinvert( L, true );
        return ( Linv.t( ) % Linv );
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

auto operator+( const std::vector< long double >& first, const std::vector< long double >& last ) -> std::vector< long double >
{
    if ( first.size( ) != last.size( ) )
    {
        throw dimSizeError{"CANNOT + VECTOR OF DIFFERENT SIZES"};
    }
    std::vector< long double > output;
    for ( int i = 0; i < first.size( ); ++i )
    {
        output.push_back( first[ i ] + last[ i ] );
    }
    return output;
}

auto operator-( const std::vector< long double >& first, const std::vector< long double >& last ) -> std::vector< long double >
{
    if ( first.size( ) != last.size( ) )
    {
        throw dimSizeError{"CANNOT - VECTOR OF DIFFERENT SIZES"};
    }
    std::vector< long double > output;
    for ( int i = 0; i < first.size( ); ++i )
    {
        output.push_back( first[ i ] - last[ i ] );
    }
    return output;
}

auto cov( const std::vector< long double >& a_vec, const std::vector< long double >& b_vec ) -> long double
{
    if ( a_vec.size( ) != b_vec.size( ) ) throw dimSizeError{"CANNOT COMPUTE COV OF INCOMPATIBLE VECTORS"};
    long double out = 0;
    auto aavg = mean(a_vec);
    auto bavg = mean(b_vec);
    for ( int i = 0; i < a_vec.size( ); ++i )
    {
        out += ( a_vec[ i ] - aavg ) * ( b_vec[ i ] - bavg );
    }
    return out / ( a_vec.size( ) - 1.0 );
}

auto corr( const std::vector< long double >& a_vec, const std::vector< long double >& b_vec ) -> long double
{
    return cov( a_vec, b_vec ) / ( s(a_vec) * s(b_vec) );
}

auto operator*( const std::vector< long double >& t_vec, const long double scalar ) -> std::vector< long double >
{
    std::vector< long double > output;
    for ( auto &elem : t_vec )
    {
        output.push_back( elem * scalar );
    }
    return output;
}

auto operator*( const long double scalar, const std::vector< long double >& t_vec ) -> std::vector< long double >
{
    std::vector< long double > output;
    for ( auto &elem : t_vec )
    {
        output.push_back( elem * scalar );
    }
    return output;
}

auto mag( const std::vector< long double >& t_vec ) -> long double
{
    long double sum = 0;
    for ( auto &elem : t_vec )
    {
        sum += pow( elem, 2 );
    }
    return sqrt( sum );
}

auto makeunit( const std::vector< long double >& t_vec ) -> std::vector< long double >
{
    return t_vec * ( 1 / mag( t_vec ) );
}

auto operator*( const std::vector< long double >& first, const std::vector< long double >& last ) -> long double
{
    if( first.size( ) != last.size( ) )
    {
        throw vectorError{"CANNOT DOT PRODUCT VECTORS OF DIFFERENT DIMENSION!"};
    }
    long double final = 0;
    for ( int i = 0; i < first.size( ); ++i )
    {
        final += first[ i ] * last[ i ];
    }
    return final;
}

auto vcomp( const std::vector< long double >& first, const std::vector< long double >& last, float error ) -> bool
{
    if ( first.size( ) != last.size( ) )
    {
        return false;
    }
    for ( int vecti = 0; vecti < first.size( ); ++vecti )
    {
        if ( std::abs( first[ vecti ] - last[ vecti ] ) > error )
        {
            return false;
        }
    }
    return true;
}

auto s2( const std::vector< long double >& t_vector ) -> long double
{
    if ( !t_vector.size( ) || t_vector.size( ) == 1 )
    {
        return 0;
    }
    long double m0 = t_vector[ 0 ];
    long double m1;
    long double s = 0;
    for ( int i = 1; i < t_vector.size( ); ++i )
    {
        m1 = m0 + ( t_vector[ i ] - m0 ) / ( i + 1 );
        s = s + ( t_vector[ i ] - m0 ) * ( t_vector[ i ] - m1 );
        m0 = m1;
    }
    return s / ( t_vector.size( ) - 1 );
}

auto s( const std::vector< long double >& t_vector ) -> long double
{
    return sqrt( s2( t_vector ) );
}

auto diag( const matrix& t_matrix ) -> matrix
{
    auto dim = std::min( t_matrix.nrow( ), t_matrix.ncol( ) );
    matrix out = init( 0, dim, 1 );
    for ( int i = 0; i < dim; ++i )
    {
        out.setelem( t_matrix.getelem( i, i, true ), i, 0, true );
    }
    return out;
}

auto sqrt( const matrix& a_matrix ) -> matrix
{
    matrix out = a_matrix;
    long double val;
    for ( int i = 0; i < out.nrow( ); ++i )
    {
        for ( int j = 0; j < out.ncol( ); ++j )
        {
            val = out.getelem( i, j, true );
            if ( std::abs( val ) < -0.00000001 ) val = 0;
            if ( val < 0 ) throw realError{"CANNOT SQRT NEGATIVE IN MATRIX ROOT"};
            else
            {
                out.setelem( sqrt(val), i, j, true );
            }
        }
    }
    return out;
}

auto operator<<( std::ostream& os, const std::vector< long double >& vec ) -> std::ostream&
{
    os << "[";
    for ( size_t i = 0; i < vec.size( ); ++i ) 
    {
        os << vec[ i ];
        if ( i < vec.size( ) - 1 ) 
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

auto operator<<( std::ostream& os, const std::vector< unsigned int >& vec ) -> std::ostream&
{
    os << "[";
    for ( size_t i = 0; i < vec.size( ); ++i ) 
    {
        os << vec[ i ];
        if ( i < vec.size( ) - 1 ) 
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

auto mean( const std::vector< long double >& t_vec ) -> long double
{
    long double sum = 0;
    for ( auto &elem : t_vec )
    {
        sum += elem;
    }
    return sum / t_vec.size( );
}

auto backsolve( const matrix& t_matrix ) -> matrix
{
    std::vector< long double > solutions;
    long double sum;
    matrix t_inv;
    for ( int xivector = t_matrix.ncol( ) - 1; xivector >= 0 ; xivector-- )
    {
        sum = 0;
        solutions.clear( );
        for ( int leadingzeros = xivector; leadingzeros < t_matrix.ncol( ) - 1; leadingzeros++ )
        {
            solutions.push_back( 0 );
        }
        for ( int depth = xivector; depth >= 0; depth-- )
        {
            if ( depth == xivector )
            {
                if ( !t_matrix.getelem( depth, depth , true ) )
                {
                    throw solutionError{"UPPER TRIANGULAR MATRIX IS SINGULAR AND NON-INVERTIBLE."};
                }
                solutions.push_back( 1.0 / ( t_matrix.getelem( depth, depth, true ) ) );
            }
            else
            {
                for ( int prevsumindex = xivector; prevsumindex > depth; --prevsumindex )
                {
                    sum += t_matrix.getelem( depth, prevsumindex, true ) * solutions[ depth ];
                }
                solutions.push_back( ( -1 * sum ) / ( t_matrix.getelem( depth, depth, true ) ) );
            }
        }
        t_inv.appendrow( solutions );
    }
    return t_inv;
}

auto triangularinvert( const matrix& t_matrix, bool lower ) -> matrix
{
    matrix output;
    std::vector< long double > e( t_matrix.nrow( ), 0 );
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
            std::reverse( next.begin( ), next.end( ) );
            output.insertcol( next, 0 );
            e[ j ] = 0;
        }
    }
    return output;
}

auto forwardsolve( const matrix& lower_matrix, const std::vector< long double >& b ) -> std::vector< long double >
{
    std::vector< long double > x;
    long double sum = 0;
    for ( int m = 0; m < b.size( ); ++m )
    {
        sum = 0;
        for ( int i = 0; i < m; ++i )
        {
            sum += lower_matrix.getelem( m, i, true ) * x[ i ];
        }
        if ( !lower_matrix.getelem( m, m, true ) )
        {
            throw solutionError{"CANNOT SOLVE LOWER TRIANGULAR MATRIX"};
        }
        x.push_back( ( b[ m ] - sum ) / lower_matrix.getelem( m, m, true ) );
    }
    return x;
}

auto backsolve( const matrix& upper_matrix, const std::vector< long double >& b ) -> std::vector< long double >
{
    std::vector< long double > x;
    long double sum = 0;
    for ( int m = b.size( ) - 1; m >= 0; --m )
    {
        sum = 0;
        for ( int i = b.size( ) - 1; i > m; --i )
        {
            sum += upper_matrix.getelem( m, i, true ) * x[ b.size( ) - 1 - i ];
        }
        if ( !upper_matrix.getelem( m, m, true ) )
        {
            throw solutionError{"CANNOT SOLVE UPPER TRIANGULAR MATRIX"};
        }
        x.push_back( ( b[ m ] - sum ) / upper_matrix.getelem( m, m, true ) );
    }
    return x;
}

auto qr_dive( const matrix& t_matrix, unsigned int pivot, std::vector< matrix >& container ) -> void
{
    std::vector< long double > ae( t_matrix.nrow( ), 0 );
    int sign = !std::signbit( t_matrix.getelem( 0, 0, true ) );
    ae[ 0 ] = ( -2*sign + 1 ) * mag( t_matrix.getcol( 0 ) );
    matrix v;
    v.appendcol( makeunit( t_matrix.getcol( 0 ) - ae ) );
    matrix Q = identity( v.nrow( ) ) - ( 2.0 * ( v % v.t( ) ) );    
    matrix Q_p = init( 0, pivot + Q.nrow( ), pivot + Q.ncol( ) );
    for ( int i_index = 0; i_index < pivot; ++i_index )
    {
        Q_p.setelem( 1, i_index, i_index, true );
    }
    for ( int q_nindex = 0; q_nindex < Q.nrow( ); ++q_nindex )
    {
        for ( int q_mindex = 0; q_mindex < Q.ncol( ); ++q_mindex )
        {
            Q_p.setelem( Q.getelem( q_nindex, q_mindex, true ), q_nindex + pivot, q_mindex + pivot, true );
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

auto qr_decomp( const matrix& t_matrix ) -> std::vector< matrix >
{
    std::vector< matrix > container;
    qr_dive( t_matrix, 0, container );
    std::vector< matrix > QR( 2 );
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
