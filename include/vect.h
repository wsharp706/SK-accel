/**
 * @brief Additional Vector Utilites
 * @author Will Sharpsteen wisharpsteen@gmail.com
 */

#include <vector>
#include <iostream>
#include <concepts>
#include <type_traits>
#include <sycl/sycl.hpp>
#include <hipSYCL/algorithms/numeric.hpp>
#include <hipSYCL/algorithms/algorithm.hpp>
#include "customexceptions.h"


#ifndef VECT_H
#define VECT_H

template < typename T >
concept FA = std::is_same_v< T, float > || 
             std::is_same_v< T, double > || 
             std::is_same_v< T, long double >;

/**
 * @brief Wrapper of std::vector class with supplemental utility and parallelization support
 */
template < FA T >
class vect
{
    private:
    bool parallel;

    public:
    std::vector< T > interior;

    vect< T >( ) : parallel( false ) { };

    ~vect< T >( ) { }

    vect< T >( const vect< T >& orig ) : parallel( false )
    {
        interior = orig.interior;
        parallel = orig.parallel;
    }
    
    vect< T >( const std::vector< T >& orig, const bool& par ) : parallel( false )
    {
        interior = orig;
        parallel = par;
    }

    vect< T >& operator=( const vect< T >& other )
    {
        if ( this != &other )
        {
            interior = other.interior;
            parallel = other.parallel;
        }
        return *this;
    }
    vect< T >( const std::initializer_list< T > init, const bool& t_parallel )
    {
        interior = init;
        parallel = t_parallel;
    }

    vect< T >( const bool& t_parallel )
    {
        parallel = t_parallel;
    }

    auto clear( ) -> void
    {
        interior.clear( );
    }

    auto push_back( const T& obj ) -> void
    {
        interior.push_back( obj );
    }

    auto size( ) const -> size_t
    {
        return interior.size( );
    }

    auto insert( std::vector< T >::const_iterator position, const T& val ) -> std::vector< T >::iterator
    {
        interior.insert( position, val );
    }

    auto operator[]( const size_t& index ) -> T&
    {
        return interior[ index ];
    }

    auto operator[]( const size_t& index ) const -> const T&
    {
        return interior[ index ];
    }

    auto data( ) -> T*
    {
        return interior.data( );
    }

    auto data( ) const -> const T*
    {
        return interior.data( );
    }

    /**
     * @brief is vector in parallel algorithm (gpu) mode
     */
    auto is_parallel( ) const -> bool
    {
        return parallel;
    }

    auto toPar( ) -> void
    {
        parallel = true;
    }

    auto toSeq( ) -> void
    {
        parallel = false;
    }

    auto flipParSeqMode( ) -> void
    {
        parallel = !parallel;
    }

    auto toVect( ) -> std::vector< T >&
    {
        return interior;
    }

    auto toVect( ) const -> const std::vector< T >&
    {
        return interior;
    }
};


// -----------------LINEAR ALGEBRA----------------------

/**
 * @brief addition operator support for std::vector
 * @param first vect
 * @param last vect
 * @return vect result from operation
 * @exception vectDimError thrown when first.size( ) != last.size( ) 
 */
template < FA T >
auto operator+( const vect< T >& first, const vect< T >& last ) -> vect< T >
{
    if ( first.is_parallel( ) && last.is_parallel( ) )
    {
        sycl::queue q{sycl::property::queue::in_order{}};
        vect< T > out( PV_add( q, first, last ), true );
        return out;
    }
    if ( first.size( ) != last.size( ) )
    {
        throw vectDimError{"CANNOT + VECTOR OF DIFFERENT SIZES"};
    }
    vect< T > output;
    for ( int i = 0; i < first.size( ); ++i )
    {
        output.push_back( first[ i ] + last[ i ] );
    }
    return output;
}
    
/**
 * @brief subtraction operator support for vect
 * @param first vect
 * @param last vect
 * @return vect result from operation
 * @exception vectDimError thrown when first.size( ) != last.size( ) 
 */
template < FA T >
auto operator-( const vect< T >& first, const vect< T >& last ) -> vect< T >
{
    if ( first.size( ) != last.size( ) )
    {
        throw vectDimError{"CANNOT - VECTOR OF DIFFERENT SIZES"};
    }
    vect< T > output;
    for ( int i = 0; i < first.size( ); ++i )
    {
        output.push_back( first[ i ] - last[ i ] );
    }
    return output;
}

/**
 * @brief scalar support for vect
 * @param t_vec vect to scale
 * @param scalar .numeric
 * @return vect scaled output
 */
template < FA T >
auto operator*( const vect< T >& t_vec, const T scalar ) -> vect< T >
{
    vect< T > output;
    for ( const auto &elem : t_vec.interior )
    {
        output.push_back( elem * scalar );
    }
    return output;
}

/**
 * @brief dot product support for vect
 * @param first vect
 * @param last vect
 * @return .numeric
 * @exception vectDimError thrown for incompatible sizes
 */
template < FA T >
auto operator*( const vect< T >& first, const vect< T >& last ) -> T
{
    if ( first.is_parallel( ) && last.is_parallel( ) )
    {
        sycl::queue q{sycl::property::queue::in_order{}};
        return PV_dot( q, first, last );
    }
    if( first.size( ) != last.size( ) )
    {
        throw vectDimError{"CANNOT DOT PRODUCT VECTORS OF DIFFERENT DIMENSION!"};
    }
    T final = 0;
    for ( int i = 0; i < first.size( ); ++i )
    {
        final += first[ i ] * last[ i ];
    }
    return final;
}

/**
 * @brief scalar support for vect
 * @param t_vec vect to scale
 * @param scalar .numeric
 * @return vect scaled output
 */
template < FA T >
auto operator*( const vect< T >& t_vec, const T& scalar ) -> vect< T >
{
    std::vector< long double > output;
    for ( auto &elem : t_vec )
    {
        output.push_back( elem * scalar );
    }
    return output;
}
/**
 * @brief Euclidean norm operator
 * @param t_vec vect
 * @return .numeric magnitude
 */
template < FA T >
auto mag( const vect< T >& t_vec ) -> T
{
    long double sum = 0;
    for ( auto &elem : t_vec.interior )
    {
        sum += pow( elem, 2 );
    }
    return sqrt( sum );
}

/**
 * @brief Creates new vect of same direction with unit length
 * @param t_vec vect to scale
 * @return unit vector of same length
 */
template < FA T >
auto makeunit( const vect< T >& t_vec ) -> vect< T >
{
    return t_vec * ( 1.0 / mag( t_vec ) );
}

/**
 * @brief comparison tool for high-precision vectors
 * @param first vect
 * @param last vect
 * @param error amount of accepted error between elements. default = 0.00001
 * @return boolean answer of comparison
 */
template < FA T >
auto vcomp( const vect< T >& first, const vect< T >& last, float error = 0.00001 ) -> bool
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

/**
 * @brief comparison tool for high-precision vectors
 * @param first vect
 * @param last vect
 * @param error amount of accepted error between elements. default = 0.00001
 * @return boolean answer of comparison
 */
template < FA T >
auto vcomp( const std::vector< T >& first, const std::vector< T >& last, float error = 0.00001 ) -> bool
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

template < FA T >
auto operator==( const vect< T >& rhs, const vect< T >& lhs ) -> bool
{
    return vcomp( rhs, lhs );
}

template < FA T >
auto operator==( const std::vector< T >& rhs, const std::vector< T >& lhs ) -> bool
{
    return vcomp( rhs, lhs );
}

// -----------------STATS--------------------------

/**
 * @brief compute covariance between two vectors
 * @param a_vec vect 
 * @param b_vec vect
 * @exception vectDimError thrown for incompatible sizings.
 * @return .numeric
 */
template < FA T >
auto cov( const vect< T >& a_vec, const vect< T >& b_vec ) -> T
{
    if ( a_vec.size( ) != b_vec.size( ) ) throw vectDimError{"CANNOT COMPUTE COV OF INCOMPATIBLE VECTORS"};
    T out = 0;
    auto aavg = mean(a_vec);
    auto bavg = mean(b_vec);
    for ( int i = 0; i < a_vec.size( ); ++i )
    {
        out += ( a_vec[ i ] - aavg ) * ( b_vec[ i ] - bavg );
    }
    return out / ( a_vec.size( ) - 1.0 );
}


/**
 * @brief compute correlation between two vectors
 * @param a_vec vect
 * @param b_vec vect
 * @exception vectDimError thrown for incompatible sizings.
 * @return .numeric
 */
template < FA T >
auto corr( const vect< T >& a_vec, const vect< T >& b_vec ) -> T
{
    return cov( a_vec, b_vec ) / ( s(a_vec) * s(b_vec) );
}


/**
 * @brief Welford's Method application of single-pass variance
 * @param t_vector vect
 * @return .numeric
 */
template < FA T >
auto s2( const vect< T >& t_vector ) -> T
{
    if ( !t_vector.size( ) || t_vector.size( ) == 1 )
    {
        return 0;
    }
    T m0 = t_vector[ 0 ];
    T m1;
    T s = 0;
    for ( int i = 1; i < t_vector.size( ); ++i )
    {
        m1 = m0 + ( t_vector[ i ] - m0 ) / ( i + 1 );
        s = s + ( t_vector[ i ] - m0 ) * ( t_vector[ i ] - m1 );
        m0 = m1;
    }
    return s / ( t_vector.size( ) - 1 );
}

/**
 * @brief Welford's Method application of single-pass standard deviation
 * @param t_vector vect
 * @return .numeric
 */
template < FA T >
auto s( const vect< T >& t_vector ) -> T
{
    return sqrt( s2( t_vector ) );
}


/**
 * @brief Compute mean of vect
 * @param t_vec vect of numeric
 */
template < FA T >
auto mean( const vect< T >& t_vec ) -> T
{
    T sum = 0;
    for ( auto &elem : t_vec )
    {
        sum += elem;
    }
    return sum / t_vec.size( );
}

//-----------------------MISC-----------------------

template < FA T >
auto operator<<( std::ostream& os, const vect< T >& vec ) -> std::ostream&
{
    os << "[";
    for ( int i = 0; i < vec.size( ); ++i ) 
    {
        os << vec.interior[ i ];
        if ( i < vec.size( ) - 1 ) 
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

template < typename T >
auto operator<<( std::ostream& os, const std::vector< T >& vec ) -> std::ostream&
{
    os << "[";
    for ( int i = 0; i < vec.size( ); ++i ) 
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

//-----------------PARALLELIZATION-----------------------

template < FA T >
auto PV_add( sycl::queue& q, const vect< T >& a, const vect< T >& b ) -> std::vector< T >
{
    if ( a.size( ) != b.size( ) ) throw vectDimError{"CANNOT + VECTORS OF UNEQUAL SIZE"};

    std::vector< T > va = a.toVect( );
    std::vector< T > vb = b.toVect( );
    std::vector< T > c( a.size( ) );

    T* dev_a = sycl::malloc_device< T >( va.size( ), q );
    T* dev_b = sycl::malloc_device< T >( va.size( ), q );
    T* dev_c = sycl::malloc_device< T >( va.size( ), q );

    q.memcpy( dev_a, va.data( ), sizeof( T ) * va.size( ) );
    q.memcpy( dev_b, vb.data( ), sizeof( T ) * vb.size( ) );
    q.memcpy( dev_c, c.data( ), sizeof( T ) * c.size( ) );

    q.parallel_for( va.size( ), [=]( sycl::id< 1 > idx )
    {
        dev_c[ idx ] = dev_a[ idx ] + dev_b[ idx ];
    });

    q.memcpy( c.data( ), dev_c, sizeof( T ) * c.size( ) );
    q.wait( );

    sycl::free( dev_a, q );
    sycl::free( dev_b, q );
    sycl::free( dev_c, q );

    return c;
}

/*
template < FA T >
auto PV_dot( sycl::queue& q, const vect< T >& a, const vect< T >& b ) -> T
{
    if ( a.size( ) != b.size( ) ) throw vectDimError{"CANNOT DOT VECTORS OF UNEQUAL SIZE"};
    auto vaFirst = a.toVect( ).begin( );
    auto vaLast = a.toVect( ).end( );
    auto vbFirst = b.toVect( ).begin( );
    return sycl::transform_reduce( q, vaFirst, vaLast, vbFirst, 0.0, std::plus<T>(), std::multiplies<T>() );
}
*/

#endif