/**
 * @brief Additional Vector Utilites
 * @author Will Sharpsteen wisharpsteen@gmail.com
 */

#include <vector>
#include <iostream>
#include <concepts>
#include <type_traits>
#include <typeinfo>
#include <sycl/sycl.hpp>
#include <hipSYCL/algorithms/numeric.hpp>
#include <hipSYCL/algorithms/algorithm.hpp>
#include "customexceptions.h"
#include "gpu.h"
#include "templates.h"
#include "util.h"


#ifndef VECT_H 
#define VECT_H 

namespace SKAS::vect 
{

    /**
     * @brief Wrapper of std::vector class with supplemental utility and parallelization support
     */
    template < SKAS::FlAd T >
    class vect
    {
        private:
        bool parallel;

        public:
        std::vector< T > interior;

        vect( ) : parallel( false ) { };

        ~vect( ) { }

        vect( const vect& orig ) : parallel( false )
        {
            interior = orig.interior;
            parallel = orig.parallel;
        }

        vect( const size_t dim, bool parallel = false )
        {
            std::vector< T > t_interior( dim );
            interior = t_interior;
            parallel = parallel;
        }

        vect( const size_t dim, const T init_value, bool parallel = false )
        {
            std::vector< T > t_interior( dim, init_value );
            interior = t_interior;
            parallel = parallel;
        }
        
        vect( const std::vector< T >& orig, const bool& par ) : parallel( false )
        {
            interior = orig;
            parallel = par;
        }

        vect( const std::vector< T >& orig ) : parallel( false )
        {
            interior = orig;
            parallel = false;
        }

        vect& operator=( const vect& other )
        {
            if ( this != &other )
            {
                interior = other.interior;
                parallel = other.parallel;
            }
            return *this;
        }

        vect& operator=( const std::vector< T >& other )
        {
            interior = other;
            parallel = false;
            return *this;
        }

        vect( const std::initializer_list< T > init, const bool& t_parallel = false )
        {
            interior = init;
            parallel = t_parallel;
        }

        vect( const bool& t_parallel )
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

        auto erase( std::vector< T >::iterator first, std::vector< T >::iterator last ) -> void
        {
            interior.erase( first, last );
        }

        auto begin( ) -> std::vector< T >::iterator
        {
            return interior.begin( );
        }

        auto begin( ) const -> std::vector< T >::const_iterator
        {
            return interior.begin( );
        }

        auto end( ) -> std::vector< T >::iterator
        {
            return interior.end( );
        }

        auto end( ) const -> std::vector< T >::const_iterator
        {
            return interior.end( );
        }

        auto insert( std::vector< T >::const_iterator position, const T& val ) -> void
        {
            interior.insert( position, val );
        }

        auto insert( std::vector< T >::iterator position, const T& val ) -> void
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

        auto enddata( ) -> T*
        {
            return interior.data( ) + interior.size( );
        }

        auto enddata( ) const -> const T*
        {
            return interior.data( ) + interior.size( );
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

        auto toVect( ) -> std::vector< T >
        {
            return interior;
        }

        auto toVect( ) const -> const std::vector< T >
        {
            return interior;
        }

        auto isEmpty( ) const -> bool
        {
            return interior.empty( );
        }

    };
};

namespace SKAS::vect::accel_vect 
{
    template < SKAS::FlAd T1 >
    auto PV_sub( const SKAS::vect::vect< T1 >& a, const SKAS::vect::vect< T1 >& b ) -> SKAS::vect::vect< T1 >;

    template < SKAS::FlAd T1 >
    auto PV_add( const SKAS::vect::vect< T1 >& a, const SKAS::vect::vect< T1 >& b ) -> SKAS::vect::vect< T1 >;

    template < SKAS::FlAd T1 >
    auto PV_dot( const SKAS::vect::vect< T1 >& a, const SKAS::vect::vect< T1 >& b ) -> T1; 

    template < SKAS::FlAd T >
    auto PV_mag( const SKAS::vect::vect< T >& a ) -> T;

    template < SKAS::FlAd T1 >
    auto PV_scale( const SKAS::vect::vect< T1 >& a, const T1 scalar ) -> SKAS::vect::vect< T1 >;

    template < SKAS::FlAd T1 >
    auto PV_cov( const SKAS::vect::vect< T1 >& a, const SKAS::vect::vect< T1 >& b ) -> T1;

    template < SKAS::FlAd T1 >
    auto PV_s2( const SKAS::vect::vect< T1 >& a ) -> T1;

    template < SKAS::FlAd T1 >
    auto PV_mean( const SKAS::vect::vect< T1 >& a ) -> T1;
};

namespace SKAS::vect
{
    // -----------------LINEAR ALGEBRA----------------------

    /**
     * @brief addition operator support for std::vector
     * @param first vect
     * @param last vect
     * @return vect result from operation
     * @exception vectDimError thrown when first.size( ) != last.size( ) 
     */
    template < SKAS::FlAd T1, SKAS::FlAd T2 >
    auto operator+( const vect< T1 >& first, const vect< T2 >& last ) -> vect< T1 >
    {
        if ( first.is_parallel( ) && last.is_parallel( ) )
        {
            return accel_vect::PV_add( first, last );
        }
        if ( first.size( ) != last.size( ) )
        {
            throw vectDimError{"CANNOT + VECTOR OF DIFFERENT SIZES"};
        }
        vect< T1 > output;
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
    template < SKAS::FlAd T1, SKAS::FlAd T2 >
    auto operator-( const vect< T1 >& first, const vect< T2 >& last ) -> vect< T1 >
    {
        if ( first.is_parallel( ) && last.is_parallel( ) )
        {
            return accel_vect::PV_sub( first, last );
        }
        if ( first.size( ) != last.size( ) )
        {
            throw vectDimError{"CANNOT - VECTOR OF DIFFERENT SIZES"};
        }
        vect< T1 > output;
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
    template < SKAS::FlAd T1, SKAS::FlAd T2 >
    auto operator*( const vect< T1 >& t_vec, const T2& scalar ) -> vect< T1 >
    {
        if ( t_vec.is_parallel( ) ) return accel_vect::PV_scale< T1 >( t_vec, scalar );
        vect< T1 > output( t_vec.size( ), false );
        size_t i = 0;
        for ( const auto &elem : t_vec.interior )
        {
            output[ i++ ] = elem * scalar;
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
    template < SKAS::FlAd T1, SKAS::FlAd T2 >
    auto operator*( const vect< T1 >& first, const vect< T2 >& last ) -> T1
    {
        if ( first.is_parallel( ) && last.is_parallel( ) ) return accel_vect::PV_dot( first, last );
        if( first.size( ) != last.size( ) )
        {
            throw vectDimError{"CANNOT DOT PRODUCT VECTORS OF DIFFERENT DIMENSION!"};
        }
        T1 final = 0;
        for ( int i = 0; i < first.size( ); ++i )
        {
            final += first[ i ] * last[ i ];
        }
        return final;
    }

    /**
     * @brief Euclidean norm operator
     * @param t_vec vect
     * @return .numeric magnitude
     */
    template < SKAS::FlAd T1 >
    auto mag( const vect< T1 >& t_vec ) -> T1
    {
        if ( t_vec.is_parallel( ) ) return accel_vect::PV_mag( t_vec );
        double sum = 0;
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
    template < SKAS::FlAd T1 >
    auto makeunit( const vect< T1 >& t_vec ) -> vect< T1 >
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
    template < SKAS::FlAd T1, SKAS::FlAd T2 >
    auto vcomp( const vect< T1 >& first, const vect< T2 >& last, float error = 0.000001 ) -> bool
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
    template < SKAS::FlAd T1, SKAS::FlAd T2 >
    auto vcomp( const std::vector< T1 >& first, const std::vector< T2 >& last, float error = 0.000001 ) -> bool
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

    template < SKAS::FlAd T1, SKAS::FlAd T2 >
    auto operator==( const vect< T1 >& rhs, const vect< T2 >& lhs ) -> bool
    {
        return vcomp( rhs, lhs );
    }

    template < SKAS::FlAd T1, SKAS::FlAd T2 >
    auto operator!=( const vect< T1 >& rhs, const vect< T2 >& lhs ) -> bool
    {
        return !vcomp( rhs, lhs );
    }

    template < SKAS::FlAd T1, SKAS::FlAd T2 >
    auto operator==( const std::vector< T1 >& rhs, const std::vector< T2 >& lhs ) -> bool
    {
        return vcomp( rhs, lhs );
    }
    
    template < SKAS::FlAd T1, SKAS::FlAd T2 >
    auto operator!=( const std::vector< T1 >& rhs, const std::vector< T2 >& lhs ) -> bool
    {
        return !vcomp( rhs, lhs );
    }

    template < SKAS::FlAd T1 >
    auto toMV( const std::vector< T1 >& t_vector, bool parallel ) -> vect< T1 >
    {
        return vect< T1 >( t_vector, parallel );
    } 

    // -----------------STATS--------------------------

    /**
     * @brief compute covariance between two vectors
     * @param a_vec vect 
     * @param b_vec vect
     * @exception vectDimError thrown for incompatible sizings.
     * @return .numeric
     */
    template < SKAS::FlAd T1, SKAS::FlAd T2 >
    auto cov( const vect< T1 >& a_vec, const vect< T2 >& b_vec ) -> T1
    {
        if ( a_vec.is_parallel( ) && b_vec.is_parallel( ) ) return accel_vect::PV_cov( a_vec, b_vec );
        if ( a_vec.size( ) != b_vec.size( ) ) throw vectDimError{"CANNOT COMPUTE COV OF INCOMPATIBLE VECTORS"};
        T1 out = 0;
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
    template < SKAS::FlAd T1, SKAS::FlAd T2 >
    auto corr( const vect< T1 >& a_vec, const vect< T2 >& b_vec ) -> T1
    {
        return cov( a_vec, b_vec ) / ( s(a_vec) * s(b_vec) );
    }


    /**
     * @brief variance
     * @param t_vector vect
     * @return .numeric
     */
    template < SKAS::FlAd T >
    auto s2( const vect< T >& t_vector ) -> T
    {
        if ( !t_vector.size( ) || t_vector.size( ) == 1 )
        {
            return 0;
        }
        if ( t_vector.is_parallel( ) ) return accel_vect::PV_s2( t_vector );
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
    template < SKAS::FlAd T >
    auto s( const vect< T >& t_vector ) -> T
    {
        return sqrt( s2( t_vector ) );
    }


    /**
     * @brief Compute mean of vect
     * @param t_vec vect of numeric
     */
    template < SKAS::FlAd T >
    auto mean( const vect< T >& t_vec ) -> T
    {
        if ( t_vec.is_parallel( ) ) return accel_vect::PV_mean( t_vec ); 
        T sum = 0;
        for ( auto &elem : t_vec )
        {
            sum += elem;
        }
        return sum / t_vec.size( );
    }

    //-----------------------MISC-----------------------

    template < SKAS::FlAd T >
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

}; //NAMESPACE SKAS::vect

namespace SKAS::vect::accel_vect
{
    template < SKAS::FlAd T1 >
    auto PV_add( const SKAS::vect::vect< T1 >& a, const SKAS::vect::vect< T1 >& b ) -> SKAS::vect::vect< T1 >
    {
        if ( a.size( ) != b.size( ) ) throw vectDimError{"CANNOT + VECTORS OF UNEQUAL SIZE"};

        sycl::queue q = gpu::ctx( ).q;

        vect< T1 > c( a.size( ), true );

        T1* dev_a = sycl::malloc_device< T1 >( a.size( ), q );
        T1* dev_b = sycl::malloc_device< T1 >( a.size( ), q );
        T1* dev_c = sycl::malloc_device< T1 >( a.size( ), q );

        q.memcpy( dev_a, a.data( ), sizeof( T1 ) * a.size( ) );
        q.memcpy( dev_b, b.data( ), sizeof( T1 ) * b.size( ) );
        q.memcpy( dev_c, c.data( ), sizeof( T1 ) * c.size( ) );

        hipsycl::algorithms::transform( q, dev_a, dev_a + a.size( ), dev_b, dev_c, std::plus<T1>() );

        q.memcpy( c.data( ), dev_c, sizeof( T1 ) * c.size( ) );
        q.wait( );

        sycl::free( dev_a, q );
        sycl::free( dev_b, q );
        sycl::free( dev_c, q );

        return c;
    }

    template < SKAS::FlAd T1 >
    auto PV_sub( const vect< T1 >& a, const vect< T1 >& b ) -> vect< T1 >
    {
        if ( a.size( ) != b.size( ) ) throw vectDimError{"CANNOT - VECTORS OF UNEQUAL SIZE"};

        sycl::queue q = gpu::ctx( ).q;

        vect< T1 > c( a.size( ), true );

        T1* dev_a = sycl::malloc_device< T1 >( a.size( ), q );
        T1* dev_b = sycl::malloc_device< T1 >( a.size( ), q );
        T1* dev_c = sycl::malloc_device< T1 >( a.size( ), q );

        q.memcpy( dev_a, a.data( ), sizeof( T1 ) * a.size( ) );
        q.memcpy( dev_b, b.data( ), sizeof( T1 ) * b.size( ) );
        q.memcpy( dev_c, c.data( ), sizeof( T1 ) * c.size( ) );

        hipsycl::algorithms::transform( q, dev_a, dev_a + a.size( ), dev_b, dev_c, std::minus<T1>() );
        
        q.memcpy( c.data( ), dev_c, sizeof( T1 ) * c.size( ) );
        q.wait( );

        sycl::free( dev_a, q );
        sycl::free( dev_b, q );
        sycl::free( dev_c, q );

        return c;
    }

    
    template < SKAS::FlAd T1 >
    auto PV_scale( const SKAS::vect::vect< T1 >& a, const T1 scalar ) -> vect< T1 >
    {
        sycl::queue q = gpu::ctx( ).q;

        vect< T1 > c( a.size( ), true );

        T1* dev_a = sycl::malloc_device< T1 >( a.size( ), q );

        q.memcpy( dev_a, a.data( ), sizeof( T1 ) * a.size( ) );

        auto f = util::make_multiplier< T1, T1 >( scalar );

        hipsycl::algorithms::transform( q, dev_a, dev_a + a.size( ), dev_a, f );

        q.memcpy( c.data( ), dev_a, sizeof( T1 ) * c.size( ) );
        q.wait( );

        sycl::free( dev_a, q );

        return c;
    }
        
        
    template < SKAS::FlAd T1 >
    auto PV_dot( const vect< T1 >& a, const vect< T1 >& b ) -> T1
    {
        if ( a.size( ) != b.size( ) ) throw vectDimError{"CANNOT DOT VECTORS OF UNEQUAL SIZE"};

        sycl::queue& q = gpu::ctx( ).q;

        T1* dev_a = sycl::malloc_device< T1 >( a.size( ), q );
        T1* dev_b = sycl::malloc_device< T1 >( a.size( ), q );
        T1* dev_c = sycl::malloc_device< T1 >( 1, q );

        q.memcpy( dev_a, a.data( ), sizeof( T1 ) * a.size( ) );
        q.memcpy( dev_b, b.data( ), sizeof( T1 ) * b.size( ) );

        hipsycl::algorithms::transform_reduce( q, gpu::ctx().ag, dev_a, dev_a + a.size( ), dev_b, dev_c, T1{0}, std::plus<T1>(), std::multiplies<T1>() );

        T1 out;
        q.memcpy( &out, dev_c, sizeof( T1 ) );
        q.wait( );

        sycl::free( dev_a, q );
        sycl::free( dev_b, q );
        sycl::free( dev_c, q );

        return out;
    }

    template < SKAS::FlAd T >
    auto PV_mag( const vect< T >& a ) -> T
    {
        if ( a.size( ) == 0 ) return T{0};
        if ( a.size( ) == 1 ) return a[0];

        sycl::queue& q = gpu::ctx( ).q;

        T* dev_a = sycl::malloc_device< T >( a.size( ), q );
        T* dev_c = sycl::malloc_device< T >( 1, q );

        q.memcpy( dev_a, a.data( ), sizeof( T ) * a.size( ) );

        hipsycl::algorithms::transform_reduce( q, gpu::ctx().ag, dev_a, dev_a + a.size( ), dev_c, T{0}, std::plus<T>(), SKAS::util::sqr<T>() );

        T out;
        q.memcpy( &out, dev_c, sizeof( T ) );
        q.wait( );

        sycl::free( dev_a, q );
        sycl::free( dev_c, q );
        
        return sqrt(out);
    }

    template < SKAS::FlAd T1 >
    auto PV_cov( const SKAS::vect::vect< T1 >& a, const SKAS::vect::vect< T1 >& b ) -> T1
    {
        if ( a.size( ) != b.size( ) ) throw vectDimError{"CANNOT COMPUTE COV OF INCOMPATIBLE SIZED VECTORS"};

        sycl::queue& q = gpu::ctx( ).q;

        auto f = util::tsum( mean( a ), mean( b ) );

        T1* dev_xbar_a = sycl::malloc_device< T1 >( 1, q );
        T1* dev_xbar_b = sycl::malloc_device< T1 >( 1, q );
        T1* dev_a = sycl::malloc_device< T1 >( a.size( ), q );
        T1* dev_b = sycl::malloc_device< T1 >( b.size( ), q );
        T1* dev_c = sycl::malloc_device< T1 >( 1, q );
        T1 zero = 0;

        q.memcpy( dev_a, a.data( ), sizeof( T1 ) * a.size( ) );
        q.memcpy( dev_b, b.data( ), sizeof( T1 ) * b.size( ) );
        q.memcpy( dev_c, &zero, sizeof( T1 ) );

        hipsycl::algorithms::transform_reduce( q, gpu::ctx().ag, dev_a, dev_a + a.size(), dev_b, dev_c, T1{0}, std::plus<T1>(), f );

        T1 out;
        q.memcpy( &out, dev_c, sizeof( T1 ) );
        q.wait( );

        sycl::free( dev_a, q );
        sycl::free( dev_b, q );
        sycl::free( dev_c, q );

        return out / (a.size( ) - 1.0);
    }

    template < SKAS::FlAd T1 >
    auto PV_s2( const SKAS::vect::vect< T1 >& t_vector ) -> T1
    {
        sycl::queue& q = gpu::ctx().q;

        auto f = util::ssum( mean( t_vector ) );

        T1* dev_a = sycl::malloc_device< T1 >( t_vector.size( ), q );
        T1* dev_c = sycl::malloc_device< T1 >( 1, q );

        q.memcpy( dev_a, t_vector.data( ), sizeof( T1 ) * t_vector.size( ) );
        
        hipsycl::algorithms::transform_reduce( q, gpu::ctx().ag, dev_a, dev_a + t_vector.size( ), dev_c, T1{0}, std::plus<T1>(), f );

        T1 out;
        q.memcpy( &out, dev_c, sizeof( T1 ) );
        q.wait( );

        sycl::free( dev_a, q );
        sycl::free( dev_c, q );

        return out / (t_vector.size( ) - 1.0 );
    }

    template < SKAS::FlAd T1 >
    auto PV_mean( const SKAS::vect::vect< T1 >& a ) -> T1
    {
        sycl::queue& q = gpu::ctx().q;

        T1* dev_a = sycl::malloc_device< T1 >( a.size( ), q );
        T1* dev_c = sycl::malloc_device< T1 >( 1, q );

        q.memcpy( dev_a, a.data( ), sizeof( T1 ) * a.size( ) );

        hipsycl::algorithms::reduce( q, gpu::ctx().ag, dev_a, dev_a + a.size( ), dev_c, T1{0}, std::plus< T1 >() );

        T1 out;
        q.memcpy( &out, dev_c, sizeof( T1 ) );
        q.wait( );

        sycl::free( dev_a, q );
        sycl::free( dev_c, q );

        return out / static_cast< T1 >(a.size( ));
    }
        
}; //NAMESPACE SKAS::vect::accel_vect

#endif




