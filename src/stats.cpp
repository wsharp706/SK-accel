/**
 * @brief Light statistical algorithms
 * @author Will Sharpsteen - wisharpsteen@gmail.com
 */
#include "matrix.h"
#include "model.h"
#include "customexceptions.h"
#include <vector>
#include <cmath>
#include "stats.h"


auto factorial( const int n ) -> int
{
    if ( n == 1 )
    {
        return 1;
    }
    return n * factorial( n - 1 );
}

auto dfactorial( const int n ) -> int
{
    if ( n % 2 )
    {
        if ( n == 1 )
        {
            return 1;
        }
        return n * dfactorial( n - 2 );
    }
    else
    {
        if ( n == 0 )
        {
            return 1;
        }
        return n * dfactorial( n - 2 );
    }
}

auto gamma_over2( const int t ) -> long double
{
    if  ( t == 1 )
    {
        return sqrt( 3.1415926535897932 );
    }
    if ( t % 2 )
    {
        return dfactorial( 2 * ( t / 2 ) - 1 ) / pow( 2, ( t / 2 ) ) * sqrt( 3.1415926535897932 );
    }
    else
    {
        return factorial( ( t / 2 ) - 1 );
    }
}

auto Gamma( const int a ) -> long double
{
    return gamma_over2( 2 * a );
}

auto Beta( const double a, const double b, bool over_2 ) -> long double
{
    if ( over_2 ) return ( gamma_over2( a ) * gamma_over2( b ) / gamma_over2( a + b ));
    else return ( Gamma( a ) * Gamma( b ) / Gamma( a + b ) );
}

auto dt( const long double x, const int df ) -> long double
{
    //return ( gamma_over2( v + 1 ) / ( gamma_over2( v ) * sqrt( v * pi ) ) ) * ( pow(( 1 + ( pow(x,2) / v )), ( -0.5 * ( v + 1 ) ) ) );
    long double v = df;
    long double log_norm = std::lgammal((v + 1.0L) / 2.0L) - std::lgammal(v / 2.0L) - 0.5L * std::log(v * 3.14159265359);
    long double log_kernel = - (v + 1.0L) / 2.0L * std::log1pl((x*x) / v);
    return expl( log_norm + log_kernel );
}

auto pt( const long double x, const int df, const long double from, const int accuracy ) -> long double
{
    auto t_df = df;
    auto a = from;
    auto n = 3 * accuracy;
    long double h = ( x - a ) / n;
    long double sum = dt( a, t_df ) + dt( x, t_df );
    for ( int i = 1; i < n; ++i )
    {
        if ( i % 3 == 0 )
        {
            sum += 2 * dt( a + i * h, t_df );
        }
        else
        {
            sum += 3 * dt( a + i * h, t_df );
        }
    }
    return ( 3.0 / 8.0 ) * h * sum;
}

auto qt( const long double p, const int df, int maxiter, long double tol ) -> long double
{
    if ( p <= 0.0 || p >= 1.0 ) throw statsError{"CANNOT RETURN qt( ) OUTSIDE ACCEPTABLE PROBABILITY"};
    double lbound = -12;
    double ubound = 12;
    double midpoint = 0;
    double p_lbound = 0;
    double p_ubound = 1;
    double p_mid;
    for ( int iter = 0; iter < maxiter; iter++ )
    {
        midpoint = 0.5 * ( lbound + ubound );
        p_mid = pt( midpoint, df );
        if ( std::abs( p_mid - p ) < tol ) return midpoint;
        if ( p_mid < p )
        {
            lbound = midpoint;
            p_lbound = p_mid;
        }
        else
        {
            ubound = midpoint;
            p_ubound = p_mid;
        }
    }
    return midpoint;
}

auto zphi( const long double x ) -> long double
{
    long double t = 1/(1+0.33267*x);
    return 1 - (0.4361836*t - 0.1201676*powl(t,2) + 0.9372980*powl(t,3))*(sqrtl(1/(2*M_PI)))*(expl(-0.5*(powl(x,2))));
}

auto dchi( const long double x, const int df, bool log ) -> long double
{
    if ( x <= 0 ) return 0;
    if ( !log ) return expl( -0.5 * df * logl( 2 ) - lgammal( 0.5 * df ) + ( 0.5 * df - 1) * logl( x ) - 0.5 * x );
    else return ( -0.5 * df * logl( 2 ) - lgammal( 0.5 * df ) + ( 0.5 * df - 1) * logl( x ) - 0.5 * x );
}

auto pf( const long double x, const int df1, const int df2, const bool ltail ) -> long double
{
    long double z = powl( ((2.0*df2 + df1*x/3.0 + df1 - 2.0)*x)/(2.0*df2+4.0*df1*x/3.0) ,1.0/3.0);
    z += -1.0*( 1.0 - ( 2.0 /(9.0*df1) ) );
    z = z / ( sqrtl(2.0/(9.0*df1)) );
    if ( !ltail ) return 1 - zphi(z);
    else return zphi(z);
}

auto qf( const long double p, const int df1, const int df2, int maxiter, long double tol ) -> long double
{
    if ( p <= 0.0 || p >= 1.0 ) throw statsError{"CANNOT RETURN qf( ) OUTSIDE ACCEPTABLE PROBABILITY"};
    double lbound = 0.00001;
    double ubound = 16;
    double midpoint;
    double p_lbound = 0;
    double p_ubound = 1;
    double p_mid;
    for ( int iter = 0; iter < maxiter; iter++ )
    {
        midpoint = 0.5 * ( lbound + ubound );
        p_mid = pf( midpoint, df1, df2 );
        if ( std::abs( p_mid - p ) < tol ) return midpoint;
        if ( p_mid < p )
        {
            lbound = midpoint;
            p_lbound = p_mid;
        }
        else
        {
            ubound = midpoint;
            p_ubound = p_mid;
        }
    }
    return midpoint;
}



