/**
 * @brief Implentation of model functionality
 * @author Will Sharpsteen - wisharpsteen@gmail.com
 */

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <bits/stdc++.h>
#include "model.h"
#include "matrix.h"
#include "stats.h"
#include "customexceptions.h"


// =========================START OF MEMBER FUNCTIONS FOR MATRIX CLASS=========================
model::model( ) : n{ 0 }
{
}

model::~model( ) 
{
}

model::model( const model& o ) : n{ 0 }
{
    rawdata = o.rawdata;
    X = o.X;
    Xt = o.Xt;
    XtX_inv =  o.XtX_inv;
    Y = o.Y;
    residuals =  o.residuals;
    beta = o.beta;
    n = o.n;
    response_col = o.response_col;
    explanatory_cols = o.explanatory_cols;
    transform = o.transform;
    index_mode = o.index_mode;
}

auto model::nparam( ) const -> unsigned int
{
    return beta.nrow( );
}

auto model::nobs( ) const -> unsigned int
{
    return n;
}

auto model::coef( ) const -> matrix
{
    return beta;
}

auto model::hat( ) const -> matrix
{
    return X % XtX_inv % Xt;
}

auto model::yhat( ) const -> matrix
{
    return hat( ) % Y;
}

auto model::res( ) const -> matrix
{
    return residuals;
}

auto model::SSE( ) const -> long double
{
    return ( Y.t() % ( identity( n ) - hat( ) ) % Y ).getelem( 1, 1 );
}

auto model::SSTO( ) const -> long double
{
    return ( Y.t( ) % ( identity( n ) - ( 1.0 / n ) * init( 1, n, n ) ) % Y ).getelem( 1, 1 );
}

auto model::SSR( ) const -> long double
{
    return ( Y.t( ) % ( hat( ) - ( 1.0 / n ) * init( 1, n, n ) ) % Y ).getelem( 1, 1 );
}

auto model::MSE( ) const -> long double 
{
    return ( 1.0 / ( nobs( ) - nparam( ) ) ) * SSE( );
}

auto model::MSR( ) const -> long double
{
    return ( 1.0 / ( nparam( ) - 1 ) ) * SSR( );
}

auto model::getcoef( const int which ) const -> long double
{
    if ( which > nparam( ) - 1 ) throw dimSizeError{"GETCOEF OUT OF RANGE"};
    return coef( ).getelem( which, 0, true );
}

auto model::getcoefSE( const int which ) const -> long double
{
    if ( which > nparam( ) - 1 ) throw dimSizeError{"GETCOEF OUT OF RANGE"};
    return SE( ).getelem( which, 0, true );
}

auto model::vcov( const std::string& which ) const -> matrix
{
    if ( which == "b" )
    {
        return MSE( ) * XtX_inv;
    }
    if ( which == "e" )
    {
        return MSE( ) * ( identity( n ) - hat( ) );
    }
    if ( which == "yh" )
    {
        return MSE( ) * hat( );
    }
    else
    {
        matrix empty;
        return empty;
    }
}

auto model::SE( ) const -> matrix
{
    return sqrt( diag( MSE( ) * XtX_inv ) );
}

auto model::bci( const float alpha ) const -> matrix
{
    matrix out = init( 0, nparam( ), 3 );
    for ( int param = 0; param < nparam( ); ++param )
    {
        out.setelem( getcoef( param ) , param, 0, true );
        out.setelem( getcoef( param ) - getcoefSE( param ) * qt( 1 - alpha/2, nobs( ) - nparam( ) ), param, 1, true );
        out.setelem( getcoef( param ) + getcoefSE( param ) * qt( 1 - alpha/2, nobs( ) - nparam( ) ), param, 2, true );
    }
    return out;
}

auto model::testF( const std::vector< unsigned int >& to_drop, bool pval ) const -> long double
{
    std::vector< unsigned int > given = explanatory_cols; 
    for ( int i = 0; i < to_drop.size( ); ++i )
    {
        auto it = find( given.begin( ), given.end( ), to_drop[ i ] );
        if ( it != given.end( ) )
        {
            given.erase( given.begin( ) + distance( given.begin( ), it ) );
        }
    }
    auto Fstat = ( ESSR( to_drop, given ) / to_drop.size( ) ) / MSE( );
    if ( !pval ) return Fstat;
    else return pf( Fstat, to_drop.size( ), nobs( ) - nparam( ) );
}

auto model::s_yh( const matrix& preds ) const -> long double
{
    auto p = preds;
    p.insertcol( { 1 }, 0 );
    return sqrt( MSE( )*( p % XtX_inv % p.t() ).getelem( 1, 1 ) );
}

auto model::predict( const matrix& preds ) const -> long double
{
    auto p = preds;
    p.insertcol( { 1 }, 0 );
    return ( p % coef( ) ).getelem( 1, 1 );
}

auto model::spred( const matrix& preds ) const -> long double
{
    return sqrt( MSE( ) + pow(s_yh( preds ),2) );
}

auto model::mrci( const float alpha, const matrix& preds ) const -> matrix
{
    auto first = predict( preds );
    auto second = s_yh( preds ) * qt( 1 - alpha / 2, nobs( ) - nparam( ) );
    matrix out = { { first - second, first + second } };
    return out;
}

auto model::pi( const float alpha, const matrix& preds ) const -> matrix
{
    auto first = predict( preds );
    auto second = spred( preds ) * qt( 1 - alpha/2, nobs( ) - nparam( ) );
    matrix out = { { first - second, first + second } };
    return out;
}

auto model::ESSR( const std::vector< unsigned int >& new_, const std::vector< unsigned int >& given ) const -> long double
{
    std::vector< unsigned int > all_v;
    for ( auto &elem : new_ )
    {
        all_v.push_back( elem );
    }
    for ( auto &elem : given )
    {
        all_v.push_back( elem );
    }
    model all = fitlm( rawdata, response_col, all_v, index_mode, transform );
    model rest = fitlm( rawdata, response_col, given, index_mode, transform );
    return rest.SSE( ) - all.SSE( );
}

auto model::R2( ) const -> long double
{
    return SSE( ) - SSTO( );
}

auto model::R2partial( const std::vector< unsigned int >& which, const std::vector< unsigned int >& given ) const -> long double
{
    auto all = given;
    for ( auto &elem : which )
    {
        all.push_back( elem );
    }
    return fitlm( rawdata, response_col, all, index_mode, transform ).ESSR( which, given ) / fitlm( rawdata, response_col, given, index_mode, transform ).SSE( );
}

auto model::rpartial( const std::vector< unsigned int >& which, const std::vector< unsigned int >& given ) const -> long double
{
    return sqrtl( R2partial( which, given ) );
}

auto model::corXX( ) const -> matrix
{
    matrix out = init( 0, nparam( )-1, nparam( )-1);
    for ( int i = 0; i < nparam( )-1; ++i )
    {
        for ( int j = 0; j < nparam( )-1; ++j )
        {
            if ( i == j ) out.setelem( 1, j, i, true );
            else out.setelem( corr( rawdata.getcol( explanatory_cols[ j ] - !index_mode ), rawdata.getcol( explanatory_cols[ i ] - !index_mode ) ), j, i, true );
        }
    }
    return out;
}

auto model::corYX( ) const -> matrix
{
    matrix out;
    std::vector< long double > rs;
    for ( auto &row : explanatory_cols )
    {
        rs.push_back( corr( rawdata.getcol( response_col - !index_mode ), rawdata.getcol( row - !index_mode ) ) );
    }
    out.appendcol( rs );
    return out;
}

auto model::destandardize( ) const -> matrix
{
    if ( transform != "standardize" )
    {
        std::cout << "Model is already not standardized." << std::endl;
        return coef( );
    }
    else
    {
        auto sy = s( preSTANDARD.getcol( response_col - !index_mode ) );
        long double sx;
        long double beta;
        long double sum = 0;
        matrix out = init(0,nparam(),1);
        for ( int i = 0; i < nparam( ) - 1; ++i )
        {
            sx = s( preSTANDARD.getcol( explanatory_cols[ i ] - !index_mode ) );
            beta = (sy/sx) * coef( ).getelem( i + 1, 0, true );
            out.setelem( beta, i + 1, 0, true );
            sum -= mean( preSTANDARD.getcol( explanatory_cols[ i ] - !index_mode ) ) * beta;
        }
        out.setelem( mean(preSTANDARD.getcol( response_col - !index_mode )) + sum, 0, 0, true );
        return out;
    }
}



// =========================END OF MEMBER FUNCTIONS FOR MATRIX CLASS=========================

auto operator<<( std::ostream& os, model& t_model ) -> std::ostream&
{
    os << "n: [" << t_model.nobs( ) << "] beta: " << t_model.coef( );
    return os;
}
