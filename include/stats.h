/**
 * @brief Stats helper functions
 * @author Will Sharpsteen - wisharpsteen@gmail.com
 */
#include "matrix.h"
#include "model.h"
#include "customexceptions.h"
#include <vector>
#include <cmath>

#ifndef STATS_H
#define STATS_H

/**
 * @brief factorial
 * @param n int
 * @return int
 */
auto factorial( const int n ) -> int;

/**
 * @brief double factorial
 * @param n int
 * @return int
 */
auto dfactorial( const int n ) -> int;

/**
 * @brief Gamma function for 1/2 int increments
 * @param t int
 * @return long double Gamma( t / 2 )
 */
auto gamma_over2( const int t ) -> long double;

/**
 * @brief Gamma function
 * @param a int
 * @return long double Gamma( a )
 */
auto Gamma( const int a ) -> long double;
/**
 * @brief Beta function
 * @param a int
 * @param b int
 * @param over_2 bool. if true, then passing Beta( a / 2, b / 2 )
 * @return long double
 */
auto Beta( const double a, const double b, bool over_2 = false ) -> long double;

/**
 * @brief pdf of chi^2 distrubution
 * @param x double
 * @param k degrees of freedom
 * @return long double
 */
auto dchi( const long double x, const int df, bool log = false ) -> long double;

/**
 * @brief cdf of standard normal
 * @param x
 * @return cdf from -infty to x of a standard normal
 */
auto zphi( const long double x ) -> long double;

/**
 * @brief pdf of student t distribution
 * @param x double
 * @param v int, df
 * @return long double
 */
auto dt( const long double x, const int df ) -> long double;

/**
 * @brief cdf function of f distribution
 * @param x double
 * @param df1 int
 * @param df2 int
 * @param from left bound for iterative process. leave alone unless required.
 * @param accuracy iteration amount for estimation.
 */
auto pf( const long double x, const int df1, const int df2, const bool ltail = false ) -> long double;

/**
 * @brief inverse-cdf function of f distribution
 * @param x double
 * @param df1 int
 * @param df2 int
 * @param tol double. tolerated amount of errror
 * @param maxiter int. typically this default value will never be hit. this is a binary search with fast convergence.
 */
auto qf( const long double p, const int df1, const int df2, int maxiter = 1000, long double tol = 0.0001 ) -> long double;
/**
 * @brief cdf function of student-t distribution
 * @param x double
 * @param df int
 * @param from left bound for iterative process. leave alone unless required.
 * @param accuracy iteration amount for estimation.
 */
auto pt( const long double x, const int df, const long double from = -100, const int accuracy = 100 ) -> long double;

/**
 * @brief inverse-cdf function of student-t distribution
 * @param x double
 * @param df int
 * @param tol double. tolerated amount of errror
 * @param maxiter int. typically this default value will never be hit. this is a binary search with fast convergence.
 */
auto qt( const long double p, const int df, int maxiter = 1000, long double tol = 0.0001 ) -> long double;

#endif