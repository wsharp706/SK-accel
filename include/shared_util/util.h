/**
 * @brief Misc utility needed 
 * @author Will Sharpsteen
 */
#include <cmath>
#include "templates.h"


#ifndef UTIL_H
#define UTIL_H

namespace SKAS::util
{
    // some functors needed for acceleration support

    // used in vect.h for PV_mag()
    template < SKAS::FlAd T >
    struct sqr
    {
        auto operator()( const T& base ) const -> T
        {
            return pow( base, 2 );
        } 
    };

    //used in vect.h for PV_scale()
    template < SKAS::FlAd T1, SKAS::FlAd T2 >
    auto make_multiplier( T1 scalar )
    {
        return [scalar]( T2 x ) { return x * scalar; };
    }

    //used in vect.h for PV_cov()
    template < SKAS::FlAd T1, SKAS::FlAd T2 >
    auto tsum( const T1& aavg, const T2& bavg )
    {
        return [aavg,bavg]( T1 ai, T2 bi ) { return (ai-aavg)*(bi-bavg); };
    }

    //used in vect.h for PV_s2()
    template < SKAS::FlAd T1 >
    auto ssum( const T1& aavg )
    {
        return [aavg]( T1 ai ) { return (ai-aavg)*(ai-aavg); };
    }
};


#endif