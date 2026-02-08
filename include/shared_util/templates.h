/**
 * @brief Templates allowed restriction
 * @author Will Sharpsteen - wisharpsteen@gmail.com
 */
#include <concepts>

#ifndef TEMPLATES_H
#define TEMPLATES_H

namespace SKAS
{
    template < typename T >
    concept FlAd =  std::is_same_v< T, float > || 
                    std::is_same_v< T, double >;

    template < typename T >
    concept NUMERIC =   std::is_same_v< T, float > ||
                        std::is_same_v< T, double > ||
                        // ==unsupported on amdgpus== std::is_same_v< T, long double > ||
                        std::is_same_v< T, int > ||
                        std::is_same_v< T, long int >;
    };

#endif