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
};

#endif