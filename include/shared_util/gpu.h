/**
 * @brief SYCL/AdaptiveCpp backend utility
 * @author Will Sharpsteen - wisharpsteen@gmail.com
 */
#include <sycl/sycl.hpp>
#include <hipSYCL/algorithms/numeric.hpp>
#include <hipSYCL/algorithms/algorithm.hpp>

#ifndef GPU_H
#define GPU_H

namespace SKAS::gpu
{
    struct gpu_context
    {
        sycl::queue q;
        hipsycl::algorithms::util::allocation_group ag;

        gpu_context() : q{sycl::property::queue::in_order{}}, ag{} {}
    };

    inline gpu_context& ctx( ) 
    {
        static gpu_context instance{ };
        return instance;
    }

};



#endif
