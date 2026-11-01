#pragma once

#include <stdint.h>

#include "LookUpTable.hpp"

namespace FixedPointMath
{
    static constexpr LUT<9, -1, 1, 8, false, &HighDefFuncLib::hdExp> expLUT;
    
    int16_t exp(int16_t x)
    {
        int16_t nx = x > 0 ? x : -x; // Take abs

        constexpr double argThreshold = 1.0;
        constexpr int16_t fixedArgThreshold = FixedPointMath::toFixedP<int16_t>(argThreshold, expLUT.dp);

        // This needs arg reduction to fit within LUT bounds
        uint16_t k = 0;
        while (nx > fixedArgThreshold)
        {
            nx >>= 1;
            k++;
        }

        nx = x > 0 ? nx : -nx;

        int16_t expOutput = expLUT.lookup(nx);
        int32_t accumulator = (int32_t)expOutput;
        
        // Rescale
        for (uint16_t i = 0; i < k; i++)
        {
            accumulator *= accumulator;
            accumulator >>= expLUT.dp;
        }

        return (int16_t)accumulator;
    }
}