#pragma once
#include <stdint.h>
#include <stddef.h>

#include "Numerics.hpp"

namespace HighDefFuncLib
{
    using ConstexprMath::exp;
    
    // Only for computational use
    constexpr double hdExp(double x)
    {
        return exp(x);
    }

    constexpr double hdSigmoid(double x)
    {
        return 1.0 / (1 + exp(-x));
    }

    constexpr double hdTanh(double x)
    {
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
    }
}

// (Entries - 1) and range should be powers of 2 for efficent computation
template <size_t Entries, int XMin, int XMax, size_t Dp, bool HasLimits, double (*refFunc)(double)>
struct LUT
{   
    struct ValWrapper
    {
        int16_t vals[HasLimits ? Entries + 2 : Entries]; // +2 for out of bounds cases
    };

    ValWrapper entries;
    
    static_assert(XMax > XMin, "LUT bounds flipped");
    static_assert(ConstexprMath::log2(Entries - 1) != -1, "LUT entries not power of 2");
    static_assert(ConstexprMath::log2(XMax - XMin) != -1, "LUT range not power of 2");

    static constexpr uint8_t dp = Dp;
    static constexpr double deltaX = (double)(XMax - XMin) / (double)(Entries - 1);
    static constexpr int32_t fixedDeltaX = FixedPointMath::toFixedP<int32_t>(deltaX, Dp);
    static constexpr int32_t fixedXMax = (int32_t)((uint32_t)XMax << Dp);
    static constexpr int32_t fixedXMin = (int32_t)((uint32_t)XMin << Dp);

    constexpr LUT() 
        : entries(populate())
    {}

    static constexpr ValWrapper populate()
    {
        ValWrapper results{}; 
        
        for (size_t i = 0; i < Entries; i++)
        {
            double x = XMin + (double)i * deltaX;
            double r = refFunc(x);
            results.vals[i] = FixedPointMath::toFixedP<int16_t>(r, Dp);
        }

        // Fill in out of bounds cases, these assume the functions have limits as they approach infinity
        if (HasLimits)
        {
            constexpr double limitEval = 15.0;
            constexpr double lowerR = refFunc(XMin - limitEval);
            constexpr double upperR = refFunc(XMax + limitEval);
            
            results.vals[Entries] = FixedPointMath::toFixedP<int16_t>(lowerR, Dp);
            results.vals[Entries + 1] = FixedPointMath::toFixedP<int16_t>(upperR, Dp);
        }

        return results;
    }
    
    // Attempted refactor, actually cant... sad
    int16_t lookup(int16_t x) const
    {
        int32_t nx = (int32_t)x;
        
        if constexpr (HasLimits)
        {
            if (nx > fixedXMax) return entries.vals[Entries + 1]; // Upper out of bounds
            else if (nx < fixedXMin) return entries.vals[Entries]; // Lower out of bounds
        }

        else
        {
            if (nx > fixedXMax) return entries.vals[Entries - 1]; // Upper out of bounds
            else if (nx < fixedXMin) return entries.vals[0]; // Lower out of bounds
        }

        constexpr int32_t logEntries = (int32_t)ConstexprMath::log2(Entries - 1);
        constexpr int32_t logRange = (int32_t)ConstexprMath::log2((XMax - XMin));

        constexpr int32_t idxPower = logRange - logEntries;
        constexpr int32_t absIdxPower = idxPower > 0 ? idxPower : -idxPower;

        int32_t xOffset = nx - fixedXMin;
        int32_t idx = ((idxPower > 0) ? xOffset >> absIdxPower : xOffset << absIdxPower) >> Dp;

        if (idx == Entries - 1) idx--;

        // Lerp
        int32_t y1 = (int32_t)entries.vals[idx], y2 = (int32_t)entries.vals[idx + 1];

        // Calculate x1
        int32_t fixedIdx = idx << Dp;
        int32_t offset = (idxPower > 0) ? fixedIdx << absIdxPower : fixedIdx >> absIdxPower; // idx * deltaX
        int32_t x1 = fixedXMin + offset;

        // Calculate t
        int32_t distToX1 = nx - x1;
        int32_t t = (idxPower > 0) ? distToX1 >> absIdxPower : distToX1 << absIdxPower; // distToX1 / fixedDeltaX

        // Final interpolation
        constexpr int32_t oneShift = 1 << Dp;
        int32_t yLerp = ((oneShift - t) * y1 + t * y2) >> Dp; // Shift needed because QN * QN = Q2N

        return (int16_t)yLerp;
    }
};