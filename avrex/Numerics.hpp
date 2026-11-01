#pragma once

#include <stdint.h>
#include <stddef.h>

/*
Structure to hold 4 ternary weight values in a single byte
using [] accesses it similar to an array

Packing Scheme (This is more efficient than twos compliment, sorry!)
00 -> -1
01 -> 0
10 -> 1
*/
struct TernaryPack
{
    uint8_t byte;
    
    int8_t operator[](uint8_t idx) const
    {
        const uint8_t dist = idx << 1; // Mult by 2
        uint8_t num = byte >> dist;
        num &= 0b11;

        return (int8_t)num - 1;
    }
};

namespace FixedPointMath
{
    template <typename I>
    constexpr I toFixedP(double n, size_t dp) // Meant to run for LUT computations
    {
        double r = n * (1 << dp); // multiply r by 2^fractional bits
        return (I)(r + (r > 0 ? 0.5 : -0.5)); // round
    }

    template <typename F, typename I>
    constexpr F toFloat(I n, size_t dp)
    {
        return n / (F)(1 << dp);
    }

    inline int16_t satCastInt16(int32_t v)
    {
        if (v > INT16_MAX) return INT16_MAX;
        else if (v < INT16_MIN) return INT16_MIN;
        else return (int16_t)v;
    }

    inline int16_t satSignedLeftShift(int16_t v, int8_t delta) // Too much work to operator overload
    {
        constexpr int32_t adjInt16Max = (int32_t)INT16_MAX;
        constexpr int32_t adjInt16tMin = (int32_t)INT16_MIN;
        
        int32_t t = (int32_t)(v) << delta;
        int16_t satVal = 0;

        if (t > adjInt16Max) satVal = INT16_MAX;
        else if (t < adjInt16tMin) satVal = INT16_MIN;
        else satVal = (int16_t)t;
        
        return satVal;
    }
}

namespace ConstexprMath
{
    constexpr double pow(double x, int y)
    {
        double p = 1.0;
        for (size_t i = 0; i < (size_t)y; i++) p *= x;
        return p;
    }

    constexpr double exp(double x)
    {
        double reducedArg = x > 0.0 ? x : -x; // Take absolute val
        int k = 0;

        // Arg must be reduced to make sure maclurin series accuracy is retained
        const double argThreshold = 0.5;
        while (reducedArg >= argThreshold)
        {
            reducedArg /= 2;
            k++;
        }

        reducedArg = x > 0 ? reducedArg : -reducedArg;
        
        // 9 terms should be enough for this e^x Maclurin series
        double reducedOut = 
        1.0 
        + reducedArg 
        + pow(reducedArg, 2) / 2.0
        + pow(reducedArg, 3) / 6.0
        + pow(reducedArg, 4) / 24.0
        + pow(reducedArg, 5) / 120.0
        + pow(reducedArg, 6) / 720.0
        + pow(reducedArg, 7) / 5'040.0
        + pow(reducedArg, 8) / 40'320.0
        + pow(reducedArg, 9) / 362'880.0;
        // sum(k = 0 -> n)(x^k/k!)

        double fullOut = pow(reducedOut, (int)pow(2.0, k));
        return fullOut;
    }

    // Fetch exponent of power of 2 (0->11)
    constexpr int log2(int x)
    {
        if (x % 2 != 0) return -1;
        int i = 0;

        while (x > 0)
        {
            x >>= 1;
            i++;
        }

        return i - 1;
    }

    // Do this for fixed points
    constexpr int log2(int x, int dp)
    {
        if (x % 2 != 0) return -1;
        int i = 0;

        while (x > 0)
        {
            x >>= 1;
            i++;
        }

        return i - 1 - dp;
    }
}