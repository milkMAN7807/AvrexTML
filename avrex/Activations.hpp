#pragma once

#include <stdint.h>

#include "LookUpTable.hpp"
#include "LookUpTableNumerics.hpp"
#include "Network.hpp"

namespace ActivationLib
{
    // Complicated LUT functions, hopefully these only get included by the compiler if explicitly used by the model
    // look into conditional instantiation or whatever
    // Also DP should be max of range and domain, because inputs get scaled to DP before use
    static constexpr LUT<9, -4, 4, 12, true, &HighDefFuncLib::hdSigmoid> sigmoidLUT;
    static constexpr LUT<9, -2, 2, 13, true, &HighDefFuncLib::hdTanh> tanhLUT;
    
    // Naive functions, all of these process vectors
    inline void relu(int16_t* bufferA, int16_t* bufferB, uint16_t size)
    {
        for (uint16_t i = 0; i < size; i++) bufferB[i] = bufferA[i] > 0 ? bufferA[i] : 0;
    }

    inline void sigmoid(int16_t* bufferA, int16_t* bufferB, uint16_t size)
    {
        for (uint16_t i = 0; i < size; i++) bufferB[i] = sigmoidLUT.lookup(bufferA[i]);
    }
    
    inline void tanh(int16_t* bufferA, int16_t* bufferB, uint16_t size)
    {
        for (uint16_t i = 0; i < size; i++) bufferB[i] = tanhLUT.lookup(bufferA[i]);
    }

    // Use with caution, very expensive
    void softmax(int16_t* bufferA, int16_t* bufferB, uint16_t size)
    {
        int16_t maxLogit = bufferA[0];
        for (uint16_t i = 1; i < size; i++) maxLogit = bufferA[i] > maxLogit ? bufferA[i] : maxLogit;

        uint16_t sum = 0;

        for (uint16_t i = 0; i < size; i++) 
        {
            uint16_t expI = FixedPointMath::exp(bufferA[i] - maxLogit);
            sum += expI;
            bufferA[i] = expI;
        }
        
        for (uint16_t i = 0; i < size; i++)
        {
            uint32_t adjBufferA = (uint32_t)bufferA[i] << FixedPointMath::expLUT.dp;
            bufferB[i] = adjBufferA / (uint32_t)sum;
        }
    }       
}  

struct ActivationFunc
{    
    void (*activationFunc)(int16_t*, int16_t*, uint16_t);
    const uint16_t layerSize;
    const uint8_t dp;

    ActivationFunc(void (*activationFuncPtr)(int16_t*, int16_t*, uint16_t), uint16_t inputLayerSize, uint8_t decimalPoint)
        : activationFunc(activationFuncPtr)
        , layerSize(inputLayerSize)
        , dp(decimalPoint)
    {}

    template <typename NetworkT>
    void compute(int16_t* bufferA, int16_t* bufferB, NetworkT* network) const // General computation
    {
        activationFunc(bufferA, bufferB, layerSize);
        network->cycleBuffer();
    }
};