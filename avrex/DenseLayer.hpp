#pragma once

#include <stdint.h>

#include "Numerics.hpp"
#include "Tensor.hpp"
#include "Network.hpp"

#include <Arduino.h>

/*
Input:

Batch->
Input    [a ... b]
|        [.  .  .]
\/       [c ... d]

Weights:
PrevLayer->
CurLayer    [a ... b]
|           [.  .  .]
\/          [c ... d]

*/

/* 
Since its not critical but alot of the dense layer and Dp shifting stuff can be done at compile time, just note this for later, theres not enough time
I only really think marginal clock cycles can be saved and minor branching reduced, not worth the risk/refactor
Just note to like... do it at some point
*/

struct DenseLayer
{
    const FlashTensor<int16_t, 1> biases; // Vector
    const FlashTensor<TernaryPack, 2> weights; // Matrix
    void (*const alphaScale)(int16_t*, int16_t*, uint16_t);
    const uint8_t dp;
    const bool useBias;

    DenseLayer(uint16_t layerSize, uint16_t prevLayerSize, const int16_t* biasStart, const TernaryPack* weightStart, void (*alphaScalePtr)(int16_t*, int16_t*, uint16_t), uint8_t decimalPoint, bool activateBias)
        : biases({ layerSize }, biasStart)
        , weights({ layerSize, prevLayerSize }, weightStart)
        , alphaScale(alphaScalePtr)
        , dp(decimalPoint)
        , useBias(activateBias)
    {}
    
    template <typename NetworkT>
    void compute(int16_t* bufferA, int16_t* bufferB, NetworkT* network) const
    {
        // buffers need to be switched (read, write)
        weightMul(weights, bufferA, bufferB);
        alphaScale(bufferB, bufferA, getLayerSize());

        if (useBias) // Can be made constexpr
        {
            biasAdd(biases, bufferA, bufferB);
            network->cycleBuffer();
        }
    } 
    
    // size is redundant here
    inline uint16_t getLayerSize() const 
    { 
        return weights.features[0]; 
    }
};