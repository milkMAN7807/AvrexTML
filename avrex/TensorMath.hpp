#pragma once

#include <stdint.h>

#include "Tensor.hpp"
#include "Numerics.hpp"

// These auto implement broadcasting instead of some generic broadcasting, this isnt a math lib...
void weightMul(const FlashTensor<TernaryPack, 2>& weights, const int16_t* const readBuffer, int16_t* const writeBuffer)
{   
    const uint8_t* bytePtr = (uint8_t*)weights.dataStart;
    uint8_t weightByte = TRY_FREAD_PTR_INT8(bytePtr);

    const uint16_t weightBytes = (weights.features[1] + 3) >> 2;
    
    for (uint16_t curNeuron = 0; curNeuron < weights.features[0]; curNeuron++) // Iterate over each of the nuerons
    {
        int32_t acc = 0;
        
        for (uint16_t curWeightByte = 0, curWeight = 0; curWeightByte < weightBytes; curWeightByte++, curWeight += 4) // Iterate over each of the weights (also number of neurons in prev layer)
        {
            if (weightByte & 0b10) acc += readBuffer[curWeight];
            else if (!(weightByte & 0b01)) acc -= readBuffer[curWeight];

            if (weightByte & 0b1000) acc += readBuffer[curWeight + 1];
            else if (!(weightByte & 0b0100)) acc -= readBuffer[curWeight + 1];

            if (weightByte & 0b10'0000) acc += readBuffer[curWeight + 2];
            else if (!(weightByte & 0b01'0000)) acc -= readBuffer[curWeight + 2];

            if (weightByte & 0b1000'0000) acc += readBuffer[curWeight + 3];
            else if (!(weightByte & 0b0100'0000)) acc -= readBuffer[curWeight + 3];

            bytePtr++;
            weightByte = TRY_FREAD_PTR_INT8(bytePtr);
        }

        writeBuffer[curNeuron] = FixedPointMath::satCastInt16(acc); // Saturate the cast to not break the network (I dont know how to run this without giving out clock cycles like charity work... making me mad)
    }
}

void biasAdd(const FlashTensor<int16_t, 1>& biases, const int16_t* const readBuffer, int16_t* const writeBuffer)
{
    for (uint16_t i = 0; i < biases.features[0]; i++) // No need to broadcast because batching isnt supported
    {
        writeBuffer[i] = readBuffer[i] + biases[i];
    }
}