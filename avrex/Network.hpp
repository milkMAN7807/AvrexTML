#pragma once

#include <stdint.h>

#include "DataStructures.hpp"
#include "Tensor.hpp"

template <size_t BufferSize, typename... LayerTs>
struct Network
{
    int16_t buffer2[BufferSize];
    int16_t* bufferA = nullptr; 
    int16_t* bufferB = buffer2;
    static constexpr uint16_t bufferSize = BufferSize;

    Metalist<LayerTs...> layers; 

    private:
        uint8_t inputDp;
        uint8_t lastDp;
    public:

    Network(LayerTs... elements, uint8_t inputDecimalPoint) 
        : layers(elements...)
        , inputDp(inputDecimalPoint)
        , lastDp(inputDecimalPoint)
    {}

    void forward(int16_t* data, uint16_t dataSize)
    {
        //for (uint16_t i = 0; i < dataSize; i++) bufferA[i] = data[i];
        bufferA = data;
        feedForward(layers);
    }

    void cycleBuffer()
    {
        int16_t* t = bufferA;
        bufferA = bufferB;
        bufferB = t;
    }

    private:
        template <typename H, typename... Ts>
        void feedForward(Metalist<H, Ts...>& list) // Fix to run over layer size and not network buffer size
        {
            // Adjust buffer for the right Q format, EXPENSIVE! god I hate this
            // Hopefully codegen is optimzed to assign assembly favored Q formats to each layer
            int8_t delta = list.val.dp - lastDp;
            
            if (delta > 0)
            {
                // Needs to be saturated as to not obliterate the network
                // Branchless version has variable bit shifts that cant be optimized by avr-gcc
                for (uint16_t i = 0; i < BufferSize; i++) 
                {                    
                    bufferA[i] = FixedPointMath::satSignedLeftShift(bufferA[i], delta);
                }
            }

            else if (delta < 0)
            {
                delta = -delta;
                for (uint16_t i = 0; i < BufferSize; i++) bufferA[i] >>= delta;
            }
            
            list.val.compute(bufferA, bufferB, this);
            lastDp = list.val.dp;

            feedForward(list.tail);
        }

        void feedForward(Metalist<>&) 
        {
            lastDp = inputDp; // Reset network
        }
};