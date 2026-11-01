#include <Arduino.h>
//#include <fix_fft/fix_fft.h>
//#include "MnistModel.hpp"
//#include "SmallMnistModel.hpp"
//#include "BetterSmallMnist.hpp"
//#include "BestMnistModel.hpp"
#include "MnistLargeModel.hpp"

void setup() 
{
    Serial.begin(500'000);
    while(Serial.available() > 0) Serial.read();
}

const int16_t imageByteCount = 392;
const int16_t imagePixelCount = 196;
int16_t currentByte = 0;
int16_t currentByteCount = 0;

int16_t dataBuffer[imagePixelCount];

void loop() 
{
    if (Serial.available() > 0)
    {
        // Read byte
        int16_t bVal = (int16_t)Serial.read();
        
        if (currentByteCount % 2 == 0) 
        {
            currentByte = bVal;
        }

        else 
        {
            currentByte += (bVal << 8); // Construct byte
            dataBuffer[currentByteCount >> 1] = currentByte;
            currentByte = 0;
        }

        currentByteCount++;

        if (currentByteCount >= imageByteCount)
        {
            uint32_t tStart = micros();
            MnistLargeModel::network.forward(dataBuffer, imagePixelCount);
            uint32_t tEnd = micros();

            int16_t maxVal = -1, maxIndex = 0; 
            int32_t computeTime = 0;

            for (uint16_t i = 0; i < 10; i++)
            {
                if (dataBuffer[i] > maxVal) 
                {
                    maxVal = dataBuffer[i];
                    maxIndex = i;
                }
            }
    
            computeTime = tEnd - tStart;
            Serial.write((int8_t)maxIndex);
            Serial.write((uint8_t)(computeTime));
            Serial.write((uint8_t)(computeTime >> 8));
            Serial.write((uint8_t)(computeTime >> 16));
            Serial.write((uint8_t)(computeTime >> 24));

            currentByteCount = 0;
        }
    }
}