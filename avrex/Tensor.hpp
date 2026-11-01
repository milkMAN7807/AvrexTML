#pragma once

#include <stdint.h>
#include <stddef.h>

#include "Numerics.hpp"
#include "Memory.hpp"

// Forward Dec
template <typename T, size_t MaxRank, size_t CurRank>
struct TensorProxy;

template <typename T, size_t MaxRank, size_t CurRank>
struct ConstTensorProxy;

// Actual Declarations
template <typename T, size_t Rank>
struct Tensor
{
    static constexpr uint8_t rank = Rank;
    uint16_t features[Rank]; // Gives the number of elements of some portion of the array, this goes out -> in
    uint16_t strides[Rank - 1]; // Last one is guaranteed to store 1 so not much point in wasting 2 bytes on it
    T* dataStart;

    Tensor(const uint16_t (&featureInput)[Rank], T* dataStart)
    {
        this->dataStart = dataStart;
        
        // Input dims
        for (uint8_t i = 0; i < Rank; i++) features[i] = featureInput[i];

        // Compute strides
        strides[Rank - 2] = features[Rank - 1];
        for (uint8_t i = Rank - 2; i > 0; i--) strides[i - 1] = features[i] * strides[i];
    }

    // Accessors
    TensorProxy<T, Rank, Rank - 1> operator[](uint16_t idx)
    {
        TensorProxy<T, Rank, Rank - 1> proxy;
        proxy.valuePtr = dataStart + strides[0] * idx;
        proxy.stridesPtr = strides;

        return proxy;
    }

    ConstTensorProxy<T, Rank, Rank - 1> operator[](uint16_t idx) const
    {
        ConstTensorProxy<T, Rank, Rank - 1> proxy;
        proxy.valuePtr = dataStart + strides[0] * idx;
        proxy.stridesPtr = strides;

        return proxy;
    }
};

// Vector tensor, redefinition is needed because main tensor access operator returns tensor proxy, simplifies nicely!
template <typename T>
struct Tensor<T, 1>
{
    static constexpr uint8_t rank = 1;
    uint16_t features[1];
    T* dataStart;

    Tensor(const uint16_t (&featureInput)[1], T* dataStart)
    {
        this->dataStart = dataStart;
        
        // Input dims
        features[0] = featureInput[0];
    }
    
    // Accessors
    T& operator[](uint16_t idx)
    {
        return *(dataStart + idx);
    }

    const T& operator[](uint16_t idx) const
    {
        return *(dataStart + idx);
    }
};

// Special packing case
template <size_t Rank>
struct Tensor<TernaryPack, Rank>
{
    static constexpr uint8_t rank = Rank;
    uint16_t features[Rank];
    uint16_t strides[Rank - 1];
    const TernaryPack* dataStart;

    Tensor(const uint16_t (&featureInput)[Rank], TernaryPack* dataStart)
    {
        this->dataStart = dataStart;
        
        // Input dims
        for (uint8_t i = 0; i < Rank; i++) features[i] = featureInput[i];

        // Compute strides
        strides[Rank - 2] = features[Rank - 1];
        for (uint8_t i = Rank - 2; i > 0; i--) strides[i - 1] = features[i] * strides[i];
    }

    TensorProxy<TernaryPack, Rank, Rank - 1> operator[](uint16_t idx) const
    {
        TensorProxy<TernaryPack, Rank, Rank - 1> proxy;
        proxy.offset = strides[0] * idx;
        proxy.valuePtr = dataStart;
        proxy.stridesPtr = strides;

        return proxy;
    }
};

template <>
struct Tensor<TernaryPack, 1>
{
    static constexpr uint8_t rank = 1;
    uint16_t features[1];
    const TernaryPack* dataStart;

    Tensor(const uint16_t (&featureInput)[1], TernaryPack* dataStart)
    {
        this->dataStart = dataStart;
        
        // Input dims
        features[0] = featureInput[0];
    }

    int8_t operator[](uint16_t idx) const
    {
        uint16_t byteIdx = idx >> 2;
        uint8_t packIdx = idx & 0b11;

        return dataStart[byteIdx][packIdx];
    }
};

template <typename T, size_t MaxRank, size_t CurRank>
struct TensorProxy
{
    const uint16_t* stridesPtr;
    T* valuePtr;

    TensorProxy<T, MaxRank, CurRank - 1> operator[](uint16_t idx) const
    { 
        TensorProxy<T, MaxRank, CurRank - 1> proxy;
        proxy.valuePtr = valuePtr + stridesPtr[MaxRank - CurRank] * idx; // Compute offset
        proxy.stridesPtr = stridesPtr;

        return proxy;
    }
};

template <typename T, size_t MaxRank>
struct TensorProxy<T, MaxRank, 1>
{
    const uint16_t* stridesPtr;
    T* valuePtr;

    T& operator[](uint16_t idx) const
    {
        return *(valuePtr + idx);
    }
};

template <size_t MaxRank, size_t CurRank>
struct TensorProxy<TernaryPack, MaxRank, CurRank>
{
    const uint16_t* stridesPtr;
    const TernaryPack* valuePtr;
    uint16_t offset;

    TensorProxy<TernaryPack, MaxRank, CurRank - 1> operator[](uint16_t idx) const
    { 
        TensorProxy<TernaryPack, MaxRank, CurRank - 1> proxy;
        proxy.offset = offset + stridesPtr[MaxRank - CurRank] * idx; // Compute offset
        proxy.valuePtr = valuePtr;
        proxy.stridesPtr = stridesPtr;

        return proxy;
    }
};

template <size_t MaxRank>
struct TensorProxy<TernaryPack, MaxRank, 1>
{
    const uint16_t* stridesPtr;
    uint16_t offset;
    const TernaryPack* valuePtr;

    int8_t operator[](uint16_t idx) const
    {
        uint16_t byteIdx = (offset + idx) >> 2; // Division by 4
        uint8_t packIdx = (offset + idx) & 0b11; // Mod 4

        return valuePtr[byteIdx][packIdx];
    }
};

template <typename T, size_t MaxRank, size_t CurRank>
struct ConstTensorProxy
{
    const uint16_t* stridesPtr;
    const T* valuePtr;

    ConstTensorProxy<T, MaxRank, CurRank - 1> operator[](uint16_t idx) const
    { 
        ConstTensorProxy<T, MaxRank, CurRank - 1> proxy;
        proxy.valuePtr = valuePtr + stridesPtr[MaxRank - CurRank] * idx; // Compute offset
        proxy.stridesPtr = stridesPtr;

        return proxy;
    }
};

template <typename T, size_t MaxRank>
struct ConstTensorProxy<T, MaxRank, 1>
{
    const uint16_t* stridesPtr;
    const T* valuePtr;

    const T& operator[](uint16_t idx) const
    {
        return *(valuePtr + idx);
    }
};

// FLASH TENSOR STUFF
template <typename T>
struct FlashTensorTypeValid
{
    static constexpr bool val = false;
};

template <>
struct FlashTensorTypeValid<int8_t>
{
    static constexpr bool val = true;
};

template <>
struct FlashTensorTypeValid<uint8_t>
{
    static constexpr bool val = true;
};

template <>
struct FlashTensorTypeValid<int16_t>
{
    static constexpr bool val = true;
};

template <>
struct FlashTensorTypeValid<uint16_t>
{
    static constexpr bool val = true;
};

template <>
struct FlashTensorTypeValid<TernaryPack>
{
    static constexpr bool val = true;
};

// Forward dec
template <typename T, size_t MaxRank, size_t CurRank>
struct FlashTensorProxy;

template <typename T, size_t Rank>
struct FlashTensor
{
    static_assert(FlashTensorTypeValid<T>::val, "Flash Tensor invalid: supported types are int8_t/uint8_t and int16_t/uint16_t");
    
    static constexpr uint8_t rank = Rank;
    uint16_t features[Rank]; // Gives the number of elements of some portion of the array, this goes out -> in
    uint16_t strides[Rank - 1]; // Last one is guaranteed to store 1 so not much point in wasting 2 bytes on it
    const T* dataStart;

    FlashTensor(const uint16_t (&featureInput)[Rank], const T* dataStart)
    {
        this->dataStart = dataStart;
        
        // Input dims
        for (uint8_t i = 0; i < Rank; i++) features[i] = featureInput[i];

        // Compute strides
        strides[Rank - 2] = features[Rank - 1];
        for (uint8_t i = Rank - 2; i > 0; i--) strides[i - 1] = features[i] * strides[i];
    }

    FlashTensorProxy<T, Rank, Rank - 1> operator[](uint16_t idx) const
    {
        FlashTensorProxy<T, Rank, Rank - 1> proxy;
        proxy.valuePtr = dataStart + strides[0] * idx;
        proxy.stridesPtr = strides;

        return proxy;
    }
};

template <>
struct FlashTensor<int8_t, 1>
{
    static constexpr uint8_t rank = 1;
    uint16_t features[1];
    const int8_t* dataStart;

    FlashTensor(const uint16_t (&featureInput)[1], const int8_t* dataStart)
    {
        this->dataStart = dataStart;
        
        // Input dims
        features[0] = featureInput[0];
    }

    int8_t operator[](uint16_t idx) const
    {
        return TRY_FREAD_PTR_INT8((dataStart + idx));
    }
};

template <>
struct FlashTensor<uint8_t, 1>
{
    static constexpr uint8_t rank = 1;
    uint16_t features[1];
    const uint8_t* dataStart;

    FlashTensor(const uint16_t (&featureInput)[1], const uint8_t* dataStart)
    {
        this->dataStart = dataStart;
        
        // Input dims
        features[0] = featureInput[0];
    }

    uint8_t operator[](uint16_t idx) const
    {
        return TRY_FREAD_PTR_INT8((dataStart + idx));
    }
};

template <>
struct FlashTensor<int16_t, 1>
{
    static constexpr uint8_t rank = 1;
    uint16_t features[1];
    const int16_t* dataStart;

    FlashTensor(const uint16_t (&featureInput)[1], const int16_t* dataStart)
    {
        this->dataStart = dataStart;
        
        // Input dims
        features[0] = featureInput[0];
    }

    int16_t operator[](uint16_t idx) const
    {
        return TRY_FREAD_PTR_INT16((dataStart + idx));
    }
};

template <>
struct FlashTensor<uint16_t, 1>
{
    static constexpr uint8_t rank = 1;
    uint16_t features[1];
    const uint16_t* dataStart;

    FlashTensor(const uint16_t (&featureInput)[1], const uint16_t* dataStart)
    {
        this->dataStart = dataStart;
        
        // Input dims
        features[0] = featureInput[0];
    }

    uint16_t operator[](uint16_t idx) const
    {
        return TRY_FREAD_PTR_INT16((dataStart + idx));
    }
};

template <>
struct FlashTensor<TernaryPack, 1>
{
    static constexpr uint8_t rank = 1;
    uint16_t features[1];
    const TernaryPack* dataStart;

    FlashTensor(const uint16_t (&featureInput)[1], const TernaryPack* dataStart)
    {
        this->dataStart = dataStart;

        // Input dims
        features[0] = featureInput[0];
    }

    int8_t operator[](uint16_t idx) const
    {
        uint16_t byteIdx = idx >> 2;
        uint8_t packIdx = idx & 0b11;

        TernaryPack byte = { TRY_FREAD_PTR_INT8((&(dataStart->byte) + byteIdx)) };

        return byte[packIdx];
    }
};

template <typename T, size_t MaxRank, size_t CurRank>
struct FlashTensorProxy
{
    const uint16_t* stridesPtr;
    const T* valuePtr;

    FlashTensorProxy<T, MaxRank, CurRank - 1> operator[](uint16_t idx) const
    { 
        FlashTensorProxy<T, MaxRank, CurRank - 1> proxy;
        proxy.valuePtr = valuePtr + stridesPtr[MaxRank - CurRank] * idx; // Compute offset
        proxy.stridesPtr = stridesPtr;

        return proxy;
    }
};

template <size_t MaxRank>
struct FlashTensorProxy<int8_t, MaxRank, 1>
{
    const uint16_t* stridesPtr;
    const int8_t* valuePtr;

    int8_t operator[](uint16_t idx) const
    {
        return TRY_FREAD_PTR_INT8((valuePtr + idx));
    }
};

template <size_t MaxRank>
struct FlashTensorProxy<uint8_t, MaxRank, 1>
{
    const uint16_t* stridesPtr;
    const uint8_t* valuePtr;

    uint8_t operator[](uint16_t idx) const
    {
        return TRY_FREAD_PTR_INT8((valuePtr + idx));
    }
};

template <size_t MaxRank>
struct FlashTensorProxy<int16_t, MaxRank, 1>
{
    const uint16_t* stridesPtr;
    const int16_t* valuePtr;

    int16_t operator[](uint16_t idx) const
    {
        return TRY_FREAD_PTR_INT16((valuePtr + idx));
    }
};

template <size_t MaxRank>
struct FlashTensorProxy<uint16_t, MaxRank, 1>
{
    const uint16_t* stridesPtr;
    const uint16_t* valuePtr;

    uint16_t operator[](uint16_t idx) const
    {
        return TRY_FREAD_PTR_INT16((valuePtr + idx));
    }
};

template <size_t MaxRank>
struct FlashTensorProxy<TernaryPack, MaxRank, 1>
{
    const uint16_t* stridesPtr;
    const TernaryPack* valuePtr;

    int8_t operator[](uint16_t idx) const
    {
        uint16_t byteIdx = idx >> 2;
        uint8_t packIdx = idx & 0b11;

        TernaryPack byte = { TRY_FREAD_PTR_INT8((&(valuePtr->byte) + byteIdx)) };

        return byte[packIdx];
    }
};