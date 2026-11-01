#pragma once

// Compiletime array of objects...
template <typename... Ts>
struct Metalist {};

template <typename H, typename... Ts>
struct Metalist<H, Ts...>
{
    H val;
    Metalist<Ts...> tail;

    Metalist(H element, Ts... elements) 
        : val(element) 
        , tail(elements...) 
    {}
};