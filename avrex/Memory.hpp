#pragma once

#if defined(__AVR__) && defined(__GNUC__)
    #include <avr/pgmspace.h>
    #define TRY_FLASH PROGMEM

    #define TRY_FREAD_INT8(x) pgm_read_byte((const int8_t*)&(x))
    #define TRY_FREAD_INT16(x) pgm_read_word((const int16_t*)&(x))

    #define TRY_FREAD_PTR_INT8(x) pgm_read_byte(x)
    #define TRY_FREAD_PTR_INT16(x) pgm_read_word(x)

    #define TRY_FREAD_ARR_INT8(x, i) pgm_read_byte((const int8_t*)(x) + (i))
    #define TRY_FREAD_ARR_INT16(x, i) pgm_read_word((const int16_t*)(x) + (i))
#else
    #define TRY_FLASH

    #define TRY_FREAD_INT8(x) (x)
    #define TRY_FREAD_INT16(x) (x)

    #define TRY_FREAD_PTR_INT8(x) (*x)
    #define TRY_FREAD_PTR_INT16(x) (*x)

    #define TRY_FREAD_ARR_INT8(x, i) ((x)[(i)])
    #define TRY_FREAD_ARR_INT16(x, i) ((x)[(i)])
#endif