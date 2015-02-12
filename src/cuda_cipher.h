/* 
 * File:   opencl_cipher.h
 * Author: syrowikb
 *
 * Created on December 14, 2013
 */

#ifndef CUDA_CIPHER_H
#define	CUDA_CIPHER_H


#include "sbox.h"
#include "global.h"
#include "gf_tables.h"


#ifdef __cplusplus
extern "C"
#endif
int cuda_main( unsigned int* expanded_key, unsigned char initial_counter[16], unsigned char* text, unsigned int size_bytes );


#endif	/* CUDA_CIPHER_H */
