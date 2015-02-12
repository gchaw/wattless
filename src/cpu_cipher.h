/* 
 * File:   cpu_cipher.h
 * Author: syrowikb
 *
 * Created on December 3, 2013
 */

#ifndef CPU_CIPHER_H
#define	CPU_CIPHER_H

#include "sbox.h"
#include "gf_tables.h"
#include "global.h"

void ctr_cpu_encrypt(unsigned int *key,unsigned char * counter_init, unsigned char *in, unsigned int size_bytes);

void cpu_encrypt(unsigned char *in, unsigned int *key, unsigned int block_count);
void cpu_decrypt(unsigned char *in, unsigned int *key, unsigned int block_count);

void cpu_encrypt_block(unsigned char *in_out, unsigned int *w);
void cpu_decrypt_block(unsigned char *in_out, unsigned int *w);

void SubBytes(unsigned char* state);
void ShiftRows(unsigned char* state);
void MixColumns(unsigned char* state);

void AddRoundKey(unsigned char* state, unsigned int* w);

void InvSubBytes(unsigned char* state);
void InvShiftRows(unsigned char* state);
void InvMixColumns(unsigned char* state);

#endif	/* CPU_CIPHER_H */

