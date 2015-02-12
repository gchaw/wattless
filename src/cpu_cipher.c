/* 
 * File:   cpu_cipher.c
 * Author: chawgary, syrowikb
 *
 * Created on December 3 2013
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "global.h"
#include "cpu_cipher.h"

void print_state(unsigned char* state) {
	int i, j;
	for(i = 0; i < 4; i++) { // column
		for(j = 0; j < BLOCK_SIZE; j++) { // row
			printf("%02x ", state[j * 4 + i]);
		}
		printf("\n");
	}
	printf("\n");
}

void cpu_print_block(unsigned char* block) {
	int i;
	for(i = 0; i < 16; i++) {
		printf("%02x", block[i]);
	}
	printf("\n");
}


void cpu_increment_counter(unsigned char * counter) {
        // increment the counter
        int i;
        for(i = 15; i != 0; i--) {
                counter[i]++;
                if(counter[i] != 0) break;
        }
}


void ctr_cpu_encrypt(unsigned int *key, unsigned char *counter_init, unsigned char *in, unsigned int size_bytes) {
	unsigned int block_count = size_bytes >> 4; // divide by 16
	unsigned int i;
	unsigned char counter[16];
	unsigned char temp[16];
	memcpy(counter, counter_init, 16); // initialize the counter
	for(i = 0; i < block_count; i++) {
		memcpy(temp, counter, 16);
		cpu_encrypt_block(temp, key);
		int j;
		for(j = 0; j < 16; j++) {
			in[i * 16 + j] = in[i * 16 + j] ^ temp[j];
		}
		cpu_increment_counter(counter);
	}
}

void cpu_encrypt(unsigned char *in, unsigned int *key, unsigned int block_count) {
	printf("cpu_encrypt()\n");
	// call cpu_encrypt_block() for each block
	int i; 
	for(i = 0; i < block_count; i++) {
		if(i < 25) {
			cpu_print_block(&in[i * 16]);
		}
		cpu_encrypt_block(&in[i * 16], key);
	}
}

void cpu_decrypt(unsigned char *in, unsigned int *key, unsigned int block_count) {
	// call cpu_encrypt_block() for each block
	int i;
	for(i = 0; i < block_count; i++) {
		cpu_decrypt_block(&in[i * 16], key);
	}
}

void cpu_encrypt_block(unsigned char *in_out, unsigned int *w) {
	unsigned char * state = (unsigned char*)malloc(4 * BLOCK_SIZE * sizeof(char));

	memcpy(state, in_out, 4 * BLOCK_SIZE);

	AddRoundKey(state, &w[0]);

	int round;
	for(round = 1; round < NUM_ROUNDS; round++) {
		SubBytes(state);
		ShiftRows(state);
		MixColumns(state);
		AddRoundKey(state, &w[round*BLOCK_SIZE]);
	}

	SubBytes(state);
	ShiftRows(state);
	AddRoundKey(state, &w[NUM_ROUNDS*BLOCK_SIZE]);

	//printf("output\n");
	//print_state(state);

	memcpy(in_out, state, 4 * BLOCK_SIZE);

	free(state);

	return;
}

void cpu_decrypt_block(unsigned char *in_out, unsigned int *w) {
	unsigned char * state = (unsigned char*)malloc(4 * BLOCK_SIZE * sizeof(char));
	memcpy(state, in_out, 4 * BLOCK_SIZE);

	AddRoundKey(state, &w[NUM_ROUNDS * BLOCK_SIZE]);

	int round;
	for(round = NUM_ROUNDS - 1; round > 0; round--) {
		InvShiftRows(state);
		InvSubBytes(state);
		AddRoundKey(state, &w[round * BLOCK_SIZE]);
		InvMixColumns(state);
	}

	InvShiftRows(state);
	InvSubBytes(state);
	AddRoundKey(state, &w[0]);

	memcpy(in_out, state, 4 * BLOCK_SIZE);
	free(state);
	return;
}

void SubBytes(unsigned char *state) //state = 16 chars
{
	int i;
	for(i = 0; i < 4 * BLOCK_SIZE; i++) {
		state[i] = sbox[state[i]];
	}
    
}

void InvSubBytes(unsigned char *state) //state = 16 chars
{
	int i;
	for(i = 0; i < 4 * BLOCK_SIZE; i++) {
		state[i] = inv_s[state[i]];
	}
    
}
void ShiftRows(unsigned char *state) 
{
    // NOTE: For whatever reason the standard uses column-major ordering ?
    // 0 1 2 3 --> 0 1 2 3  | 0  4  8  12 --> 0   4  8 12
    // 0 1 2 3 --> 1 2 3 0  | 1  5  9  13 --> 5   9 13  1
    // 0 1 2 3 --> 2 3 0 1  | 2  6  10 14 --> 10 14  2  6
    // 0 1 2 3 --> 3 0 1 2  | 3  7  11 15 --> 15  3  7 11
	unsigned char temp = state[1];

	state[1] = state[5];
	state[5] = state[9];
	state[9] = state[13];
	state[13] = temp;

	temp = state[2];
	state[2] = state[10];
	state[10] = temp;
	temp = state[6];
	state[6] = state[14];
	state[14] = temp;
	
	temp = state[3];
	state[3] = state[15];
	state[15] = state[11];
	state[11] = state[7];
	state[7] = temp;
}

void InvShiftRows(unsigned char *state) {
	unsigned char temp = state[13];
	state[13] = state[9];
	state[9] = state[5];
	state[5] = state[1];
	state[1] = temp;

	temp = state[2];
	state[2] = state[10];
	state[10] = temp;
	temp = state[6];
	state[6] = state[14];
	state[14] = temp;

	temp = state[3];
	state[3] = state[7];
	state[7] = state[11];
	state[11] = state[15];
	state[15] = temp;
}

void AddRoundKey(unsigned char *state, unsigned int *w) {
	int i;
	for(i = 0; i < BLOCK_SIZE; i++) { // column
		state[i * 4 + 0] = state[i * 4 + 0] ^ ((w[i] >> (8 * 3)) & 0xFF);
		state[i * 4 + 1] = state[i * 4 + 1] ^ ((w[i] >> (8 * 2)) & 0xFF);
		state[i * 4 + 2] = state[i * 4 + 2] ^ ((w[i] >> (8 * 1)) & 0xFF);
		state[i * 4 + 3] = state[i * 4 + 3] ^ ((w[i] >> (8 * 0)) & 0xFF);
	}
}

// See "Efficient Software Implementation of AES on 32-bit platforms"
void MixColumns(unsigned char *state) {
	unsigned char * s = (unsigned char *)malloc(4 * BLOCK_SIZE * sizeof(char));
	memcpy(s, state, 4 * BLOCK_SIZE);
	int i; 
	for(i = 0; i < BLOCK_SIZE; i++) { // column
		unsigned char * x = (unsigned char*)&s[i*4];
		unsigned char * y = (unsigned char*)&state[i*4];
		y[0] = x[1] ^ x[2] ^ x[3];
		y[1] = x[0] ^ x[2] ^ x[3];
		y[2] = x[0] ^ x[1] ^ x[3];
		y[3] = x[0] ^ x[1] ^ x[2];
		x[0] = GF_2[x[0]];
		x[1] = GF_2[x[1]];
		x[2] = GF_2[x[2]];
		x[3] = GF_2[x[3]];
		y[0] ^= x[0] ^ x[1];
		y[1] ^= x[1] ^ x[2];
		y[2] ^= x[2] ^ x[3];
		y[3] ^= x[3] ^ x[0];
	}
	free(s);
}

void InvMixColumns(unsigned char *state) {
	// this is a little more complicated
	unsigned char * s = (unsigned char *)malloc(4 * BLOCK_SIZE * sizeof(char));
	memcpy(s, state, 4 * BLOCK_SIZE);
	int i; 
	for(i = 0; i < BLOCK_SIZE; i++) { // column
		unsigned char * x = (unsigned char*)&s[i*4];
		unsigned char * y = (unsigned char*)&state[i*4];
		y[0] = x[1] ^ x[2] ^ x[3];
		y[1] = x[0] ^ x[2] ^ x[3];
		y[2] = x[0] ^ x[1] ^ x[3];
		y[3] = x[0] ^ x[1] ^ x[2];
		x[0] = GF_2[x[0]];
		x[1] = GF_2[x[1]];
		x[2] = GF_2[x[2]];
		x[3] = GF_2[x[3]];
		y[0] ^= x[0] ^ x[1];
		y[1] ^= x[1] ^ x[2];
		y[2] ^= x[2] ^ x[3];
		y[3] ^= x[3] ^ x[0];
		x[0] = GF_2[x[0]];
		x[1] = GF_2[x[1]];
		x[2] = GF_2[x[2]];
		x[3] = GF_2[x[3]];
		y[0] ^= x[0] ^ x[2];
		y[1] ^= x[1] ^ x[3];
		y[2] ^= x[2] ^ x[0];
		y[3] ^= x[3] ^ x[1];
		x[0] = GF_2[x[0]];
		x[1] = GF_2[x[1]];
		x[2] = GF_2[x[2]];
		x[3] = GF_2[x[3]];
		y[0] ^= x[0] ^ x[1] ^ x[2] ^ x[3];
		y[1] ^= x[0] ^ x[1] ^ x[2] ^ x[3];
		y[2] ^= x[0] ^ x[1] ^ x[2] ^ x[3];
		y[3] ^= x[0] ^ x[1] ^ x[2] ^ x[3];

	}
	free(s);
}






