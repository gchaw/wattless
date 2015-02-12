/* 
 * File:   expand_key.c
 * Author: syrowikb
 *
 * Created on November 25 2013
 */

#include <stdio.h>
#include "global.h"
#include "sbox.h"

unsigned int SubWord(unsigned int w) {
	unsigned int i = (sbox[(w >> 24) & 0xFF] << 24) | (sbox[(w >> 16) & 0xFF] << 16);
	i |= (sbox[(w >> 8) & 0xFF] << 8) | sbox[w & 0xFF];
	return i;
}

unsigned int RotWord(unsigned int w) {
	unsigned char temp = (w >> 24) & 0xFF;
	return ((w << 8) | temp);
}

static const unsigned int Rcon[] = {
	0x00000000, 0x01000000, 0x02000000, 0x04000000,
	0x08000000, 0x10000000, 0x20000000, 0x40000000,
	0x80000000, 0x1B000000, 0x36000000,
	// more than 10 will not be used for 128 bit blocks
};

void KeyExpansion(unsigned char* key, unsigned int* w) {
	unsigned int temp;
	int i = 0;

	for(i = 0; i < KEY_SIZE; i++) {
		w[i] = (key[4*i] << 24) | (key[4*i + 1] << 16) | (key[4*i + 2] << 8) | key[4*i + 3];
	}

	for(; i < BLOCK_SIZE * (NUM_ROUNDS + 1); i++) {
		temp = w[i - 1];
		//printf("%d %08x %08x %08x %08x\n", i, temp, RotWord(temp), SubWord(RotWord(temp)), Rcon[i / KEY_SIZE]);
		if(i % KEY_SIZE == 0) {
			temp = SubWord(RotWord(temp)) ^ Rcon[i / KEY_SIZE];
		}
		w[i] = w[i - KEY_SIZE] ^ temp;
	}
}
