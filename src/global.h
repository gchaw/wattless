/*
 * File: global.h
 * Author: 
 * Created: November 24 2013
 * 
 * global defines and function prototypes
 */

#ifndef GLOBAL_H
#define	GLOBAL_H

#define BLOCK_SIZE 4    // Nb Block size (as per standard)
#define KEY_SIZE 4	// Nk Key size (AES-128)
#define NUM_ROUNDS 10   // Nr Number of rounds (AES-128)

void KeyExpansion(unsigned char* key, unsigned int* w);

#endif /* GLOBAL_H */
