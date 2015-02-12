/* 
 * File:   cuda_cipher.c
 * Author: chawgary
 *
 * Created on December 1, 2013
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include "global.h"
#include "sbox.h"
#include "gf_tables.h"
//#include "expand_key.h"


#include "cuda_cipher.h"


//Defines
#define EXPANDED_KEY_SIZE 176;//16 bytes * 11 rounds = 176 bytes
#define EXPANDED_KEY_SIZE_INT 44;//blocksize * ( numrounds + 1 )

#define NUM_THREADS 256


#define PRINT_CUDA_ERROR(err, err_msg) 	                     \
	if( err != cudaSuccess )                             \
	{                                                    \
		printf( err_msg );                           \
		printf( ". %s\n", cudaGetErrorString(err) ); \
		return -1;                                   \
	}														



//DEBUG STUFF
#define DEBUG 0
#define STATE_PRINT_DEBUG 0	
#define DEBUG_PRINT_ROUND( round ) if( DEBUG ) printf("ROUND %d\n", round );
	
#define DEBUG_PRINT_STATE( step )                                                                               \
	if( DEBUG )                                                                                             \
	{                                                                                                       \
		err = cudaMemcpy(counter_state, d_counter_state, counter_state_size, cudaMemcpyDeviceToHost);  	\
		cudaThreadSynchronize();                                                                        \
		printf( step );	printf("\t");                                                                   \
		print_state( &counter_state[STATE_PRINT_DEBUG << 4] );                                          \
		err = cudaMemcpy(d_counter_state, counter_state, counter_state_size, cudaMemcpyHostToDevice);   \
		cudaThreadSynchronize();                                                                        \
	}


//Device Globals

__device__ const unsigned char dev_sbox[256] = { SBOX };
__device__ const unsigned char g_GF_2[256] = { GF2 };
__device__ unsigned int g_expanded_key[44];
__device__ unsigned char g_counter_initial[16]; // 16bytes | 128bits



//for debugging
void print_state(unsigned char* state) {
	int i;

	printf("{");
	for(i = 0; i < 16; i++ )
	{
		printf("0x%02x, ", state[i]);
	}
	/*
	for(i = 0; i < 4; i++) { // column
		for(j = 0; j < BLOCK_SIZE; j++) { // row
			printf("%02x ", state[j * 4 + i]);
		}
		printf("\n");
	}
	
	*/
	printf("}\n");
}


// not much way to improve kernels unless there is a way to calculate the appropriate
// value faster than getting it from the lookup table (calculating would be more
// secure...)

//Device kernel for SubBytes step
__device__ void SubBytes(unsigned char *state) {
	state[0]  = dev_sbox[state[0]];
	state[1]  = dev_sbox[state[1]];
	state[2]  = dev_sbox[state[2]];
	state[3]  = dev_sbox[state[3]];
	state[4]  = dev_sbox[state[4]];
	state[5]  = dev_sbox[state[5]];
	state[6]  = dev_sbox[state[6]];
	state[7]  = dev_sbox[state[7]];
	state[8]  = dev_sbox[state[8]];
	state[9]  = dev_sbox[state[9]];
	state[10] = dev_sbox[state[10]];
	state[11] = dev_sbox[state[11]];
	state[12] = dev_sbox[state[12]];
	state[13] = dev_sbox[state[13]];
	state[14] = dev_sbox[state[14]];
	state[15] = dev_sbox[state[15]];
}

//Device kernel for ShiftRows step
__device__ void ShiftRows(unsigned char *state) {   
	unsigned char temp = state[1];
        
        //NOTE: column-major ordering
	// 0 1 2 3 --> 0 1 2 3  | 0  4  8  12 --> 0   4  8 12  
	// 0 1 2 3 --> 1 2 3 0  | 1  5  9  13 --> 5   9 13  1     
	// 0 1 2 3 --> 2 3 0 1  | 2  6  10 14 --> 10 14  2  6   
	// 0 1 2 3 --> 3 0 1 2  | 3  7  11 15 --> 15  3  7 11       	
	state[1]  = state[5];
	state[5]  = state[9];
	state[9]  = state[13];
	state[13] = temp;
	
	temp = state[2];
	state[2]  = state[10];
	state[10] = temp;
	temp = state[6];
	state[6]  = state[14];
	state[14] = temp;
	
	temp = state[3];
	state[3]  = state[15];
	state[15] = state[11];
	state[11] = state[7];
	state[7]  = temp;
}

//Device kernel for AddRoundKey step
__device__ void AddRoundKey(unsigned char *state, uint *w_) {
	unsigned int w = w_[0];
	state[3]  = state[3] ^ (w & 0xFF);
	w >>= 8;
	state[2]  = state[2] ^ (w & 0xFF);
	w >>= 8;
	state[1]  = state[1] ^ (w & 0xFF);
	w >>= 8;
	state[0]  = state[0] ^ (w & 0xFF);

	w = w_[1];
	state[7]  = state[7] ^ (w & 0xFF);
	w >>= 8;
	state[6]  = state[6] ^ (w & 0xFF);
	w >>= 8;
	state[5]  = state[5] ^ (w & 0xFF);
	w >>= 8;
	state[4]  = state[4] ^ (w & 0xFF);

	w = w_[2];
	state[11] = state[11] ^ (w & 0xFF);
	w >>= 8;
	state[10] = state[10] ^ (w & 0xFF);
	w >>= 8;
	state[9]  = state[9]  ^ (w & 0xFF);
	w >>= 8;
	state[8]  = state[8]  ^ (w & 0xFF);

	w = w_[3];
	state[15] = state[15] ^ (w & 0xFF);
	w >>= 8;
	state[14] = state[14] ^ (w & 0xFF);
	w >>= 8;
	state[13] = state[13] ^ (w & 0xFF);
	w >>= 8;
	state[12] = state[12] ^ (w & 0xFF);
}

//Device kernel for MixColumns step
//See "Efficient Software Implementation of AES on 32-bit platforms"
__device__ void MixColumns(unsigned char *state) {
	unsigned char x[4];
	x[0] = state[0];
	x[1] = state[1];
	x[2] = state[2];
	x[3] = state[3];
	unsigned char * y = (unsigned char *)&state[0];
	y[0] = x[1] ^ x[2] ^ x[3];
	y[1] = x[0] ^ x[2] ^ x[3];
	y[2] = x[0] ^ x[1] ^ x[3];
	y[3] = x[0] ^ x[1] ^ x[2];
	x[0] = g_GF_2[x[0]];
	x[1] = g_GF_2[x[1]];
	x[2] = g_GF_2[x[2]];
	x[3] = g_GF_2[x[3]];
	y[0] ^= x[0] ^ x[1];
	y[1] ^= x[1] ^ x[2];
	y[2] ^= x[2] ^ x[3];
	y[3] ^= x[3] ^ x[0];
	
	x[0] = state[4];
	x[1] = state[5];
	x[2] = state[6];
	x[3] = state[7];
	y = (unsigned char *)&state[4];
	y[0] = x[1] ^ x[2] ^ x[3];
	y[1] = x[0] ^ x[2] ^ x[3];
	y[2] = x[0] ^ x[1] ^ x[3];
	y[3] = x[0] ^ x[1] ^ x[2];
	x[0] = g_GF_2[x[0]];
	x[1] = g_GF_2[x[1]];
	x[2] = g_GF_2[x[2]];
	x[3] = g_GF_2[x[3]];
	y[0] ^= x[0] ^ x[1];
	y[1] ^= x[1] ^ x[2];
	y[2] ^= x[2] ^ x[3];
	y[3] ^= x[3] ^ x[0];

	x[0] = state[8];
	x[1] = state[9];
	x[2] = state[10];
	x[3] = state[11];
	y = (unsigned char *)&state[8];
	y[0] = x[1] ^ x[2] ^ x[3];
	y[1] = x[0] ^ x[2] ^ x[3];
	y[2] = x[0] ^ x[1] ^ x[3];
	y[3] = x[0] ^ x[1] ^ x[2];
	x[0] = g_GF_2[x[0]];
	x[1] = g_GF_2[x[1]];
	x[2] = g_GF_2[x[2]];
	x[3] = g_GF_2[x[3]];
	y[0] ^= x[0] ^ x[1];
	y[1] ^= x[1] ^ x[2];
	y[2] ^= x[2] ^ x[3];
	y[3] ^= x[3] ^ x[0];

	x[0] = state[12];
	x[1] = state[13];
	x[2] = state[14];
	x[3] = state[15];
	y = (unsigned char *)&state[12];
	y[0] = x[1] ^ x[2] ^ x[3];
	y[1] = x[0] ^ x[2] ^ x[3];
	y[2] = x[0] ^ x[1] ^ x[3];
	y[3] = x[0] ^ x[1] ^ x[2];
	x[0] = g_GF_2[x[0]];
	x[1] = g_GF_2[x[1]];
	x[2] = g_GF_2[x[2]];
	x[3] = g_GF_2[x[3]];
	y[0] ^= x[0] ^ x[1];
	y[1] ^= x[1] ^ x[2];
	y[2] ^= x[2] ^ x[3];
	y[3] ^= x[3] ^ x[0];
}


//Main Cuda Kernel
//Encrypts one block of data
//The thread works on the state (128bit block)
__global__ void cuda_main_kernel( unsigned int* key, unsigned char initial_counter[16], unsigned char* text, unsigned int num_blocks )
{	
	//get thread id
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int tid = y * (blockDim.x * gridDim.x ) + x;// offset

	if( tid > num_blocks ) return;

	//Initial state is the block number + initial counter
	unsigned char state[16];
	state[15] = tid & 0xFF;
	state[14] = (tid >> 8) & 0xFF;
	state[13] = (tid >> 16) & 0xFF;
	state[12] = (tid >> 24) & 0xFF;
	state[11] = 0; state[10] = 0; state[9] = 0; state[8] = 0; state[7] = 0; state[6] = 0;
	state[5] = 0; state[4] = 0; state[3] = 0; state[2] = 0; state[1] = 0; state[0] = 0;
	

	// Copy our state into private memory
	unsigned char temp, temp2;
	unsigned char overflow = 0;
	if(tid < num_blocks) {
		for(int i = 15; i != -1; i--) {
			temp = g_counter_initial[i];
			temp2 = state[i];
			state[i] += temp + overflow;
			overflow = ((int)temp2 + (int)temp + (int)overflow > 255);
		}
	}

	//ENCRYPT THE STATE

	///////////////////////////////////////////	
	/*
	AddRoundKey( state, &g_expanded_key[0];
	for(int i = 1; i < 10; i++) {
		SubBytes(state);
		ShiftRows(state);
		MixColumns(state);
		AddRoundKey(state, &g_expanded_key[4 * i]);
	}
	SubBytes(state);
	ShiftRows(state);
	AddRoundKey(state, &g_expanded_key[4 * 10]);
	*/
	///////////////////////////////////////////

	//The following is an unrolled version of the code in comments above.

	//Round 0
	AddRoundKey(state, &g_expanded_key[0]);

	//Round 1
	SubBytes(state);
	ShiftRows(state);
	MixColumns(state);
	AddRoundKey(state, &g_expanded_key[4]);

	//Round 2
	SubBytes(state);
	ShiftRows(state);
	MixColumns(state);
	AddRoundKey(state, &g_expanded_key[8]);

	//Round 3
	SubBytes(state);
	ShiftRows(state);
	MixColumns(state);
	AddRoundKey(state, &g_expanded_key[12]);

	//Round 4
	SubBytes(state);
	ShiftRows(state);
	MixColumns(state);
	AddRoundKey(state, &g_expanded_key[16]);

	//Round 5
	SubBytes(state);
	ShiftRows(state);
	MixColumns(state);
	AddRoundKey(state, &g_expanded_key[20]);

	//Round 6
	SubBytes(state);
	ShiftRows(state);
	MixColumns(state);
	AddRoundKey(state, &g_expanded_key[24]);

	//Round 7
	SubBytes(state);
	ShiftRows(state);
	MixColumns(state);
	AddRoundKey(state, &g_expanded_key[28]);

	//Round 8
	SubBytes(state);
	ShiftRows(state);
	MixColumns(state);
	AddRoundKey(state, &g_expanded_key[32]);

	//Round 9
	SubBytes(state);
	ShiftRows(state);
	MixColumns(state);
	AddRoundKey(state, &g_expanded_key[36]);

	//Round 10
	SubBytes(state);
	ShiftRows(state);
	//no mix columns
	AddRoundKey(state, &g_expanded_key[40]);


	// xor the input text with the encrypted counter state
	if(tid < num_blocks) {

		text[(tid << 4) + 0] = text[(tid << 4) + 0] ^ state[0];
		text[(tid << 4) + 1] = text[(tid << 4) + 1] ^ state[1];
		text[(tid << 4) + 2] = text[(tid << 4) + 2] ^ state[2];
		text[(tid << 4) + 3] = text[(tid << 4) + 3] ^ state[3];
		text[(tid << 4) + 4] = text[(tid << 4) + 4] ^ state[4];
		text[(tid << 4) + 5] = text[(tid << 4) + 5] ^ state[5];
		text[(tid << 4) + 6] = text[(tid << 4) + 6] ^ state[6];
		text[(tid << 4) + 7] = text[(tid << 4) + 7] ^ state[7];
		text[(tid << 4) + 8] = text[(tid << 4) + 8] ^ state[8];
		text[(tid << 4) + 9] = text[(tid << 4) + 9] ^ state[9];
		text[(tid << 4) + 10] = text[(tid << 4) + 10] ^ state[10];
		text[(tid << 4) + 11] = text[(tid << 4) + 11] ^ state[11];
		text[(tid << 4) + 12] = text[(tid << 4) + 12] ^ state[12];
		text[(tid << 4) + 13] = text[(tid << 4) + 13] ^ state[13];
		text[(tid << 4) + 14] = text[(tid << 4) + 14] ^ state[14];
		text[(tid << 4) + 15] = text[(tid << 4) + 15] ^ state[15];

	}

}


//Main Cuda Function for counter mode AES encryption and decryption 
//This function does the setup for the kernels
extern "C" int cuda_main( unsigned int* expanded_key, unsigned char initial_counter[16], unsigned char* text, unsigned int size_bytes)
{

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float t;
	float cuda_kernel_time = 0, cuda_HtoD_time = 0, cuda_DtoH_time = 0;

	cudaError_t err = cudaSuccess;
	unsigned int num_states = size_bytes >> 4;

	unsigned int counter_state_size = 16*sizeof(unsigned char)*num_states;


	unsigned char* d_text;
	cudaMalloc( (void**) &d_text, counter_state_size);
	cudaEventRecord(start, 0);
	cudaMemcpy( d_text, text, counter_state_size, cudaMemcpyHostToDevice );
	cudaEventRecord(stop, 0);

	err = cudaGetLastError();
	if(err != cudaSuccess) { printf("error 4: %s\n", cudaGetErrorString(err)); }

	// Get kernel time
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&t, start, stop);
	cuda_HtoD_time = t;

	//copy tables to device memory
	err = cudaMemcpyToSymbol(g_counter_initial, &initial_counter[0], 16, size_t(0), cudaMemcpyHostToDevice);
	PRINT_CUDA_ERROR(err, "COPY INITIAL COUNTER FAIL");//check error
	
	err = cudaMemcpyToSymbol(g_expanded_key, &expanded_key[0], 176, size_t(0), cudaMemcpyHostToDevice);
	PRINT_CUDA_ERROR(err, "COPY EXPANDED KEY FAIL");//check error

	//kernel configuration
	int threadsPerBlock = NUM_THREADS;//256
	int blocksPerGrid = (num_states + threadsPerBlock - 1) / threadsPerBlock;

	//call main kernel
	cudaEventRecord(start);
	cuda_main_kernel<<<blocksPerGrid, threadsPerBlock>>>( expanded_key, initial_counter, d_text, num_states);
	cudaEventRecord(stop);
	err = cudaGetLastError();
	PRINT_CUDA_ERROR( err, "CUDA MAIN KERNEL ERROR" );//check error
	
	err = cudaGetLastError();
	if(err != cudaSuccess) { printf("error 4: %s\n", cudaGetErrorString(err)); }

	// Get kernel time
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&t, start, stop);
	cuda_kernel_time = t;

	cudaEventRecord(start);
	cudaMemcpy( text, d_text, counter_state_size, cudaMemcpyDeviceToHost );
	cudaEventRecord(stop);

	// Get kernel time
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&t, start, stop);
	cuda_DtoH_time += t;


	cudaFree(d_text);

	printf("Transfer to device: %f ms\n", cuda_HtoD_time);
	printf("Kernel time:        %f ms\n", cuda_kernel_time);
	printf("Transfer to host:   %f ms\n", cuda_DtoH_time);
	
	return 0;

}


