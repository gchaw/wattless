/* 
 * File:   opencl_ctr_kernel.cl
 * Author: syrowikb
 *
 * Created on December 11 2013
 */

#define BLOCK_SIZE 4
#define NUM_KEYS 11

uchar constant _GF_2[256] = {
/* 00 */ 0x00U, 0x02U, 0x04U, 0x06U, 0x08U, 0x0aU, 0x0cU, 0x0eU,
/* 08 */ 0x10U, 0x12U, 0x14U, 0x16U, 0x18U, 0x1aU, 0x1cU, 0x1eU,
/* 10 */ 0x20U, 0x22U, 0x24U, 0x26U, 0x28U, 0x2aU, 0x2cU, 0x2eU,
/* 18 */ 0x30U, 0x32U, 0x34U, 0x36U, 0x38U, 0x3aU, 0x3cU, 0x3eU,
/* 20 */ 0x40U, 0x42U, 0x44U, 0x46U, 0x48U, 0x4aU, 0x4cU, 0x4eU,
/* 28 */ 0x50U, 0x52U, 0x54U, 0x56U, 0x58U, 0x5aU, 0x5cU, 0x5eU,
/* 30 */ 0x60U, 0x62U, 0x64U, 0x66U, 0x68U, 0x6aU, 0x6cU, 0x6eU,
/* 38 */ 0x70U, 0x72U, 0x74U, 0x76U, 0x78U, 0x7aU, 0x7cU, 0x7eU,
/* 40 */ 0x80U, 0x82U, 0x84U, 0x86U, 0x88U, 0x8aU, 0x8cU, 0x8eU,
/* 48 */ 0x90U, 0x92U, 0x94U, 0x96U, 0x98U, 0x9aU, 0x9cU, 0x9eU,
/* 50 */ 0xa0U, 0xa2U, 0xa4U, 0xa6U, 0xa8U, 0xaaU, 0xacU, 0xaeU,
/* 58 */ 0xb0U, 0xb2U, 0xb4U, 0xb6U, 0xb8U, 0xbaU, 0xbcU, 0xbeU,
/* 60 */ 0xc0U, 0xc2U, 0xc4U, 0xc6U, 0xc8U, 0xcaU, 0xccU, 0xceU,
/* 68 */ 0xd0U, 0xd2U, 0xd4U, 0xd6U, 0xd8U, 0xdaU, 0xdcU, 0xdeU,
/* 70 */ 0xe0U, 0xe2U, 0xe4U, 0xe6U, 0xe8U, 0xeaU, 0xecU, 0xeeU,
/* 78 */ 0xf0U, 0xf2U, 0xf4U, 0xf6U, 0xf8U, 0xfaU, 0xfcU, 0xfeU,
/* 80 */ 0x1bU, 0x19U, 0x1fU, 0x1dU, 0x13U, 0x11U, 0x17U, 0x15U,
/* 88 */ 0x0bU, 0x09U, 0x0fU, 0x0dU, 0x03U, 0x01U, 0x07U, 0x05U,
/* 90 */ 0x3bU, 0x39U, 0x3fU, 0x3dU, 0x33U, 0x31U, 0x37U, 0x35U,
/* 98 */ 0x2bU, 0x29U, 0x2fU, 0x2dU, 0x23U, 0x21U, 0x27U, 0x25U, 
/* a0 */ 0x5bU, 0x59U, 0x5fU, 0x5dU, 0x53U, 0x51U, 0x57U, 0x55U,
/* a8 */ 0x4bU, 0x49U, 0x4fU, 0x4dU, 0x43U, 0x41U, 0x47U, 0x45U,
/* b0 */ 0x7bU, 0x79U, 0x7fU, 0x7dU, 0x73U, 0x71U, 0x77U, 0x75U,
/* b8 */ 0x6bU, 0x69U, 0x6fU, 0x6dU, 0x63U, 0x61U, 0x67U, 0x65U,
/* c0 */ 0x9bU, 0x99U, 0x9fU, 0x9dU, 0x93U, 0x91U, 0x97U, 0x95U,
/* c8 */ 0x8bU, 0x89U, 0x8fU, 0x8dU, 0x83U, 0x81U, 0x87U, 0x85U,
/* d0 */ 0xbbU, 0xb9U, 0xbfU, 0xbdU, 0xb3U, 0xb1U, 0xb7U, 0xb5U,
/* d8 */ 0xabU, 0xa9U, 0xafU, 0xadU, 0xa3U, 0xa1U, 0xa7U, 0xa5U,
/* e0 */ 0xdbU, 0xd9U, 0xdfU, 0xddU, 0xd3U, 0xd1U, 0xd7U, 0xd5U,
/* e8 */ 0xcbU, 0xc9U, 0xcfU, 0xcdU, 0xc3U, 0xc1U, 0xc7U, 0xc5U,
/* f0 */ 0xfbU, 0xf9U, 0xffU, 0xfdU, 0xf3U, 0xf1U, 0xf7U, 0xf5U,
/* f8 */ 0xebU, 0xe9U, 0xefU, 0xedU, 0xe3U, 0xe1U, 0xe7U, 0xe5U,
};

uchar constant _sbox[256] = {
/*	0	 1	 2	 3	 4	 5	 6	 7	 8	 9	 A	 B	 C	 D	 E	 F*/
   0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
   0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
   0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
   0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
   0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
   0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
   0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
   0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
   0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
   0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
   0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
   0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
   0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
   0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
   0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
   0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
/*	0	 1	 2	 3	 4	 5	 6	 7	 8	 9	 A	 B	 C	 D	 E	 F*/
};


// not much way to improve this unless there is a way to calculate the appropriate
// value faster than getting it from the lookup table (calculating would be more
// secure...)
inline void SubBytes(uchar *state, local uchar *sbox) {
//inline void SubBytes(uchar *state, constant uchar *sbox) {
	state[0]  = sbox[state[0]];
	state[1]  = sbox[state[1]];
	state[2]  = sbox[state[2]];
	state[3]  = sbox[state[3]];
	state[4]  = sbox[state[4]];
	state[5]  = sbox[state[5]];
	state[6]  = sbox[state[6]];
	state[7]  = sbox[state[7]];
	state[8]  = sbox[state[8]];
	state[9]  = sbox[state[9]];
	state[10] = sbox[state[10]];
	state[11] = sbox[state[11]];
	state[12] = sbox[state[12]];
	state[13] = sbox[state[13]];
	state[14] = sbox[state[14]];
	state[15] = sbox[state[15]];
}

// not much way to improve this
inline void ShiftRows(uchar *state) {   
	uchar temp = state[1];
	
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


inline void AddRoundKey(uchar *state, local uint *w_) {
//inline void AddRoundKey(uchar *state, global uint *w_) {
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


inline void MixColumns(uchar *state, local uchar *GF_2) {
//inline void MixColumns(uchar *state, constant uchar *GF_2) {
	uchar x[4];
	x[0] = state[0];
	x[1] = state[1];
	x[2] = state[2];
	x[3] = state[3];
	uchar * y = (uchar *)&state[0];
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
	
	x[0] = state[4];
	x[1] = state[5];
	x[2] = state[6];
	x[3] = state[7];
	y = (uchar *)&state[4];
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

	x[0] = state[8];
	x[1] = state[9];
	x[2] = state[10];
	x[3] = state[11];
	y = (uchar *)&state[8];
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

	x[0] = state[12];
	x[1] = state[13];
	x[2] = state[14];
	x[3] = state[15];
	y = (uchar *)&state[12];
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


__kernel void aes_ctr_encrypt(__global uint * _key, __global uchar * _counter_init, __global uchar * in_out, const unsigned int num_blocks)
{
	///*
	// copy lookup tables, key, and counter_init into local memory
	__local uchar GF_2[256];
	__local uchar sbox[256];
	__local uint key[11 * 4];
	__local uchar counter_init[16];

	int init = get_local_id(0) + get_local_size(0) * get_local_id(1);
	int step = get_local_size(0) * get_local_size(1);
	
	for(int i = init; i < 256; i += step) {
		GF_2[i] = _GF_2[i];
		sbox[i] = _sbox[i];
	}
	for(int i = init; i < 44; i += step) {
		key[i] = _key[i];
	}
	for(int i = init; i < 16; i += step) {
		counter_init[i] = _counter_init[i];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	//*/

	/*
	constant uchar * GF_2 = _GF_2;
	constant uchar *sbox = _sbox;
	global uint *key = _key;
	global uchar *counter_init = _counter_init;
	*/

	// calculate the global id, and make this the initial value of the counter
	uint gid = get_global_id(0);
	__private uchar state[16];
	state[15] = gid & 0xFF;
	state[14] = (gid >> 8) & 0xFF;
	state[13] = (gid >> 16) & 0xFF;
	state[12] = (gid >> 24) & 0xFF;
	state[11] = 0; state[10] = 0; state[9] = 0; state[8] = 0; state[7] = 0; state[6] = 0;
	state[5] = 0; state[4] = 0; state[3] = 0; state[2] = 0; state[1] = 0; state[0] = 0;

	// copy our state into private memory (add counter_init to global id to get incremented counter)
	uchar temp, temp2;
	uchar overflow = 0;
	if(gid < num_blocks) {
		temp = counter_init[0];
		temp2 = state[0];
		state[0] += temp + overflow;
		overflow = ((int)temp2 + (int)temp + (int)overflow > 255);
		temp = counter_init[1];
		temp2 = state[1];
		state[1] += temp + overflow;
		overflow = ((int)temp2 + (int)temp + (int)overflow > 255);
		temp = counter_init[2];
		temp2 = state[2];
		state[2] += temp + overflow;
		overflow = ((int)temp2 + (int)temp + (int)overflow > 255);
		temp = counter_init[3];
		temp2 = state[3];
		state[3] += temp + overflow;
		overflow = ((int)temp2 + (int)temp + (int)overflow > 255);
		temp = counter_init[4];
		temp2 = state[4];
		state[4] += temp + overflow;
		overflow = ((int)temp2 + (int)temp + (int)overflow > 255);
		temp = counter_init[5];
		temp2 = state[5];
		state[5] += temp + overflow;
		overflow = ((int)temp2 + (int)temp + (int)overflow > 255);
		temp = counter_init[6];
		temp2 = state[6];
		state[6] += temp + overflow;
		overflow = ((int)temp2 + (int)temp + (int)overflow > 255);
		temp = counter_init[7];
		temp2 = state[7];
		state[7] += temp + overflow;
		overflow = ((int)temp2 + (int)temp + (int)overflow > 255);
		temp = counter_init[8];
		temp2 = state[8];
		state[8] += temp + overflow;
		overflow = ((int)temp2 + (int)temp + (int)overflow > 255);
		temp = counter_init[9];
		temp2 = state[9];
		state[9] += temp + overflow;
		overflow = ((int)temp2 + (int)temp + (int)overflow > 255);
		temp = counter_init[10];
		temp2 = state[10];
		state[10] += temp + overflow;
		overflow = ((int)temp2 + (int)temp + (int)overflow > 255);
		temp = counter_init[11];
		temp2 = state[11];
		state[11] += temp + overflow;
		overflow = ((int)temp2 + (int)temp + (int)overflow > 255);
		temp = counter_init[12];
		temp2 = state[12];
		state[12] += temp + overflow;
		overflow = ((int)temp2 + (int)temp + (int)overflow > 255);
		temp = counter_init[13];
		temp2 = state[13];
		state[13] += temp + overflow;
		overflow = ((int)temp2 + (int)temp + (int)overflow > 255);
		temp = counter_init[14];
		temp2 = state[14];
		state[14] += temp + overflow;
		overflow = ((int)temp2 + (int)temp + (int)overflow > 255);
		temp = counter_init[15];
		temp2 = state[15];
		state[15] += temp + overflow;
		overflow = ((int)temp2 + (int)temp + (int)overflow > 255);
	}

	AddRoundKey(state, &key[0]);

	SubBytes(state, sbox);
	ShiftRows(state);
	MixColumns(state, GF_2);
	AddRoundKey(state, &key[4]);
	SubBytes(state, sbox);
	ShiftRows(state);
	MixColumns(state, GF_2);
	AddRoundKey(state, &key[8]);
	SubBytes(state, sbox);
	ShiftRows(state);
	MixColumns(state, GF_2);
	AddRoundKey(state, &key[12]);
	SubBytes(state, sbox);
	ShiftRows(state);
	MixColumns(state, GF_2);
	AddRoundKey(state, &key[16]);
	SubBytes(state, sbox);
	ShiftRows(state);
	MixColumns(state, GF_2);
	AddRoundKey(state, &key[20]);
	SubBytes(state, sbox);
	ShiftRows(state);
	MixColumns(state, GF_2);
	AddRoundKey(state, &key[24]);
	SubBytes(state, sbox);
	ShiftRows(state);
	MixColumns(state, GF_2);
	AddRoundKey(state, &key[28]);
	SubBytes(state, sbox);
	ShiftRows(state);
	MixColumns(state, GF_2);
	AddRoundKey(state, &key[32]);
	SubBytes(state, sbox);
	ShiftRows(state);
	MixColumns(state, GF_2);
	AddRoundKey(state, &key[36]);

	SubBytes(state, sbox);
	ShiftRows(state);
	AddRoundKey(state, &key[40]);

	// xor the input with the encrypted counter
	if(gid < num_blocks) {
		in_out[(gid << 4) + 0] = in_out[(gid << 4) + 0] ^ state[0];
		in_out[(gid << 4) + 1] = in_out[(gid << 4) + 1] ^ state[1];
		in_out[(gid << 4) + 2] = in_out[(gid << 4) + 2] ^ state[2];
		in_out[(gid << 4) + 3] = in_out[(gid << 4) + 3] ^ state[3];
		in_out[(gid << 4) + 4] = in_out[(gid << 4) + 4] ^ state[4];
		in_out[(gid << 4) + 5] = in_out[(gid << 4) + 5] ^ state[5];
		in_out[(gid << 4) + 6] = in_out[(gid << 4) + 6] ^ state[6];
		in_out[(gid << 4) + 7] = in_out[(gid << 4) + 7] ^ state[7];
		in_out[(gid << 4) + 8] = in_out[(gid << 4) + 8] ^ state[8];
		in_out[(gid << 4) + 9] = in_out[(gid << 4) + 9] ^ state[9];
		in_out[(gid << 4) + 10] = in_out[(gid << 4) + 10] ^ state[10];
		in_out[(gid << 4) + 11] = in_out[(gid << 4) + 11] ^ state[11];
		in_out[(gid << 4) + 12] = in_out[(gid << 4) + 12] ^ state[12];
		in_out[(gid << 4) + 13] = in_out[(gid << 4) + 13] ^ state[13];
		in_out[(gid << 4) + 14] = in_out[(gid << 4) + 14] ^ state[14];
		in_out[(gid << 4) + 15] = in_out[(gid << 4) + 15] ^ state[15];
	}
}

















