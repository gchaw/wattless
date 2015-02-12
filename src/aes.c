/* 
 * File:   cpu_cipher.c
 * Author: chawgary, syrowikb
 *
 * Created on November 24 2013
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <cuda_runtime.h>

#include "global.h"
#include "cpu_cipher.h"
#include "opencl_cipher.h"
#include "cuda_cipher.h"

FILE *fkey;
FILE *finput;
FILE *foutput;
FILE *fcounter;
bool counter_mode = false;
bool encrypt_ppm_header = true;
bool do_decrypt = false;
bool print_output = false;

typedef enum {
	cpu,
	cuda,
	opencl
} Method;

Method method = cpu;

void usage(void) {
	printf(	"Usage:\n"
		//"\t ./aes -i <input> -k <key> [-d] [-c <ctr_init>] [-o <output>] [-s] [-p] [-m <cpu|cuda|opencl>]\n"
		"\t ./aes -i <input> -k <key> -c <ctr_init> [-o <output>] [-s] [-p] [-m <cpu|cuda|opencl>]\n"
		"\t -i <input>\tinput file to encrypt\n"
		"\t -k <key>\tkey to use for encryption\n"
		//"\t -d\t\tdecrypt\n"
		"\t -c <ctr_init>\tuse counter mode, and initialize counter\n"
		"\t -o <output>\toutput file (default is <input>.out)\n"
		"\t -s\t\tinput is a ppm file, so do not encrypt the header\n"
		"\t -p\t\tprint output\n"
		"\t -m\t\tmethod (default is cpu)\n"
	      );
}

void increment_counter(unsigned char * counter) {
	// increment the counter
	int i;
	for(i = 15; i != 0; i--) {
		counter[i]++;
		if(counter[i] != 0) break;
	}
}

void print_16_char(unsigned char *c) {
	int i;
	for(i = 0; i < 16; i++) {
		printf("%02x", c[i]);
	}
	printf("\n");
}

void parse_input(int argc, char* argv[]) {
	bool output_specified = false;
	char * in_file_name;

	int i;
	for(i = 1; i < argc; i++) {
		if(strcmp(argv[i], "-i") == 0) {
			finput = fopen(argv[i + 1], "rb");
			in_file_name = argv[i + 1];
			if(!finput) {
				printf("Unable to open input file\n"); 
				exit(1);
			}
		}
		if(strcmp(argv[i], "-k") == 0) {
			fkey = fopen(argv[i + 1], "rb");
			if(!fkey) {
				printf("Unable to open key file\n");
				exit(1);
			}
		}
		if(strcmp(argv[i], "-c") == 0) {
			counter_mode = true;
			fcounter = fopen(argv[i + 1], "rb");
			if(!fcounter) {
				printf("Unable to open counter initialization file");
				printf(" \"%s\"\n", argv[i + 1]);
				exit(1);
			}
		}
		if(strcmp(argv[i], "-o") == 0) {
			output_specified = true;
			foutput = fopen(argv[i + 1], "wb");
			if(!foutput) {
				printf("Unable to access output file\n");
				exit(1);
			}
		}
		if(strcmp(argv[i], "-s") == 0) {
			encrypt_ppm_header = false;
		}
		if(strcmp(argv[i], "-d") == 0) {
			do_decrypt = true;
		}
		if(strcmp(argv[i], "-p") == 0) {
			print_output = true;
		}
		if(strcmp(argv[i], "-m") == 0) {
			if(strcmp(argv[i + 1], "cpu") == 0) {
				method = cpu;
			} else if(strcmp(argv[i + 1], "cuda") == 0) {
				method = cuda;
			} else if(strcmp(argv[i + 1], "opencl") == 0) {
				method = opencl;
			} else {
				printf("Invalid method: \"%s\".\n", argv[i + 1]);
				exit(1);
			}
		}
	}

	// if an output file was not specified, make one called <input>.out
	if(!output_specified) {
		char output_name[256];
		snprintf(output_name, sizeof(output_name), "%s%s", in_file_name, ".out");
		foutput = fopen(output_name, "wb");
		if(!foutput) {
			printf("Unable to access output file\n");
			exit(1);
		}
		printf("creating output file \"%s\".\n", output_name);
	}
}

// this function is not very robust (partly due to the ppm file format...)
void skip_ppm_header(void) {
	// read until we finish the header
	char buffer[1024];
	int i = 0;
	char c;
	buffer[i++] = fgetc(finput);
	buffer[i++] = fgetc(finput);
	if(buffer[0] != 'P' || buffer[1] != '6') {
		printf("Invalid file format.  No PPM header detected.\n");
		exit(1);
	}
	// TODO Read until line doesn't begin with '#', then get width, height
	// and bitdepth(?) from next two lines.  The line after that should
	// contain the picture data
	while(true) {
		// read until we find "255\n"
		if((c = fgetc(finput)) == 0x0a) {
			if(buffer[i - 3] == '2' && buffer[i - 2] == '5'
					&& buffer[i - 1] == '5') {
				buffer[i++] = c;
				break;
			}
		}
		buffer[i++] = c;
		if(i > 1022) {
			int j;
			// write buffer to output
			for(j = 0; j < i; j++) {
				fputc(buffer[j], foutput);
			}
			i = 0;
		}	
	}
	int j;
	for(j = 0; j < i; j++) {
		fputc(buffer[j], foutput);
	}
}

void ctr_encrypt(unsigned int * key, unsigned char * counter_init, unsigned char * in_out, unsigned int size_bytes) {
	struct timespec start, end;
	if(method == cpu) {
		clock_gettime(CLOCK_REALTIME, &start);
		ctr_cpu_encrypt(key, counter_init, in_out, size_bytes);
		clock_gettime(CLOCK_REALTIME, &end);
	} else if(method == cuda) {
		clock_gettime(CLOCK_REALTIME, &start);
		cuda_main( key, counter_init, in_out, size_bytes );		
		clock_gettime(CLOCK_REALTIME, &end);
	} else if(method == opencl) {
		clock_gettime(CLOCK_REALTIME, &start);
		ctr_opencl_encrypt(key, counter_init, in_out, size_bytes);
		clock_gettime(CLOCK_REALTIME, &end);
	}
	unsigned long clock_gettime_usec = ((end.tv_sec - start.tv_sec) * 1000000 + \
							(end.tv_nsec - start.tv_nsec) / 1000);
	printf("aes.c: clock_gettime() %lu us\n", clock_gettime_usec);
}

void _encrypt(unsigned char *data, unsigned int *key, unsigned int block_count) {
	if(method == cpu) {
		cpu_encrypt(data, key, block_count);
	} else if(method == cuda) {  // CUDA and OpenCL encrypt were not implemented
		//cuda_encrypt(data, key);
		//printf("going to cuda main\n");
		//cuda_main();
	} else if(method == opencl) {
		//opencl_encrypt(data, key, block_count);
	}
}
void _decrypt(unsigned char *data, unsigned int *key, unsigned int block_count) {
	if(method == cpu) {
		cpu_decrypt(data, key, block_count);
	} else if(method == cuda) {  // CUDA and OpenCL decrypt were not implemented
		//cuda_decrypt(in, out, w);
		printf("CUDA decrypt not implemented\n");
	} else if(method == opencl) {
		//opencl_decrypt(in, out, w);
		printf("OpenCL decrypt not implemented\n");
	}
}

int main(int argc, char* argv[]) {
	if(argc < 5) {
		printf("You must at least specify an input file and a key.\n");
		usage();
		return 1;
	}
	parse_input(argc, argv);

	unsigned int count;
	unsigned char key[16];
	unsigned char buf[16];
	unsigned char counter[16];
	int num_keys = BLOCK_SIZE * (NUM_ROUNDS + 1);
	unsigned int* expanded_key = (unsigned int*)malloc(num_keys * sizeof(int));

	// read the key, and do key expansion
	fread(&key, 1, 16, fkey);
	fclose(fkey);
	KeyExpansion(key, expanded_key);

	// if requested, skip (do not encrypt) the header on ppm files
	if(!encrypt_ppm_header) {
		printf("skipping the ppm header\n");
		skip_ppm_header();
	}
	
	// make buffer big enough for all data to encrypt
	unsigned int start = ftell(finput);
	fseek(finput, 0L, SEEK_END);
	unsigned int end = ftell(finput);
	fseek(finput, start, SEEK_SET);
	unsigned int current_pos = ftell(finput);
	unsigned int size_of_data_in_bytes = end - start;
	unsigned int pad_needed = 0;
	unsigned int num_padded_blocks = 0;
	// add one to the size to keep track of the amount of padding added
	if(! do_decrypt) {
		size_of_data_in_bytes++;
	}
	if((size_of_data_in_bytes & 0xF) != 0) {
		if(do_decrypt) {
			printf("cannot decrypt a file that is not aligned to 128 bits!!\n");
			exit(1);
		}
		pad_needed = 16 - (size_of_data_in_bytes & 0xF);
		printf("will need to pad with %u bytes\n", pad_needed);
		size_of_data_in_bytes += pad_needed;
		num_padded_blocks = 1;
	}
	unsigned int num_blocks = size_of_data_in_bytes >> 4;
	unsigned char * in_out;
	if(method == cuda) {
		cudaMallocHost((void **)&in_out, size_of_data_in_bytes * sizeof(char));
	} else {
		in_out = (unsigned char*)malloc(size_of_data_in_bytes * sizeof(char));
	}

	if(! do_decrypt) { // as long as we are encrypting....
		// the last byte tells us how many padding bytes were added
		in_out[size_of_data_in_bytes - 1] = pad_needed + 1;
	}


	if(counter_mode) {
		printf("in counter mode\n");

		// initialize the counter
		fread(&counter, 1, 16, fcounter);
		fclose(fcounter);

		// read from file to get data and put it in the in_out buffer
		unsigned int block_index = 0;
		unsigned int i;
		while((count = fread(&buf, 1, 16, finput)) == 16) {
			memcpy(&in_out[block_index * 16], buf, 16);
			block_index++;
		}
		if(count > 0) {
			for(;count < 15; count++) { buf[count] = 0x00; }
			memcpy(&in_out[block_index * 16], buf, 15);
		}

		// do encryption
		ctr_encrypt(expanded_key, counter, in_out, size_of_data_in_bytes);

		// TODO: read last byte, and remove padding if decrypting
		if(do_decrypt) {
			int discard_bytes =  (int)in_out[size_of_data_in_bytes - 1];
			printf("need to discard %d bytes\n", discard_bytes);
			size_of_data_in_bytes -= discard_bytes;
		}


		// write result to output file
		int chunk_index = 0;
		int chunk_size = 1024;
		while((chunk_index + 1) * chunk_size < size_of_data_in_bytes) {
			fwrite(&in_out[chunk_index * chunk_size], 1, chunk_size, foutput);
			chunk_index++;
		}
		unsigned int bytes_left = size_of_data_in_bytes - chunk_size * chunk_index;
		fwrite(&in_out[chunk_index * chunk_size], 1, bytes_left, foutput);
		



	} else {
		printf("not in counter mode\n");

		// read from file into in_out buffer
		int chunk_index = 0;
		int chunk_size = 1024;
		while((count = fread(&in_out[chunk_index * chunk_size], 1, chunk_size, finput))) {
			if(count < chunk_size) break;
			chunk_index++;
		}
		// pad the last block
		count += chunk_size * chunk_index;
		printf("padding from %d to %d\n", count, size_of_data_in_bytes);
		for(; count < size_of_data_in_bytes; count++) {
			in_out[count] = 0x00;
		}

		// do encrpytion/decryption
		if(do_decrypt) {
			_decrypt(in_out, expanded_key, num_blocks);
		} else {
			_encrypt(in_out, expanded_key, num_blocks);
		}

		// write result to output file
		chunk_index = 0;
		while((chunk_index + 1) * chunk_size < size_of_data_in_bytes) {
			fwrite(&in_out[chunk_index * chunk_size], 1, chunk_size, foutput);
			chunk_index++;
		}
		unsigned int bytes_left = size_of_data_in_bytes - chunk_size * chunk_index;
		fwrite(&in_out[chunk_index * chunk_size], 1, bytes_left, foutput);
	}
	
	// free malloc'd memory
	if(method == cuda) {
		cudaFreeHost(in_out);
	} else {
		free(in_out);
	}
	free(expanded_key);

	// close input and output files
	fclose(finput);
	fclose(foutput);
        
	// print victory message
    printf("I AM WATTLESS!!!\n\n");

	return 0;
}

