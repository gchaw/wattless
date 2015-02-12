/* 
 * File:   opencl_cipher.c
 * Author: syrowikb
 *
 * Created on December 3 2013
 */



#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>
#include "opencl_cipher.h"

#define MAX_SOURCE_SIZE (0x100000)

void Print_state(unsigned char* state) {
	int i, j;
	for(i = 0; i < 4; i++) { // column
		for(j = 0; j < 4; j++) { // row
			printf("%02x ", state[j * 4 + i]);
		}
		printf("\n");
	}
	printf("\n");
}

void print_block(unsigned char* block) {
	int i;
	for(i = 0; i < 16; i++) {
		printf("%02x", block[i]);
	}
	printf("\n");
}

void ctr_opencl_encrypt(unsigned int *key, unsigned char * counter_init, unsigned char *in, unsigned int size_bytes) {
	unsigned int block_count = size_bytes / 16;
	struct timespec start, end;

	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;
	cl_context context = NULL;
	cl_command_queue command_queue = NULL;
	cl_mem d_key = NULL;
	cl_mem d_counter_init = NULL;
	cl_mem d_in_out = NULL;
	cl_mem d_num_blocks = NULL;
	cl_program program = NULL;
	cl_kernel kernel = NULL;
	cl_uint status_num_devices;
	cl_uint status_num_platforms;
	cl_int status;
	cl_event timing_event;
	cl_ulong start_time, end_time;
	size_t d_in_out_size = size_bytes * sizeof(unsigned char);
	int key_size = BLOCK_SIZE * (NUM_ROUNDS + 1);

	// Get source file and read in the kernel
	FILE *fp;
	const char fileName[] = "./opencl_cipher.cl";
	size_t source_size;
	char *source_str;
	fp = fopen(fileName, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char *)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp );
	fclose( fp );


	// Get the platforms, and choose the first one
	status = clGetPlatformIDs(1, &platform_id, &status_num_platforms);
	if(status != CL_SUCCESS) {
		printf("Error getting platforms\n");
		exit(1);
	}

	// Query the platfrom and choose the first device
	status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &status_num_devices);
	if(status != CL_SUCCESS) {
		printf("Error getting device: %d \n", status);
		printf("%s\n", my_error_codes);
		exit(1);
	}

	// create the context
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &status);
	//context = clCreateContext(0, 1, &device_id, NULL, &status);
	if(status != CL_SUCCESS) {
		printf("Error creating context\n");
	}

	// create the command queue
	command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &status);
	if(status != CL_SUCCESS) {
		printf("Error creating command queue1\n");
	}

	// allocate memory on devide 
	d_key = clCreateBuffer(context, CL_MEM_READ_WRITE, key_size * sizeof(int), NULL, &status);
	if(status != CL_SUCCESS) {
		printf("Error clCreateBuffer2\n");
	}
	d_counter_init = clCreateBuffer(context, CL_MEM_READ_WRITE, 16 * sizeof(char), NULL, &status);
	if(status != CL_SUCCESS) {
		printf("Error clCreateBuffer3\n");
	}
	d_in_out = clCreateBuffer(context, CL_MEM_READ_WRITE, d_in_out_size, NULL, &status);
	if(status != CL_SUCCESS) {
		printf("Error clCreateBuffer4\n");
	}
	d_num_blocks = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned int), NULL, &status);
	if(status != CL_SUCCESS) {
		printf("Error clCreateBuffer5\n");
	}

	// copy data from host memory to device memory
	status = clEnqueueWriteBuffer(command_queue, d_key, CL_TRUE, 0, key_size * sizeof(int), key, 0, NULL, &timing_event);
	if(status != CL_SUCCESS) {
		printf("Error clEnqueueWriteBuffer 6\n");
	}
	status = clEnqueueWriteBuffer(command_queue, d_counter_init, CL_TRUE, 0, 16 * sizeof(char), counter_init, 0, NULL, &timing_event);
	if(status != CL_SUCCESS) {
		printf("Error clEnqueueWriteBuffer 7\n");
	}
	clock_gettime(CLOCK_REALTIME, &start);
	status = clEnqueueWriteBuffer(command_queue, d_in_out, CL_TRUE, 0, d_in_out_size, in, 0, NULL, &timing_event);
	if(status != CL_SUCCESS) {
		printf("Error clEnqueueWriteBuffer 8\n");
	}
	clock_gettime(CLOCK_REALTIME, &end);

	//*
	// time how long it takes to copy data to device
	status = clFinish(command_queue);
	clGetEventProfilingInfo(timing_event, CL_PROFILING_COMMAND_START,
			sizeof(cl_ulong), &start_time, NULL);
	clGetEventProfilingInfo(timing_event, CL_PROFILING_COMMAND_END,
			sizeof(cl_ulong), &end_time, NULL);
	cl_ulong kernelMemCpyToDevice3 = end_time - start_time;
	//*/

	status = clEnqueueWriteBuffer(command_queue, d_num_blocks, CL_TRUE, 0, sizeof(unsigned int), &block_count, 0, NULL, &timing_event);
	if(status != CL_SUCCESS) {
		printf("Error clEnqueueWriteBuffer\n");
	}


	// create the program from the source file
	program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &status);
	if(status != CL_SUCCESS) {
		printf("Error clCreateProgramWithSource\n");
	}

	status = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	if(status != CL_SUCCESS) {
		printf("Error: %d clBuildProgram \n", status);
	}
	// print out compile errors
	if (status == CL_BUILD_PROGRAM_FAILURE) {
		// Determine the size of the log
		size_t log_size;
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

		// Allocate memory for the log
		char *log = (char *) malloc(log_size);

		// Get the log
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

		// Print the log
		printf("%s\n", log);
	}

	// make the kernel
	kernel = clCreateKernel(program, "aes_ctr_encrypt", &status);
	if(status != CL_SUCCESS) {
		printf("Error: clCreateKernel%d \n", status);
	}

	unsigned long clock_gettime_usec = ((end.tv_sec - start.tv_sec) * 1000000 + \
							(end.tv_nsec - start.tv_nsec) / 1000);
	printf("Building the program took %lu us\n", clock_gettime_usec);

	// set up the kernel arguments

	// first argument - AES key
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_key);
	if(status != CL_SUCCESS) {
		printf("Error clSetKernelArg\n");
	}
	// second argument - Counter initialization
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_counter_init);
	if(status != CL_SUCCESS) {
		printf("Error clSetKernelArg\n");
	}
	// third argument - input/output buffer
	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_in_out);
	if(status != CL_SUCCESS) {
		printf("Error clSetKernelArg\n");
	}
	// fourth argument - number of blocks to process (also, number of valid threads)
	status = clSetKernelArg(kernel, 3, sizeof(cl_int), &d_num_blocks);
	if(status != CL_SUCCESS) {
		printf("Error: %d\n clSetKernelArg", status);
		printf("%s\n", my_error_codes);
	}

	
	// Set up the local and global work group size
	int local_w_s = block_count;
	int global_w_s = ((block_count / 32) + 1) * 32;
	
	// the GTX 780 and 480 can handle 1024 threads per block, but 256 works nicely
	int threads_per_block = 256; 
	if(block_count > threads_per_block) {
		local_w_s = threads_per_block;
		global_w_s = (block_count / threads_per_block) + 1;
		global_w_s *= threads_per_block;
	}

	size_t global_work_size[3] = {global_w_s, 0, 0};
	size_t local_work_size[3]  = {local_w_s, 0, 0};

	// run the kernel
	status = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &timing_event);
	if(status != CL_SUCCESS) {
		printf("EnqueueNDRangeKernel() error %d\n", status);
		printf("%s\n", my_error_codes);
	}

	//*
	// time how long it takes to run the kernel
	status = clFinish(command_queue);
	if(status != CL_SUCCESS) {
		printf("clFinish() error %d\n", status);
		printf("%s\n", my_error_codes);
	}
	clGetEventProfilingInfo(timing_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
	clGetEventProfilingInfo(timing_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
	cl_ulong kernelExecTimeNs = end_time - start_time;
	//*/


	// copy data from device memory to host memory
	status = clEnqueueReadBuffer(command_queue, d_in_out, CL_TRUE, 0, d_in_out_size, in, 0, NULL, &timing_event);
	if(status != CL_SUCCESS) {
		printf("Error reading data from device %d\n", status);
		printf("%s\n", my_error_codes);
	}


	//*
	// time how long it takes to copy data to device
	status = clFinish(command_queue);
	clGetEventProfilingInfo(timing_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
	clGetEventProfilingInfo(timing_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
	cl_ulong kernelMemCpyFromDevice = end_time - start_time;
	//*/

	// free memory resources
	status = clFinish(command_queue);
	status = clReleaseKernel(kernel);
	status = clReleaseProgram(program);
	status = clReleaseMemObject(d_in_out);
	if(status != CL_SUCCESS) {
		printf("clReleaseMemObject(d_in_out) error %d\n", status);
		printf("%s\n", my_error_codes);
	}
	status = clReleaseMemObject(d_counter_init);
	if(status != CL_SUCCESS) {
		printf("clReleaseMemObject(d_counter_init) error %d\n", status);
		printf("%s\n", my_error_codes);
	}
	status = clReleaseMemObject(d_key);
	if(status != CL_SUCCESS) {
		printf("clReleaseMemObject(d_key) error %d\n", status);
		printf("%s\n", my_error_codes);
	}
	status = clReleaseMemObject(d_num_blocks);
	if(status != CL_SUCCESS) {
		printf("clReleaseMemObject(d_num_blocks) error %d\n", status);
		printf("%s\n", my_error_codes);
	}
	status = clReleaseCommandQueue(command_queue);
	status = clReleaseContext(context);

	free(source_str);


	//*
	//printf("MemCpy1 to device took %luus\n", kernelMemCpyToDevice1 / 1000);
	//printf("MemCpy2 to device took %luus\n", kernelMemCpyToDevice2 / 1000);
	printf("MemCpy3 to device took %luus\n", kernelMemCpyToDevice3 / 1000);
	//printf("MemCpy4 to device took %luus\n", kernelMemCpyToDevice4 / 1000);
	printf("OpenCL kernel took %luus to run\n", kernelExecTimeNs / 1000);
	printf("MemCpy from device took %luus\n", kernelMemCpyFromDevice / 1000);
	printf("Total kernel + transfer time: %luus\n", (kernelMemCpyToDevice3 +
				kernelExecTimeNs + kernelMemCpyFromDevice) / 1000);
	//*/

	return;
}



