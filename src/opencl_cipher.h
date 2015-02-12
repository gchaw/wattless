/* 
 * File:   opencl_cipher.h
 * Author: syrowikb
 *
 * Created on December 3, 2013
 */

#ifndef OPENCL_CIPHER_H
#define	OPENCL_CIPHER_H

#include "sbox.h"
#include "global.h"

static char * my_error_codes = " \n\
#define CL_SUCCESS 0\n\
#define CL_DEVICE_NOT_FOUND -1\n\
#define CL_DEVICE_NOT_AVAILABLE -2\n\
#define CL_COMPILER_NOT_AVAILABLE -3\n\
#define CL_MEM_OBJECT_ALLOCATION_FAILURE -4\n\
#define CL_OUT_OF_RESOURCES -5\n\
#define CL_OUT_OF_HOST_MEMORY -6\n\
#define CL_PROFILING_INFO_NOT_AVAILABLE -7\n\
#define CL_MEM_COPY_OVERLAP -8\n\
#define CL_IMAGE_FORMAT_MISMATCH -9\n\
#define CL_IMAGE_FORMAT_NOT_SUPPORTED -10\n\
#define CL_BUILD_PROGRAM_FAILURE -11\n\
#define CL_MAP_FAILURE -12\n\
#define CL_INVALID_VALUE -30\n\
#define CL_INVALID_DEVICE_TYPE -31\n\
#define CL_INVALID_PLATFORM -32\n\
#define CL_INVALID_DEVICE -33\n\
#define CL_INVALID_CONTEXT -34\n\
#define CL_INVALID_QUEUE_PROPERTIES -35\n\
#define CL_INVALID_COMMAND_QUEUE -36\n\
#define CL_INVALID_HOST_PTR -37\n\
#define CL_INVALID_MEM_OBJECT -38\n\
#define CL_INVALID_IMAGE_FORMAT_DESCRIPTOR -39\n\
#define CL_INVALID_IMAGE_SIZE -40\n\
#define CL_INVALID_SAMPLER -41\n\
#define CL_INVALID_BINARY -42\n\
#define CL_INVALID_BUILD_OPTIONS -43\n\
#define CL_INVALID_PROGRAM -44\n\
#define CL_INVALID_PROGRAM_EXECUTABLE -45\n\
#define CL_INVALID_KERNEL_NAME -46\n\
#define CL_INVALID_KERNEL_DEFINITION -47\n\
#define CL_INVALID_KERNEL -48\n\
#define CL_INVALID_ARG_INDEX -49\n\
#define CL_INVALID_ARG_VALUE -50\n\
#define CL_INVALID_ARG_SIZE -51\n\
#define CL_INVALID_KERNEL_ARGS -52\n\
#define CL_INVALID_WORK_DIMENSION -53\n\
#define CL_INVALID_WORK_GROUP_SIZE -54\n\
#define CL_INVALID_WORK_ITEM_SIZE -55\n\
#define CL_INVALID_GLOBAL_OFFSET -56\n\
#define CL_INVALID_EVENT_WAIT_LIST -57\n\
#define CL_INVALID_EVENT -58\n\
#define CL_INVALID_OPERATION -59\n\
#define CL_INVALID_GL_OBJECT -60\n\
#define CL_INVALID_BUFFER_SIZE -61\n\
#define CL_INVALID_MIP_LEVEL -62\n\
#define CL_INVALID_GLOBAL_WORK_SIZE -63 ";


void ctr_opencl_encrypt(unsigned int *, unsigned char *, unsigned char *, unsigned int);

#endif	/* OPENCL_CIPHER_H */
