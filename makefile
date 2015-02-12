# File:		makefile
# Author:	syrowikb
# Created:	November 29 2013


# go into source directory, and call that makefile.
# create symbolic links to the aes program and the OpenCL kernel
all:
	cd src; make; cd ..; \
		if [ ! -f aes ] ; then ln -s src/aes ./; fi; \
		if [ ! -f opencl_cipher.cl ] ; then ln -s src/opencl_cipher.cl ./; fi;


# remove all build files
clean:
	rm aes; rm opencl_cipher.cl; cd src; make clean


# test cases for the cpu implementation
# encrypt and decrypt the james_watt.ppm image
.PHONY: cpu_encrypt
cpu_encrypt:
	./aes -i james_watt.ppm -o james_watt_cpu_encrypted.ppm -k test.key -c test.ctr -s

.PHONY: cpu_decrypt
cpu_decrypt: 
	./aes -i james_watt_cpu_encrypted.ppm -o james_watt_cpu_decrypted.ppm -k test.key -c test.ctr -s -d

.PHONY: cpu_run
cpu_run: cpu_encrypt cpu_decrypt


# encrypt and decrypt the large (5Mb) james_watt_large.ppm image using the CPU
.PHONY: cpu_encrypt_large
cpu_encrypt_large:
	./aes -i james_watt_large.ppm -o james_watt_large_cpu_encrypted.ppm -k test.key -c test.ctr -s

.PHONY: cpu_decrypt_large
cpu_decrypt_large: 
	./aes -i james_watt_large_cpu_encrypted.ppm -o james_watt_large_cpu_decrypted.ppm -k test.key -c test.ctr -s -d

.PHONY: cpu_run_large
cpu_run_large: cpu_encrypt_large cpu_decrypt_large


# encrypt and decrypt the huge (21Mb) james_watt_huge.ppm image using the CPU
.PHONY: cpu_encrypt_huge
cpu_encrypt_huge:
	./aes -i james_watt_huge.ppm -o james_watt_huge_cpu_encrypted.ppm -k test.key -c test.ctr -s

.PHONY: cpu_decrypt_huge
cpu_decrypt_huge: 
	./aes -i james_watt_huge_cpu_encrypted.ppm -o james_watt_huge_cpu_decrypted.ppm -k test.key -c test.ctr -s -d

.PHONY: cpu_run_huge
cpu_run_huge: cpu_encrypt_huge cpu_decrypt_huge




# test cases for the opencl implementation
# encrypt and decrypt the james_watt.ppm image
.PHONY: opencl_encrypt
opencl_encrypt:
	./aes -i james_watt.ppm -o james_watt_ocl_encrypted.ppm -k test.key -c test.ctr -s -m opencl

.PHONY: opencl_decrypt
opencl_decrypt: 
	./aes -i james_watt_ocl_encrypted.ppm -o james_watt_ocl_decrypted.ppm -k test.key -c test.ctr -s -m opencl -d

.PHONY: opencl_run
opencl_run: opencl_encrypt opencl_decrypt


# encrypt and decrypt the large (5Mb) james_watt_large.ppm image using OpenCL
.PHONY: opencl_encrypt_large
opencl_encrypt_large:
	./aes -i james_watt_large.ppm -o james_watt_large_ocl_encrypted.ppm -k test.key -c test.ctr -s -m opencl

.PHONY: opencl_decrypt_large
opencl_decrypt_large: 
	./aes -i james_watt_large_ocl_encrypted.ppm -o james_watt_large_ocl_decrypted.ppm -k test.key -c test.ctr -s -m opencl -d

.PHONY: opencl_run_large
opencl_run_large: opencl_encrypt_large opencl_decrypt_large

# encrypt and decrypt the huge (21Mb) james_watt_huge.ppm image using OpenCL
.PHONY: opencl_encrypt_huge
opencl_encrypt_huge:
	./aes -i james_watt_huge.ppm -o james_watt_huge_ocl_encrypted.ppm -k test.key -c test.ctr -s -m opencl

.PHONY: opencl_decrypt_huge
opencl_decrypt_huge: 
	./aes -i james_watt_huge_ocl_encrypted.ppm -o james_watt_huge_ocl_decrypted.ppm -k test.key -c test.ctr -s -m opencl -d

.PHONY: opencl_run_huge
opencl_run_huge: opencl_encrypt_huge opencl_decrypt_huge




# test cases for the CUDA implementation
.PHONY: cuda_encrypt
cuda_encrypt:
	./aes -i james_watt.ppm -o james_watt_cuda_encrypted.ppm -k test.key -c test.ctr -s -m cuda

.PHONY: cuda_decrypt
cuda_decrypt: 
	./aes -i james_watt_cuda_encrypted.ppm -o james_watt_cuda_decrypted.ppm -k test.key -c test.ctr -s -m cuda -d

.PHONY: cuda_run
cuda_run: cuda_encrypt cuda_decrypt


# encrypt and decrypt the large (5Mb) james_watt_large.ppm image using CUDA
.PHONY: cuda_encrypt_large
cuda_encrypt_large:
	./aes -i james_watt_large.ppm -o james_watt_large_cuda_encrypted.ppm -k test.key -c test.ctr -s -m cuda

.PHONY: cuda_decrypt_large
cuda_decrypt_large: 
	./aes -i james_watt_large_cuda_encrypted.ppm -o james_watt_large_cuda_decrypted.ppm -k test.key -c test.ctr -s -m cuda -d

.PHONY: cuda_run_large
cuda_run_large: cuda_encrypt_large cuda_decrypt_large

# encrypt and decrypt the huge (21Mb) james_watt_huge.ppm image using CUDA
.PHONY: cuda_encrypt_huge
cuda_encrypt_huge:
	./aes -i james_watt_huge.ppm -o james_watt_huge_cuda_encrypted.ppm -k test.key -c test.ctr -s -m cuda

.PHONY: cuda_decrypt_huge
cuda_decrypt_huge: 
	./aes -i james_watt_huge_cuda_encrypted.ppm -o james_watt_huge_cuda_decrypted.ppm -k test.key -c test.ctr -s -m cuda -d

.PHONY: cuda_run_huge
cuda_run_huge: cuda_encrypt_huge cuda_decrypt_huge
