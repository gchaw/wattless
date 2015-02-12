#!/bin/bash

# Generate random files of sizes 128K, 256K, 512K, 1M, 2M, 5M, 10M, 20M
`dd if=/dev/urandom of=test128KB.rand bs=1024 count=128`
`dd if=/dev/urandom of=test256KB.rand bs=1024 count=256`
`dd if=/dev/urandom of=test512KB.rand bs=1024 count=512`
`dd if=/dev/urandom of=test1MB.rand bs=1048576 count=1`
`dd if=/dev/urandom of=test2MB.rand bs=1048576 count=2`
`dd if=/dev/urandom of=test5MB.rand bs=1048576 count=5`
`dd if=/dev/urandom of=test10MB.rand bs=1048576 count=10`
`dd if=/dev/urandom of=test20MB.rand bs=1048576 count=20`
