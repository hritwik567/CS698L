#!/bin/bash
./p3 4096 2 1>out_4096_2
./p3 4096 4 1>out_4096_4
./p3 4096 8 1>out_4096_8
./p3 4096 16 1>out_4096_16
./p3 256 1 1>out_256_1
./p3 512 2 1>out_512_2
./p3 512 4 1>out_512_4
./p3 1024 8 1>out_1024_8
./p3 1024 16 1>out_1024_16
./p3 2048 16 1>out_2048_16
./p3 4096 16 1>out_4096_16