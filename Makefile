CULDFLAGS=-L/usr/local/cuda/lib

test: cgls_test.cu cgls.cuh
	nvcc -O3 -m64 -arch=sm_20 -o $@ -lcublas -lcusparse $(CULDFLAGS) $<
	./test

clean:
	rm -rf *.o test
