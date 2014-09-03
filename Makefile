
test: cgls_test.cu cgls.cuh
	nvcc -O3 -m64 -arch=sm_30 -o $@ -lcublas -lcusparse -L/usr/local/cuda/lib64/ $<
	./test

clean:
	rm -rf *.o test
