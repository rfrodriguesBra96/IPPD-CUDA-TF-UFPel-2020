#targets

COMPILE = nvcc
CPP = nbody.cpp
FILE_NAME = main.cu
FILE_NAME_OUT = exe

All:
	$(COMPILE) $(FILE_NAME) $(CPP) -o $(FILE_NAME_OUT)
Clean:
	rm-f $(FILE_NAME_OUT)

