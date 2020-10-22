
NVCC=nvcc
INCLUDE= -I../libgputk
LIBgpuTK= -L../build -lgputk -lcuda
LIBS= -lOpenCL $(LIBgpuTK)

all: template


template:
	$(NVCC) -std=c++11 template.cpp $(INCLUDE) $(LIBS) -o Tiled1DConvolution_Template 


dataset_generator:
	$(NVCC) -std=c++11 dataset_generator.cpp $(INCLUDE) $(LIBS) -o dataset_generator


clean:
	rm -f Tiled1DConvolution_Template dataset_generator

test:
	$(NVCC) -std=c++11 template.cpp $(INCLUDE) $(LIBS) -o Tiled1DConvolution_Template
	for n in $$(seq 0 9) ; \
	do \
		./Tiled1DConvolution_Template \
			-i 1DConvolution/Dataset/$$n/input0.raw,1DConvolution/Dataset/$$n/input1.raw \
			-e 1DConvolution/Dataset/$$n/output.raw -o 1DConvolution/Dataset/$$n/infer.raw -t vector ; \
	done
