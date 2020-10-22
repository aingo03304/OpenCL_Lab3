
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
