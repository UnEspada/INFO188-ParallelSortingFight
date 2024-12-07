# Variables de compilación
CXX = g++
NVCC = nvcc
CXXFLAGS = -O3 -fopenmp
NVCCFLAGS = -std=c++11

# Archivos fuente y ejecutables
CPU_SRC = main_cpu.cpp
CUDA_SRC = main_cuda.cu
MAIN_SRC = main.cpp
CPU_EXE = main_cpu
CUDA_EXE = main_cuda
MAIN_EXE = main

# Regla por defecto
all: $(CPU_EXE) $(CUDA_EXE) $(MAIN_EXE)

# Compilar main_cpu
$(CPU_EXE): $(CPU_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

# Compilar main_cuda
$(CUDA_EXE): $(CUDA_SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Compilar main
$(MAIN_EXE): $(MAIN_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

# Limpieza de los ejecutables
tidy:
	rm -f $(CPU_EXE) $(CUDA_EXE) $(MAIN_EXE)

# Limpieza completa
clean: tidy
	rm -rf *.o