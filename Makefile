# Makefile para compilar y ejecutar el programa de comparación entre Bitonic Sort y Merge Sort

# Compiladores
NVCC = nvcc
CXX = g++

# Flags de compilación
CXXFLAGS = -O3 -fopenmp
NVFLAGS = -O3 -arch=sm_60

# Archivos
TARGET = prog
OBJ_CUDA = main_cuda.o
OBJ_CPP = main_cpu.o

# Regla principal
all: $(TARGET)

$(OBJ_CUDA): main_cuda.cu
	$(NVCC) -c main_cuda.cu $(NVFLAGS) -o $(OBJ_CUDA)

$(OBJ_CPP): main_cpu.cpp
	$(CXX) -c main_cpu.cpp $(CXXFLAGS) -o $(OBJ_CPP)

$(TARGET): $(OBJ_CUDA) $(OBJ_CPP)
	$(CXX) $(OBJ_CUDA) $(OBJ_CPP) $(CXXFLAGS) -o $(TARGET)

# Limpiar archivos generados
clean:
	rm -f $(TARGET) *.o *.csv

# Uso del programa
run:
	./$(TARGET) 1000000 0 8 # Cambiar parámetros según necesidad
