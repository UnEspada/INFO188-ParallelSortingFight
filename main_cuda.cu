#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <numeric> 
#include <algorithm> 
#include <cstdlib> // Para rand()

using namespace std;

// CUDA kernel para Bitonic Sort
__global__ void bitonicSortKernel(int *data, int n, int step, int subStep) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    int ixj = idx ^ subStep;

    if (ixj < n && idx < ixj) { // Validar índices y asegurar comparaciones válidas
        if ((idx & step) == 0) {
            // Ascendente
            if (data[idx] > data[ixj]) {
                int temp = data[idx];
                data[idx] = data[ixj];
                data[ixj] = temp;
            }
        } else {
            // Descendente
            if (data[idx] < data[ixj]) {
                int temp = data[idx];
                data[idx] = data[ixj];
                data[ixj] = temp;
            }
        }
    }
}

// Función para Bitonic Sort en la GPU
void bitonicSort(int *data, int n, int threadsPerBlock) {
    int *d_data;
    size_t size = n * sizeof(int);

    // Asignar memoria en el dispositivo
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);

    // Configurar grid y bloques
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    // Realizar el ordenamiento bitónico
    for (int step = 2; step <= n; step <<= 1) {
        for (int subStep = step >> 1; subStep > 0; subStep >>= 1) {
            bitonicSortKernel<<<blocks, threadsPerBlock>>>(d_data, n, step, subStep);
            cudaDeviceSynchronize(); // Sincronizar GPU después de cada fase
        }
    }

    // Copiar los datos de vuelta al host
    cudaMemcpy(data, d_data, size, cudaMemcpyDeviceToHost);

    // Liberar memoria
    cudaFree(d_data);
}

void medirYGuardarTiemposGPU(int n, int nt) {
    vector<double> tiempos;
    int numPruebas = 100;

    for (int i = 0; i < numPruebas; ++i) {
        // Generar datos aleatorios
        vector<int> data(n);
        for (int j = 0; j < n; ++j) {
            data[j] = rand() % 100;
        }

        // Copiar datos a la GPU
        int* d_data;
        cudaMalloc((void**)&d_data, n * sizeof(int));
        cudaMemcpy(d_data, data.data(), n * sizeof(int), cudaMemcpyHostToDevice);

        // Medir tiempo
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        bitonicSort<<<(n + nt - 1) / nt, nt>>>(d_data, n, nt, (n + nt - 1) / nt);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);

        tiempos.push_back(elapsedTime);

        // Liberar memoria
        cudaFree(d_data);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Calcular tiempo promedio
    double tiempoPromedio = accumulate(tiempos.begin(), tiempos.end(), 0.0) / numPruebas;

    // Guardar resultados en archivo CSV
    string nombreArchivo = "pruebas/tiempos_vs_n_cuda.csv";
    ofstream file(nombreArchivo, ios::out | ios::app);
    if (file.is_open()) {
        file << n << "," << tiempoPromedio << "\n";
        file.close();
    } else {
        cerr << "No se pudo abrir el archivo para guardar los resultados.\n";
    }

    cout << "Resultados de tiempo vs n en GPU guardados en: " << nombreArchivo << "\n";
}


void calcularYGuardarSpeedupEficienciaGPU(int n, int nt) {
    vector<int> blockCounts;
    vector<double> speedups;
    vector<double> eficiencias;

    for (int bloques = 1; bloques <= nt; ++bloques) {
        // Generar datos aleatorios
        vector<int> data(n);
        for (int i = 0; i < n; ++i) {
            data[i] = rand() % 100;
        }

        // Copiar datos a la GPU
        int* d_data;
        cudaMalloc((void**)&d_data, n * sizeof(int));
        cudaMemcpy(d_data, data.data(), n * sizeof(int), cudaMemcpyHostToDevice);

        // Medir tiempo con 1 bloque
        cudaEvent_t start1, stop1;
        cudaEventCreate(&start1);
        cudaEventCreate(&stop1);

        cudaEventRecord(start1);
        bitonicSort<<<1, n>>>(d_data, n, 1, 1);
        cudaEventRecord(stop1);
        cudaEventSynchronize(stop1);

        float timeWith1Block;
        cudaEventElapsedTime(&timeWith1Block, start1, stop1);

        // Medir tiempo con `bloques`
        cudaMemcpy(d_data, data.data(), n * sizeof(int), cudaMemcpyHostToDevice);
        cudaEvent_t startT, stopT;
        cudaEventCreate(&startT);
        cudaEventCreate(&stopT);

        cudaEventRecord(startT);
        bitonicSort<<<bloques, n / bloques>>>(d_data, n, n / bloques, bloques);
        cudaEventRecord(stopT);
        cudaEventSynchronize(stopT);

        float timeWithTBlocks;
        cudaEventElapsedTime(&timeWithTBlocks, startT, stopT);

        // Calcular speedup y eficiencia
        double speedup = timeWith1Block / timeWithTBlocks;
        double eficiencia = speedup / bloques;

        // Guardar resultados
        blockCounts.push_back(bloques);
        speedups.push_back(speedup);
        eficiencias.push_back(eficiencia);

        // Liberar memoria
        cudaFree(d_data);
        cudaEventDestroy(start1);
        cudaEventDestroy(stop1);
        cudaEventDestroy(startT);
        cudaEventDestroy(stopT);
    }

    // Guardar resultados en archivo CSV
    string nombreArchivo = "pruebas/speedup_eficiencia_cuda.csv";
    ofstream file(nombreArchivo, ios::out | ios::app);
    if (file.is_open()) {
        file << "Bloques,Speedup,Eficiencia\n"; // Encabezado
        for (size_t i = 0; i < blockCounts.size(); ++i) {
            file << blockCounts[i] << "," << speedups[i] << "," << eficiencias[i] << "\n";
        }
        file.close();
    } else {
        cerr << "No se pudo abrir el archivo para guardar los resultados.\n";
    }

    cout << "Resultados de speedup y eficiencia paralela en GPU guardados en: " << nombreArchivo << "\n";
}


int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Usage: %s <number of elements> <number of threads>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int nt = atoi(argv[2]);

    // Asegúrate de que n sea potencia de 2
    

    // Generar datos aleatorios
    vector<int> data(n);
    for (int i = 0; i < n; ++i) {
        data[i] = rand() % 100;
    }

    // Mostrar datos antes de ordenar
    cout << "Datos antes de ordenar:\n";
    for (int i : data) cout << i << " ";
    cout << "\n";

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // Ordenar usando Bitonic Sort
    
    bitonicSort(data.data(), n, nt);

    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed_time = 0;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    // Mostrar datos después de ordenar
    cout << "Datos después de ordenar:\n";
    for (int i : data) cout << i << " ";
    cout << "\n";

    cout << "\n\n";
    cout << "Tiempo de ejecucion BitonicSort: " << elapsed_time << "s\n";

    //benchmarking y guardar tiempos


    medirYGuardarTiemposGPU(n, nt);
    calcularYGuardarSpeedupEficienciaGPU(n, nt);


    return 0;

}
