#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <numeric> 
#include <algorithm> 
#include <cstdlib> // Para rand()
#include <chrono> // Para medir el tiempo de ejecución

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
void bitonicSort(int *data, int n, int threadsPerBlock, int numBlocks) {
    int *d_data;
    size_t size = n * sizeof(int);

    // Asignar memoria en el dispositivo
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);

    // Realizar el ordenamiento bitónico
    for (int step = 2; step <= n; step <<= 1) {
        for (int subStep = step >> 1; subStep > 0; subStep >>= 1) {
            bitonicSortKernel<<<numBlocks, threadsPerBlock>>>(d_data, n, step, subStep);
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
    vector<double> tiempos_Promedios;
    vector<int> nValues = {100, 1000, 10000, 100000, 1000000, 10000000};
    vector<int> nProofs;
    int numPruebas = 100;

    for(int j = 0; j < nValues.size(); j++){
        n = nValues[j];
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

            auto totalStart = chrono::high_resolution_clock::now();

            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            bitonicSort(d_data, n, nt, (n + nt - 1) / nt);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            auto totalEnd = chrono::high_resolution_clock::now();

            float elapsedTime = 0;
            cudaEventElapsedTime(&elapsedTime, start, stop);



            tiempos.push_back(chrono::duration<double>(totalEnd - totalStart).count());

            // Liberar memoria
            cudaFree(d_data);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        // Calcular tiempo promedio
        double tiempoPromedio = accumulate(tiempos.begin(), tiempos.end(), 0.0) / numPruebas;
        tiempos_Promedios.push_back(tiempoPromedio);
        nProofs.push_back(n);
    }

    // Guardar resultados en archivo CSV
    string nombreArchivo = "pruebas/tiempos_vs_n_cuda.csv";
    ofstream file(nombreArchivo, ios::out | ios::app);
    if (file.is_open()) {
        for (size_t i = 0; i < tiempos_Promedios.size(); ++i) {
            file << nProofs[i] << "," << tiempos_Promedios[i] << "\n";
        }
        file.close();
    } else {
        cerr << "No se pudo abrir el archivo para guardar los resultados.\n";
    }

    cout << tiempos_Promedios.size() << nProofs.size() << "\n";

    cout << "Resultados de tiempo vs n en GPU guardados en: " << nombreArchivo << "\n";
}



// Función para calcular y guardar speedup y eficiencia con múltiples pruebas
void calcularYGuardarSpeedupEficienciaGPU(int n, int nt, int numPruebas) {
    // Obtener el número de SMs de la GPU
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int numSMs = prop.multiProcessorCount;
    int maxBlocks = 5 * numSMs; // Rango hasta 5 veces el número de SMs

    vector<int> blockCounts;
    vector<double> speedups;
    vector<double> eficiencias;

    // Generar datos aleatorios iniciales
    vector<int> data(n);
    for (int i = 0; i < n; ++i) {
        data[i] = rand() % 100;
    }

    // Copiar datos a la GPU
    int *d_data;
    cudaMalloc((void **)&d_data, n * sizeof(int));
    cudaMemcpy(d_data, data.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    // Medir tiempo con 1 bloque para múltiples pruebas
    vector<float> tiempos1Bloque;
    for (int prueba = 0; prueba < numPruebas; ++prueba) {
        cudaMemcpy(d_data, data.data(), n * sizeof(int), cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        bitonicSort(d_data, n, nt, 1); // Ejecutar con 1 bloque
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        tiempos1Bloque.push_back(elapsedTime);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Calcular tiempo promedio con 1 bloque
    float timeWith1Block = accumulate(tiempos1Bloque.begin(), tiempos1Bloque.end(), 0.0f) / numPruebas;

    // Iterar para diferentes números de bloques
    for (int bloques = 1; bloques <= maxBlocks; ++bloques) {
        vector<float> tiemposConBloques;
        for (int prueba = 0; prueba < numPruebas; ++prueba) {
            cudaMemcpy(d_data, data.data(), n * sizeof(int), cudaMemcpyHostToDevice);

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            bitonicSort(d_data, n, nt, bloques); // Ejecutar con `bloques`
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float elapsedTime;
            cudaEventElapsedTime(&elapsedTime, start, stop);
            tiemposConBloques.push_back(elapsedTime);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        // Calcular tiempo promedio con `bloques`
        float timeWithTBlocks = accumulate(tiemposConBloques.begin(), tiemposConBloques.end(), 0.0f) / numPruebas;

        // Calcular speedup y eficiencia
        double speedup = timeWith1Block / timeWithTBlocks;
        double eficiencia = speedup / bloques;

        // Guardar resultados
        blockCounts.push_back(bloques);
        speedups.push_back(speedup);
        eficiencias.push_back(eficiencia);
    }

    // Liberar memoria
    cudaFree(d_data);

    // Guardar resultados en archivo CSV
    string nombreArchivo = "pruebas/speedup_eficiencia_promedios_cuda.csv";
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

    int numBlocks = (n + nt - 1) / nt;
    // Asegúrate de que n sea potencia de 2
    

    // Generar datos aleatorios
    vector<int> data(n);
    for (int i = 0; i < n; ++i) {
        data[i] = rand() % 100;
    }

    // Mostrar datos antes de ordenar
    /*cout << "Datos antes de ordenar:\n";
    for (int i : data) cout << i << " ";
    cout << "\n";*/
    cudaEvent_t start, stop;
    auto totalStart = chrono::high_resolution_clock::now();
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
// Código original
    cudaEventRecord(start, 0);
    bitonicSort(data.data(), n, nt, numBlocks);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    auto totalEnd = chrono::high_resolution_clock::now();

    float elapsed_time = 0;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    cout << "\n\n";
    cout << "Tiempo de ejecucion BitonicSort: " << chrono::duration<double>(totalEnd - totalStart).count() << " segundos\n";

    // Mostrar datos después de ordenar
    /*cout << "Datos después de ordenar:\n";
    for (int i : data) cout << i << " ";
    cout << "\n";*/

    //benchmarking y guardar tiempos


    //Pruebas benchmark

    //medirYGuardarTiemposGPU(10000000, 12);
    //calcularYGuardarSpeedupEficienciaGPU(1048576, 256, 100);


    return 0;

}
