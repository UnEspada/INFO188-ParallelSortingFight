#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <string>
#include <ctime>  // Para obtener el tiempo actual y usarlo como parte del nombre del archivo
#include <sstream>

using namespace std;


void guardarResultadosCSV(const string &nombreArchivo, const vector<double> &tiempos, const vector<int> &nValues) {
    ofstream file(nombreArchivo, ios::out | ios::app);  // Abrir en modo append
    if (file.is_open()) {
        for (size_t i = 0; i < nValues.size(); ++i) {
            file << nValues[i] << "," << tiempos[i] << "\n";  // Guardar n y tiempo
        }
        file.close();
    } else {
        cerr << "No se pudo abrir el archivo para guardar los resultados.\n";
    }
}


// Función para mezclar dos mitades ordenadas
void merge(vector<int> &data, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; ++i) L[i] = data[left + i];
    for (int i = 0; i < n2; ++i) R[i] = data[mid + 1 + i];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            data[k] = L[i];
            ++i;
        } else {
            data[k] = R[j];
            ++j;
        }
        ++k;
    }

    while (i < n1) {
        data[k] = L[i];
        ++i;
        ++k;
    }

    while (j < n2) {
        data[k] = R[j];
        ++j;
        ++k;
    }
}

// Función recursiva de MergeSort
void mergeSort(vector<int> &data, int left, int right, int depth = 0) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        // Paralelizar las llamadas recursivas utilizando OpenMP
        #pragma omp parallel sections if(depth < 4) // Controlar paralelismo a cierto nivel
        {
            #pragma omp section
            mergeSort(data, left, mid, depth + 1);

            #pragma omp section
            mergeSort(data, mid + 1, right, depth + 1);
        }

        // Mezclar las dos mitades ordenadas
        merge(data, left, mid, right);
    }
}


void ejecutarPruebas(int n, int nt) {
    vector<double> tiempos;  // Para almacenar los tiempos de ejecución
    double tiempo = 0;
    vector<int> nValues;  // Para almacenar los valores de n
    vector<int> tiemposPromedio;
    vector<int> valoresN = {100, 1000, 10000, 100000, 1000000, 10000000};
    for(int j = 0 ; j < valoresN.size(); j++){
        n = valoresN[j];
        for (int i = 0; i < 100; i++) {
            // Generar los datos aleatorios para cada ejecución
            vector<int> data(n);
            for (int z = 0; z < n; ++z) {
                data[z] = rand() % 100;  // Puedes cambiar el rango si es necesario
            }

            // Iniciar el temporizador para esta ejecución
            auto start = omp_get_wtime();

            // Ejecutar el mergeSort con OpenMP
            #pragma omp parallel
            {
                #pragma omp single
                mergeSort(data, 0, n - 1);  // Llamada al mergeSort paralelo
            }

            // Detener el temporizador y sumar el tiempo de ejecución
            auto end = omp_get_wtime();
            double execTime = end - start;  // Tiempo de ejecución de esta iteración
            tiempo += execTime;
            //tiempos.push_back(execTime);  // Acumular el tiempo
        }
        tiempos.push_back(tiempo/100);
        nValues.push_back(n);
    }
    // Guardar los resultados en un archivo CSV
    string nombreArchivo = "pruebas/tiempos_vs_n_openmp.csv";
    guardarResultadosCSV(nombreArchivo, tiempos, nValues);
    cout << "Resultados guardados en: " << nombreArchivo << "\n";
}



void calcularYGuardarSpeedupEficiencia(int n, int maxThreads) {
    vector<int> threadCounts(maxThreads);
    vector<double> averageSpeedups(maxThreads, 0.0);
    vector<double> averageEficiencia(maxThreads, 0.0);

    int numPruebas = 100; // Número de pruebas para promediar

    for (int prueba = 0; prueba < numPruebas; ++prueba) {
        for (int t = 1; t <= maxThreads; t++) {
            // Configurar el número de threads de OpenMP
            omp_set_num_threads(t);

            // Generar datos aleatorios
            vector<int> data(n);
            for (int j = 0; j < n; ++j) {
                data[j] = rand() % 100;
            }

            // Medir el tiempo con 1 thread
            omp_set_num_threads(1);
            auto start1 = omp_get_wtime();
            #pragma omp parallel
            {
                #pragma omp single
                mergeSort(data, 0, n - 1);
            }
            auto end1 = omp_get_wtime();
            double timeWith1Thread = end1 - start1;

            // Medir el tiempo con t threads
            omp_set_num_threads(t); // Volver a configurar los threads
            auto startT = omp_get_wtime();
            #pragma omp parallel
            {
                #pragma omp single
                mergeSort(data, 0, n - 1);
            }
            auto endT = omp_get_wtime();
            double timeWithTThreads = endT - startT;

            // Calcular speedup y eficiencia paralela
            double speedup = timeWith1Thread / timeWithTThreads;
            double eficiencia = speedup / t;

            // Acumular resultados para calcular el promedio
            averageSpeedups[t - 1] += speedup;
            averageEficiencia[t - 1] += eficiencia;

            // Registrar el número de threads una vez
            if (prueba == 0) {
                threadCounts[t - 1] = t;
            }
        }
    }

    // Calcular promedios
    for (int i = 0; i < maxThreads; ++i) {
        averageSpeedups[i] /= numPruebas;
        averageEficiencia[i] /= numPruebas;
    }

    // Guardar los resultados en un archivo CSV
    string nombreArchivo = "pruebas/speedup_eficiencia_openmp.csv";
    ofstream file(nombreArchivo, ios::out | ios::app);
    if (file.is_open()) {
        file << "Threads,Speedup,Eficiencia\n"; // Encabezado
        for (size_t i = 0; i < threadCounts.size(); ++i) {
            file << threadCounts[i] << "," << averageSpeedups[i] << "," << averageEficiencia[i] << "\n";
        }
        file.close();
    } else {
        cerr << "No se pudo abrir el archivo para guardar los resultados.\n";
    }

    cout << "Resultados de speedup y eficiencia guardados en: " << nombreArchivo << "\n";
}


int main(int argc, char **argv) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <number of elements> <number of threads>\n";
        return 1;
    }

    int n = atoi(argv[1]);
    int nt = atoi(argv[2]);

    if (n <= 0 || nt <= 0) {
        cerr << "El número de elementos y el número de threads deben ser positivos.\n";
        return 1;
    }

    // Configurar el número de threads de OpenMP
    omp_set_num_threads(nt);

    // Generar datos aleatorios
    vector<int> data(n);
    for (int i = 0; i < n; ++i) {
        data[i] = rand() % 100;
    }

    // Mostrar datos antes de ordenar
    /*cout << "Datos antes de ordenar:\n";
    for (int i : data) cout << i << " ";*/
    cout << "\n";

    // Ordenar usando MergeSort con OpenMP

    auto start = omp_get_wtime(); // Iniciar el temporizador

    #pragma omp parallel
    {
        #pragma omp single
        mergeSort(data, 0, n - 1);
    }

    auto end = omp_get_wtime(); // Finalizar el temporizador

    // Mostrar datos después de ordenar
    /*cout << "Datos después de ordenar:\n";
    for (int i : data) cout << i << " ";*/
    cout << "\n\n";
    cout << "Tiempo de ejecucion mergeSort: " << end - start << "s\n";


    //Pruebas benchmark

    //ejecutarPruebas(1000000, 12); //el primer valor es el n y el segundo threads (colocar threads segun pc)
    //calcularYGuardarSpeedupEficiencia(100000, 12); //lo mismo que el anterior (colocar threads segun pc)


    return 0;
}

