#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdlib>
#include <sstream>
#include <iomanip>
#include <array>
#include <memory> // Para shared_ptr
#include <omp.h>



using namespace std;

// Función para ejecutar un comando de terminal y capturar su salida
string runCommand(const string &command) {
    array<char, 128> buffer;
    string result;

    // Renombrar la variable para evitar conflictos con `popen`
    shared_ptr<FILE> cmdPipe(popen(command.c_str(), "r"), pclose);
    if (!cmdPipe) {
        cerr << "Error: Failed to run command: " << command << endl;
        return "";
    }

    while (fgets(buffer.data(), buffer.size(), cmdPipe.get()) != nullptr) {
        result += buffer.data();
    }

    return result;
}

// Función para medir el tiempo de ejecución de un comando
double measureExecutionTime(const string &command) {
    auto start = chrono::high_resolution_clock::now();
    runCommand(command);
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double>(end - start).count();
}

// Función para generar un benchmark entre CPU y GPU
void CPU_vs_GPU(int n, int nt) {
    vector<pair<int, double>> cpuTimes;
    vector<pair<int, double>> gpuTimes;

    // Número de repeticiones por tamaño de entrada
    int repetitions = 100;

    cout << endl << endl << "Benchmarking CPU vs GPU..." << endl;
    cout << "Numero de pruebas: " << repetitions << endl;

    for (int i = 1000; i <= n; i *= 2) {
        double cpuTime = 0.0, gpuTime = 0.0;

        for (int j = 0; j < repetitions; ++j) {
            // Medir tiempo para CPU
            cpuTime += measureExecutionTime("./main_cpu " + to_string(i) + " " + to_string(nt));

            // Medir tiempo para GPU
            gpuTime += measureExecutionTime("./main_cuda " + to_string(i) + " " + to_string(nt));
        }

        // Promediar tiempos
        cpuTime /= repetitions;
        gpuTime /= repetitions;

        cpuTimes.emplace_back(i, cpuTime);
        gpuTimes.emplace_back(i, gpuTime);

        cout << "n=" << i << " | CPU: " << cpuTime << "s, GPU: " << gpuTime << "| Repeticiones: " << repetitions << "s\n";
    }

    // Guardar resultados en archivos CSV
    ofstream cpuFile("cpu_vs_gpu_cpu.csv"), gpuFile("cpu_vs_gpu_gpu.csv");
    cpuFile << "n,Time(s)\n";
    gpuFile << "n,Time(s)\n";

    for (size_t k = 0; k < cpuTimes.size(); ++k) {
        cpuFile << cpuTimes[k].first << "," << cpuTimes[k].second << "\n";
        gpuFile << gpuTimes[k].first << "," << gpuTimes[k].second << "\n";
    }

    cpuFile.close();
    gpuFile.close();

    cout << "Results saved to cpu_vs_gpu_cpu.csv and cpu_vs_gpu_gpu.csv\n";
}

int main(int argc, char **argv) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <number of elements> <mode (0=CPU, 1=GPU)> <number of threads>\n";
        return 1;
    }

    int n = stoi(argv[1]);
    int modo = stoi(argv[2]);
    int nt = stoi(argv[3]);

    if (n <= 0 || nt <= 0 || (modo != 0 && modo != 1)) {
        cerr << "Invalid arguments. Ensure n > 0, nt > 0, and mode is 0 or 1.\n";
        return 1;
    }

    string program;

    if(modo == 0) {
        program = "./main_cpu";
        cout << "CPU mode selected\n";
        cout << "Executing CPU mode...\n";
    } else {
        program = "./main_cuda";
        cout << "GPU mode selected\n";
        cout << "Executing GPU mode...\n";
    }

    // Construir el comando
    ostringstream command;
    command << program << " " << n << " " << nt;

    // Medir tiempo de ejecución
    double elapsed = measureExecutionTime(command.str());
    cout << "Execution time: " << elapsed << " seconds\n";


    cout << "\n==============================================\n";

    // Realizar benchmarks
    CPU_vs_GPU(n, nt);

    return 0;
}
