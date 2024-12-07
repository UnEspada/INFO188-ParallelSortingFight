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

// Función para ejecutar un comando y capturar su salida, imprimiéndola directamente
string runCommand(const string &command) {
    array<char, 128> buffer;
    string result;

    // Renombrar la variable para evitar conflictos con `popen`
    shared_ptr<FILE> cmdPipe(popen(command.c_str(), "r"), pclose);
    if (!cmdPipe) {
        cerr << "Error: Failed to run command: " << command << endl;
        return "";
    }

    // Leer la salida del comando y mostrarla en tiempo real
    while (fgets(buffer.data(), buffer.size(), cmdPipe.get()) != nullptr) {
        cout << buffer.data(); // Imprimir en consola
        result += buffer.data(); // Guardar en la cadena result
    }

    return result;
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

    if (modo == 0) {
        program = "./main_cpu";
        cout << "CPU mode selected\n";
        cout << "Executing CPU mode...\n";
        runCommand(program + " " + to_string(n) + " " + to_string(nt));
    } else {
        program = "./main_cuda";
        cout << "GPU mode selected\n";
        cout << "Executing GPU mode...\n";
        runCommand(program + " " + to_string(n) + " " + to_string(nt));
    }

    cout << "\n===============================================\n";
    return 0;
}
