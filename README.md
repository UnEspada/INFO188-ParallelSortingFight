## Instrucciones de compilado y ejecucion


- **Compilar todo con make**
- **Si desea compilar por separado:**
  - main_cuda.cpp: g++ main_cpu.cpp -o main_cpu -O3 -fopenmp
  - main_cuda.cu: nvcc -std=c++11 -o main_cuda main_cuda.cu
  - main.cpp: g++ main.cpp -o main -O3 -fopenmp
- Por ultimo en cada archivo especializado, es decir, main_cpu y main_cuda en sus respectivos mains hay lineas comentadas que sirven para el benchmark y generacion de tablas csv para su posterior analisis con jupyter (cabe resaltar que se debe tener instalado jupyter), para instalar jupyter debe ejecutar los siguientes comandos:
  - **Windows:**

    - pip install jupyter notebook matplotlib numpy scikit-image numba imageio pyqt5
  - **Ubuntu/Linux:**

    - sudo apt update
    - sudo apt install jupyter-notebook
    - sudo apt install python3 python3-pip
    - pip install --user jupyter matplotlib numpy scikit-image numba imageio pyqt5
