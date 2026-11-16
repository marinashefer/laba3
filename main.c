#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <locale.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#endif

char* read_kernel_file(const char* filename, size_t* size) {
    FILE* file = fopen(filename, "rb");
    if (!file) { printf("cannot open %s\n", filename); exit(1); }
    fseek(file, 0, SEEK_END);
    *size = ftell(file);
    rewind(file);
    char* content = (char*)malloc(*size + 1);
    fread(content, 1, *size, file);
    content[*size] = '\0';
    fclose(file);
    return content;
}

// CPU-реализация определителя
int factorial(int n) { int res = 1; for (int i = 2; i <= n; i++) res *= i; return res; }

void index_to_perm_cpu(int idx, int n, int* perm) {
    int avail[16]; for (int i = 0; i < n; i++) avail[i] = i;
    for (int i = 0; i < n; i++) {
        int f = factorial(n - 1 - i); int pos = idx / f; idx %= f;
        perm[i] = avail[pos];
        for (int j = pos; j < n - 1 - i; j++) avail[j] = avail[j + 1];
    }
}

int count_inv_cpu(int* perm, int n) {
    int inv = 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (perm[i] > perm[j]) inv++;
    return inv;
}

double det_cpu(float* A, int n) {
    int total = factorial(n);
    double res = 0.0;
    int perm[16];
    for (int i = 0; i < total; i++) {
        index_to_perm_cpu(i, n, perm);
        int inv = count_inv_cpu(perm, n);
        double prod = 1.0;
        for (int j = 0; j < n; j++) prod *= A[j * n + perm[j]];
        res += (inv % 2 != 0) ? -prod : prod;
    }
    return res;
}

int main() {
    setlocale(LC_ALL, "ru_RU.UTF-8");
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel_det;

    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_queue_properties props[] = {0};
    queue = clCreateCommandQueueWithProperties(context, device, props, &err);

    size_t kernel_size;
    char* kernel_source = read_kernel_file("kernel.cl", &kernel_size);
    program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_size, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("ошибка компиляции ядра!\n");
        return 1;
    }
    kernel_det = clCreateKernel(program, "det_leibniz", &err);
    free(kernel_source);

    printf("OpenCL initialized successfully on AMD GPU!\n\n");

    // задача 1: определитель матрицы
    int N = 8;
    int total_perms = factorial(N);
    float* h_A = (float*)malloc(N * N * sizeof(float));
    srand(42);
    for (int i = 0; i < N * N; i++) h_A[i] = (float)(rand() % 100) / 10.0f;

    clock_t start_cpu = clock();
    double res_cpu = det_cpu(h_A, N);
    clock_t end_cpu = clock();
    printf("[CPU] определитель = %f, время = %f сек\n",
           res_cpu, (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC);

    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                N * N * sizeof(float), h_A, &err);
    cl_mem d_results = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                      total_perms * sizeof(float), NULL, &err);

    clSetKernelArg(kernel_det, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel_det, 1, sizeof(int), &N);
    clSetKernelArg(kernel_det, 2, sizeof(int), &total_perms);
    clSetKernelArg(kernel_det, 3, sizeof(cl_mem), &d_results);

    size_t global_size = total_perms;
    clock_t start_gpu = clock();
    clEnqueueNDRangeKernel(queue, kernel_det, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    clFinish(queue);
    clock_t end_gpu = clock();

    float* h_results = (float*)malloc(total_perms * sizeof(float));
    clEnqueueReadBuffer(queue, d_results, CL_TRUE, 0,
                        total_perms * sizeof(float), h_results, 0, NULL, NULL);

    double res_gpu = 0.0;
    for (int i = 0; i < total_perms; i++) res_gpu += h_results[i];
    printf("[GPU] определитель = %f, время = %f сек\n",
           res_gpu, (double)(end_gpu - start_gpu) / CLOCKS_PER_SEC);

    double diff = fabs(res_cpu - res_gpu);
    double rel_err = diff / fabs(res_cpu);
    printf("абсолютная разница: %f\n", diff);
    printf("относительная погрешность: %e\n", rel_err);
    printf("match: %s\n\n", rel_err < 1e-4 ? "YES" : "NO");

    FILE* f = fopen("matrix.txt", "w");
    if (f) {
        fprintf(f, "размер матрицы: %d x %d\n\n", N, N);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++)
                fprintf(f, "%8.4f ", h_A[i * N + j]);
            fprintf(f, "\n");
        }
        fprintf(f, "\nопределитель CPU: %f\n", res_cpu);
        fprintf(f, "определитель GPU: %f\n", res_gpu);
        fclose(f);
    }

    // задача 2: брутфорс MD5

    // очистка
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_results);
    clReleaseKernel(kernel_det);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(h_A);
    free(h_results);
    return 0;
}