// задача 1: определитель матрицы по формуле лейбница

int factorial(int n) {
    int res = 1;
    for (int i = 2; i <= n; i++) res *= i;
    return res;
}

// avail - приватный массив
void index_to_perm(int idx, int n, int* avail, int* perm_out) {
    for (int i = 0; i < n; i++) avail[i] = i;
    for (int i = 0; i < n; i++) {
        int f = factorial(n - 1 - i);
        int pos = idx / f;
        idx %= f;
        perm_out[i] = avail[pos];
        for (int j = pos; j < n - 1 - i; j++)
            avail[j] = avail[j + 1];
    }
}

int count_inversions(int* perm, int n) {
    int inv = 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (perm[i] > perm[j]) inv++;
    return inv;
}

__kernel void det_leibniz(__global float* A, int n, int total_perms, __global float* results) {
    int idx = get_global_id(0);
    if (idx >= total_perms) return;

    int avail[16];
    int perm[16];
    index_to_perm(idx, n, avail, perm);
    int inv = count_inversions(perm, n);

    float prod = 1.0f;
    for (int j = 0; j < n; j++)
        prod *= A[j * n + perm[j]];

    results[idx] = (inv % 2 != 0) ? -prod : prod;
}

// задача 2: брутфорс MD5