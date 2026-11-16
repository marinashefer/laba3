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
#define F(x, y, z) ((z) ^ ((x) & ((y) ^ (z))))
#define G(x, y, z) ((y) ^ ((z) & ((x) ^ (y))))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | ~(z)))

#define STEP(f, a, b, c, d, x, t, s) \
    (a) += f((b), (c), (d)) + (x) + (t); \
    (a) = (((a) << (s)) | (((a) & 0xffffffff) >> (32 - (s)))); \
    (a) += (b);

void md5_transform(unsigned int* state, unsigned char* block) {
    unsigned int a = state[0], b = state[1], c = state[2], d = state[3];
    unsigned int x[16];
    for (int i = 0; i < 16; i++)
        x[i] = ((unsigned int)block[i*4]) | (((unsigned int)block[i*4+1])<<8) |
               (((unsigned int)block[i*4+2])<<16) | (((unsigned int)block[i*4+3])<<24);

    STEP(F, a, b, c, d, x[ 0], 0xd76aa478,  7) STEP(F, d, a, b, c, x[ 1], 0xe8c7b756, 12)
    STEP(F, c, d, a, b, x[ 2], 0x242070db, 17) STEP(F, b, c, d, a, x[ 3], 0xc1bdceee, 22)
    STEP(F, a, b, c, d, x[ 4], 0xf57c0faf,  7) STEP(F, d, a, b, c, x[ 5], 0x4787c62a, 12)
    STEP(F, c, d, a, b, x[ 6], 0xa8304613, 17) STEP(F, b, c, d, a, x[ 7], 0xfd469501, 22)
    STEP(F, a, b, c, d, x[ 8], 0x698098d8,  7) STEP(F, d, a, b, c, x[ 9], 0x8b44f7af, 12)
    STEP(F, c, d, a, b, x[10], 0xffff5bb1, 17) STEP(F, b, c, d, a, x[11], 0x895cd7be, 22)
    STEP(F, a, b, c, d, x[12], 0x6b901122,  7) STEP(F, d, a, b, c, x[13], 0xfd987193, 12)
    STEP(F, c, d, a, b, x[14], 0xa679438e, 17) STEP(F, b, c, d, a, x[15], 0x49b40821, 22)
    STEP(G, a, b, c, d, x[ 1], 0xf61e2562,  5) STEP(G, d, a, b, c, x[ 6], 0xc040b340,  9)
    STEP(G, c, d, a, b, x[11], 0x265e5a51, 14) STEP(G, b, c, d, a, x[ 0], 0xe9b6c7aa, 20)
    STEP(G, a, b, c, d, x[ 5], 0xd62f105d,  5) STEP(G, d, a, b, c, x[10], 0x02441453,  9)
    STEP(G, c, d, a, b, x[15], 0xd8a1e681, 14) STEP(G, b, c, d, a, x[ 4], 0xe7d3fbc8, 20)
    STEP(G, a, b, c, d, x[ 9], 0x21e1cde6,  5) STEP(G, d, a, b, c, x[14], 0xc33707d6,  9)
    STEP(G, c, d, a, b, x[ 3], 0xf4d50d87, 14) STEP(G, b, c, d, a, x[ 8], 0x455a14ed, 20)
    STEP(G, a, b, c, d, x[13], 0xa9e3e905,  5) STEP(G, d, a, b, c, x[ 2], 0xfcefa3f8,  9)
    STEP(G, c, d, a, b, x[ 7], 0x676f02d9, 14) STEP(G, b, c, d, a, x[12], 0x8d2a4c8a, 20)
    STEP(H, a, b, c, d, x[ 5], 0xfffa3942,  4) STEP(H, d, a, b, c, x[ 8], 0x8771f681, 11)
    STEP(H, c, d, a, b, x[11], 0x6d9d6122, 16) STEP(H, b, c, d, a, x[14], 0xfde5380c, 23)
    STEP(H, a, b, c, d, x[ 1], 0xa4beea44,  4) STEP(H, d, a, b, c, x[ 4], 0x4bdecfa9, 11)
    STEP(H, c, d, a, b, x[ 7], 0xf6bb4b60, 16) STEP(H, b, c, d, a, x[10], 0xbebfbc70, 23)
    STEP(H, a, b, c, d, x[13], 0x289b7ec6,  4) STEP(H, d, a, b, c, x[ 0], 0xeaa127fa, 11)
    STEP(H, c, d, a, b, x[ 3], 0xd4ef3085, 16) STEP(H, b, c, d, a, x[ 6], 0x04881d05, 23)
    STEP(H, a, b, c, d, x[ 9], 0xd9d4d039,  4) STEP(H, d, a, b, c, x[12], 0xe6db99e5, 11)
    STEP(H, c, d, a, b, x[15], 0x1fa27cf8, 16) STEP(H, b, c, d, a, x[ 2], 0xc4ac5665, 23)
    STEP(I, a, b, c, d, x[ 0], 0xf4292244,  6) STEP(I, d, a, b, c, x[ 7], 0x432aff97, 10)
    STEP(I, c, d, a, b, x[14], 0xab9423a7, 15) STEP(I, b, c, d, a, x[ 5], 0xfc93a039, 21)
    STEP(I, a, b, c, d, x[12], 0x655b59c3,  6) STEP(I, d, a, b, c, x[ 3], 0x8f0ccc92, 10)
    STEP(I, c, d, a, b, x[10], 0xffeff47d, 15) STEP(I, b, c, d, a, x[ 1], 0x85845dd1, 21)
    STEP(I, a, b, c, d, x[ 8], 0x6fa87e4f,  6) STEP(I, d, a, b, c, x[15], 0xfe2ce6e0, 10)
    STEP(I, c, d, a, b, x[ 6], 0xa3014314, 15) STEP(I, b, c, d, a, x[13], 0x4e0811a1, 21)
    STEP(I, a, b, c, d, x[ 4], 0xf7537e82,  6) STEP(I, d, a, b, c, x[11], 0xbd3af235, 10)
    STEP(I, c, d, a, b, x[ 2], 0x2ad7d2bb, 15) STEP(I, b, c, d, a, x[ 9], 0xeb86d391, 21)

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
}

__kernel void md5_bruteforce(__global unsigned char* target_digest,
                             __global char* charset, int charset_len, int max_len,
                             __global int* found_flag, __global char* found_str) {
    int idx = get_global_id(0);
    char str[16];
    int len = 0, temp = idx, offset = 0, powers = 1;

    for (int l = 1; l <= max_len; l++) {
        powers *= charset_len;
        if (idx < offset + powers) {
            len = l;
            temp = idx - offset;
            break;
        }
        offset += powers;
    }
    if (len == 0) return;

    for (int i = len - 1; i >= 0; i--) {
        str[i] = charset[temp % charset_len];
        temp /= charset_len;
    }

    unsigned char block[64];
    for (int i = 0; i < 64; i++) block[i] = 0;
    for (int i = 0; i < len; i++) block[i] = str[i];
    block[len] = 0x80;
    unsigned int bits = len * 8;
    block[56] = bits & 0xFF;
    block[57] = (bits >> 8) & 0xFF;
    block[58] = (bits >> 16) & 0xFF;
    block[59] = (bits >> 24) & 0xFF;

    unsigned int state[4] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476};
    md5_transform(state, block);

    unsigned char digest[16];
    for (int i = 0; i < 4; i++) {
        digest[i*4+0] = (state[i] >> 0) & 0xFF;
        digest[i*4+1] = (state[i] >> 8) & 0xFF;
        digest[i*4+2] = (state[i] >> 16) & 0xFF;
        digest[i*4+3] = (state[i] >> 24) & 0xFF;
    }

    int match = 1;
    for (int i = 0; i < 16; i++)
        if (digest[i] != target_digest[i]) { match = 0; break; }

    if (match && *found_flag == 0) {
        *found_flag = 1;
        for (int i = 0; i < len; i++) found_str[i] = str[i];
        found_str[len] = '\0';
    }
}
