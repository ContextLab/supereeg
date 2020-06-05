blur = '''

__device__ void warpReduce(volatile float* s1, volatile float* s2, volatile float* s3, int tid) {
    if (BLOCKSIZE >= 64) {
        s1[tid] += s1[tid + 32];
        s2[tid] += s2[tid + 32];
        s3[tid] += s3[tid + 32];
    }
    if (BLOCKSIZE >= 32) {
        s1[tid] += s1[tid + 16];
        s2[tid] += s2[tid + 16];
        s3[tid] += s3[tid + 16];
    }
    if (BLOCKSIZE >= 16) {
        s1[tid] += s1[tid + 8];
        s2[tid] += s2[tid + 8];
        s3[tid] += s3[tid + 8];
    }
    if (BLOCKSIZE >= 8) { 
        s1[tid] += s1[tid + 4];
        s2[tid] += s2[tid + 4];
        s3[tid] += s3[tid + 4];
    }
    if (BLOCKSIZE >= 4) {
        s1[tid] += s1[tid + 2];
        s2[tid] += s2[tid + 2];
        s3[tid] += s3[tid + 2];
    }
    if (BLOCKSIZE >= 2) {
        s1[tid] += s1[tid + 1];
        s2[tid] += s2[tid + 1];
        s3[tid] += s3[tid + 1];
    }
}

extern "C"
__global__ void bigReduce(float* lzp, float* lzn, float* nwt, int n, 
                                            float* kpval, float* knval, float* wval, float maxval) {
    __shared__ float w_shared[BLOCKSIZE];
    __shared__ float kp_shared[BLOCKSIZE];
    __shared__ float kn_shared[BLOCKSIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * BLOCKSIZE * 2 + tid;
    int gridSize = BLOCKSIZE * 2 * gridDim.x;
    w_shared[tid] = 0;
    kp_shared[tid] = 0;
    kn_shared[tid] = 0;
    while (i < n) {
        float w = nwt[i] - maxval;
        w_shared[tid] += exp(w);
        kp_shared[tid] += exp(lzp[i] + w);
        kn_shared[tid] += exp(lzn[i] + w);
        if (i + BLOCKSIZE < n) {
            w = nwt[i + BLOCKSIZE] - maxval;
            w_shared[tid] += exp(w);
            kp_shared[tid] += exp(lzp[i + BLOCKSIZE] + w);
            kn_shared[tid] += exp(lzn[i + BLOCKSIZE] + w);
        }
        i += gridSize;
    }
    __syncthreads();

    if (BLOCKSIZE >= 512) { if (tid < 256 ) {
        w_shared[tid] += w_shared[tid + 256]; 
        kp_shared[tid] += kp_shared[tid + 256]; 
        kn_shared[tid] += kn_shared[tid + 256]; 
    } __syncthreads(); }
    if (BLOCKSIZE >= 256) { if (tid < 128 ) {
        w_shared[tid] += w_shared[tid + 128]; 
        kp_shared[tid] += kp_shared[tid + 128]; 
        kn_shared[tid] += kn_shared[tid + 128]; 
    } __syncthreads(); }
    if (BLOCKSIZE >= 128) { if (tid < 64 ) {
        w_shared[tid] += w_shared[tid + 64]; 
        kp_shared[tid] += kp_shared[tid + 64]; 
        kn_shared[tid] += kn_shared[tid + 64]; 
    } __syncthreads(); }

    if (tid < 32) warpReduce(w_shared, kp_shared, kn_shared, tid);
    if (tid == 0) {
        atomicAdd(wval, w_shared[0]);
        atomicAdd(kpval, kp_shared[0]);
        atomicAdd(knval, kn_shared[0]);
    }
}

extern "C"
__global__ void outerTriagSum(float* xwt, float* ywt, float* out, int* ktx, int* kty, int n) {
    int i = blockIdx.x * BLOCKSIZE + threadIdx.x;
    if (i < n) {
        out[i] = xwt[ktx[i]] + ywt[kty[i]];
    }
}

extern "C"
__global__ void logAndAdd(float* kp, float* kn, float* w, float* maxvals, int n) {
    int i = blockIdx.x * BLOCKSIZE + threadIdx.x;
    if (i < n) {
        float maxval = maxvals[i];
        kp[i] = log(kp[i]) + maxval;
        kn[i] = log(kn[i]) + maxval;
        if (w[i] != 0.) w[i] = log(w[i]) + maxval;
    }
}
'''