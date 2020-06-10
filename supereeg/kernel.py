blur = r'''

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
__global__ void blur(float* lzp, float* lzn, float* weights, float* Zp, int* wtx, int* wty, int* ktx, int* kty,
    float* kp, float* kn, float* w, float* maxes, int w_n_rows, int w_n_cols, int n_kt) {

    __shared__ float w_shared[BLOCKSIZE];
    __shared__ float kp_shared[BLOCKSIZE];
    __shared__ float kn_shared[BLOCKSIZE];

    int tid = threadIdx.x;
    int blid = blockIdx.x;

    int x = wtx[blid];
    int y = wty[blid];
    float* xwt = &weights[x*w_n_cols];
    float* ywt = &weights[y*w_n_cols];

    float neg_inf = __int_as_float(0xff800000);

    w_shared[tid] = 0;
    kp_shared[tid] = 0;
    kn_shared[tid] = 0;

    float maxval = maxes[blid];
    if (maxval <= 1) { // we can check if the max is invalid, which indicates a match; this reduces 
                                // global memory accesses
        int i = threadIdx.x;
        while (i < n_kt) {
            float w = xwt[ktx[i]] + ywt[kty[i]] - maxval;
            w_shared[tid] += exp(w);
            kp_shared[tid] += exp(lzp[i] + w);
            kn_shared[tid] += exp(lzn[i] + w);
            i += BLOCKSIZE;
        }
        __syncthreads();

        if (BLOCKSIZE == 1024) { if (tid < 512) {
            w_shared[tid] += w_shared[tid + 512];
            kp_shared[tid] += kp_shared[tid + 512];
            kn_shared[tid] += kn_shared[tid + 512];
        } __syncthreads(); }
        if (BLOCKSIZE >= 512) { if (tid < 256) {
            w_shared[tid] += w_shared[tid + 256];
            kp_shared[tid] += kp_shared[tid + 256];
            kn_shared[tid] += kn_shared[tid + 256];
        } __syncthreads(); }
        if (BLOCKSIZE >= 256) { if (tid < 128) {
            w_shared[tid] += w_shared[tid + 128];
            kp_shared[tid] += kp_shared[tid + 128];
            kn_shared[tid] += kn_shared[tid + 128];
        } __syncthreads(); }
        if (BLOCKSIZE >= 128) { if (tid < 64) {
            w_shared[tid] += w_shared[tid + 64];
            kp_shared[tid] += kp_shared[tid + 64];
            kn_shared[tid] += kn_shared[tid + 64];
        } __syncthreads(); }

        if (tid < 32) warpReduce(w_shared, kp_shared, kn_shared, tid);
        if (tid == 0) {
            kp[blid] = log(kp_shared[0]) + maxval;
            kn[blid] = log(kn_shared[0]) + maxval;
            if (w_shared[0] != 0.) w[blid] = log(w_shared[0]) + maxval;
        }
    } else {
        float zval = Zp[x*w_n_rows + y];
        if (zval > 0) {
            kp[blid] = log(zval);
            kn[blid] = neg_inf;
        } else {
            kp[blid] = neg_inf;
            kn[blid] = log(abs(zval));
        }
    }

}

__device__ void warpReduce(volatile float *sdata, unsigned int tid) {
    if (BLOCKSIZE >= 64) sdata[tid] = max(sdata[tid], sdata[tid + 32]);
    if (BLOCKSIZE >= 32) sdata[tid] = max(sdata[tid], sdata[tid + 16]);
    if (BLOCKSIZE >= 16) sdata[tid] = max(sdata[tid], sdata[tid + 8]);
    if (BLOCKSIZE >= 8) sdata[tid] = max(sdata[tid], sdata[tid + 4]);
    if (BLOCKSIZE >= 4) sdata[tid] = max(sdata[tid], sdata[tid + 2]);
    if (BLOCKSIZE >= 2) sdata[tid] = max(sdata[tid], sdata[tid + 1]);
}

extern "C"
__global__ void arrmax(float* weights, float* maxes, bool* matches, int* wtx, int* wty, int* ktx, int* kty, int w_n_rows, int w_n_cols, int n_kt) {
    __shared__ float sdata[BLOCKSIZE];

    int tid = threadIdx.x;
    int blid = blockIdx.x;

    int x = wtx[blid];
    int y = wty[blid];
    float* xwt = &weights[x*w_n_cols];
    float* ywt = &weights[y*w_n_cols];
    sdata[tid] = -0x7ff00000;
    int i = threadIdx.x;
    while (i < n_kt) {
        sdata[tid] = max(xwt[ktx[i]] + ywt[kty[i]], sdata[tid]);
        i += BLOCKSIZE;
    }
    __syncthreads();
    if (BLOCKSIZE == 1024) { if (tid < 512) { sdata[tid] = max(sdata[tid], sdata[tid + 512]); } __syncthreads(); }
    if (BLOCKSIZE >= 512) { if (tid < 256) { sdata[tid] = max(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
    if (BLOCKSIZE >= 256) { if (tid < 128) { sdata[tid] = max(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
    if (BLOCKSIZE >= 128) { if (tid < 64) { sdata[tid] = max(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
    if (tid < 32) warpReduce(sdata, tid);
    if (tid == 0) {
        if (matches[x*w_n_rows + y]) {
            maxes[blid] = 3.14; // max will never be positive, this indicates
                                            // that a match has been found to reduce memory access
                                            // in second kernel
        } else {
            maxes[blid] = sdata[0];
        }
    }
}

extern "C"
__global__ void integrated(float* lzp, float* lzn, float* weights, float* Zp, int* wtx, int* wty, int* ktx, int* kty,
                                            float* kp, float* kn, float* w, bool* matches, int w_n_rows, int w_n_cols, int n_kt) {

    __shared__ float sdata[BLOCKSIZE];
    __shared__ float w_shared[BLOCKSIZE];
    __shared__ float kp_shared[BLOCKSIZE];
    __shared__ float kn_shared[BLOCKSIZE];

    int tid = threadIdx.x;
    int blid = blockIdx.x;

    int x = wtx[blid];
    int y = wty[blid];
    float* xwt = &weights[x*w_n_cols];
    float* ywt = &weights[y*w_n_cols];
    sdata[tid] = -0x7ff00000;
    int i = threadIdx.x;
    while (i < n_kt) {
        sdata[tid] = max(xwt[ktx[i]] + ywt[kty[i]], sdata[tid]);
        i += BLOCKSIZE;
    }
    __syncthreads();
    if (BLOCKSIZE == 1024) { if (tid < 512) { sdata[tid] = max(sdata[tid], sdata[tid + 512]); } __syncthreads(); }
    if (BLOCKSIZE >= 512) { if (tid < 256) { sdata[tid] = max(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
    if (BLOCKSIZE >= 256) { if (tid < 128) { sdata[tid] = max(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
    if (BLOCKSIZE >= 128) { if (tid < 64) { sdata[tid] = max(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
    if (tid < 32) warpReduce(sdata, tid);
    __syncthreads();

    float maxval = sdata[0];
    __syncthreads();
    float neg_inf = __int_as_float(0xff800000);

    w_shared[tid] = 0;
    kp_shared[tid] = 0;
    kn_shared[tid] = 0;
    i = threadIdx.x;
    if (matches[x*w_n_rows + y]) { 
        while (i < n_kt) {
            float w = xwt[ktx[i]] + ywt[kty[i]] - maxval;
            w_shared[tid] += exp(w);
            kp_shared[tid] += exp(lzp[i] + w);
            kn_shared[tid] += exp(lzn[i] + w);
            i += BLOCKSIZE;
        }
        __syncthreads();

        if (BLOCKSIZE == 1024) { if (tid < 512) {
            w_shared[tid] += w_shared[tid + 512];
            kp_shared[tid] += kp_shared[tid + 512];
            kn_shared[tid] += kn_shared[tid + 512];
        } __syncthreads(); }
        if (BLOCKSIZE >= 512) { if (tid < 256) {
            w_shared[tid] += w_shared[tid + 256];
            kp_shared[tid] += kp_shared[tid + 256];
            kn_shared[tid] += kn_shared[tid + 256];
        } __syncthreads(); }
        if (BLOCKSIZE >= 256) { if (tid < 128) {
            w_shared[tid] += w_shared[tid + 128];
            kp_shared[tid] += kp_shared[tid + 128];
            kn_shared[tid] += kn_shared[tid + 128];
        } __syncthreads(); }
        if (BLOCKSIZE >= 128) { if (tid < 64) {
            w_shared[tid] += w_shared[tid + 64];
            kp_shared[tid] += kp_shared[tid + 64];
            kn_shared[tid] += kn_shared[tid + 64];
        } __syncthreads(); }

        if (tid < 32) warpReduce(w_shared, kp_shared, kn_shared, tid);
        if (tid == 0) {
            kp[blid] = log(kp_shared[0]) + maxval;
            kn[blid] = log(kn_shared[0]) + maxval;
            if (w_shared[0] != 0.) w[blid] = log(w_shared[0]) + maxval;
        }
    } else {
        float zval = Zp[x*w_n_rows + y];
        if (zval > 0) {
            kp[blid] = log(zval);
            kn[blid] = neg_inf;
        } else {
            kp[blid] = neg_inf;
            kn[blid] = log(abs(zval));
        }
    }
}
'''