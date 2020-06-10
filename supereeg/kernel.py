# blur = '''

# __device__ void warpReduce(volatile float* s1, volatile float* s2, volatile float* s3, int tid) {
#     if (BLOCKSIZE >= 64) {
#         s1[tid] += s1[tid + 32];
#         s2[tid] += s2[tid + 32];
#         s3[tid] += s3[tid + 32];
#     }
#     if (BLOCKSIZE >= 32) {
#         s1[tid] += s1[tid + 16];
#         s2[tid] += s2[tid + 16];
#         s3[tid] += s3[tid + 16];
#     }
#     if (BLOCKSIZE >= 16) {
#         s1[tid] += s1[tid + 8];
#         s2[tid] += s2[tid + 8];
#         s3[tid] += s3[tid + 8];
#     }
#     if (BLOCKSIZE >= 8) { 
#         s1[tid] += s1[tid + 4];
#         s2[tid] += s2[tid + 4];
#         s3[tid] += s3[tid + 4];
#     }
#     if (BLOCKSIZE >= 4) {
#         s1[tid] += s1[tid + 2];
#         s2[tid] += s2[tid + 2];
#         s3[tid] += s3[tid + 2];
#     }
#     if (BLOCKSIZE >= 2) {
#         s1[tid] += s1[tid + 1];
#         s2[tid] += s2[tid + 1];
#         s3[tid] += s3[tid + 1];
#     }
# }

# // extern "C"
# __global__ void outerTriagSum(float* xwt, float* ywt, float* out, int* ktx, int* kty, int n) {
#     int i = blockIdx.x * BLOCKSIZE + threadIdx.x;
#     if (i < n) {
#         out[i] = xwt[ktx[i]] + ywt[kty[i]];
#     }
# }

# __device__ void warpReduce(volatile int *sdata, unsigned int tid) {
#     if (BLOCKSIZE >= 64) sdata[tid] = max(sdata[tid], sdata[tid + 32]);
#     if (BLOCKSIZE >= 32) sdata[tid] = max(sdata[tid], sdata[tid + 16]);
#     if (BLOCKSIZE >= 16) sdata[tid] = max(sdata[tid], sdata[tid + 8]);
#     if (BLOCKSIZE >= 8) sdata[tid] = max(sdata[tid], sdata[tid + 4]);
#     if (BLOCKSIZE >= 4) sdata[tid] = max(sdata[tid], sdata[tid + 2]);
#     if (BLOCKSIZE >= 2) sdata[tid] = max(sdata[tid], sdata[tid + 1]);
# }


# __global__ void arrmax(float* g_idata, int* g_odata, unsigned int n) {
#     __shared__ int sdata[BLOCKSIZE];
#     unsigned int tid = threadIdx.x;
#     unsigned int i = blockIdx.x*(BLOCKSIZE*2) + tid;
#     unsigned int gridSize = BLOCKSIZE*2*gridDim.x;
#     sdata[tid] = 0;
#     while (i < n) { 
#         sdata[tid] = __float_as_int(g_idata[i]);
#         if (i + BLOCKSIZE < n) sdata[tid] = max(sdata[tid], __float_as_int(g_idata[i+BLOCKSIZE]));
#         i += gridSize; 
#     }
#     __syncthreads();
#     if (BLOCKSIZE >= 512) { if (tid < 256) { sdata[tid] = max(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
#     if (BLOCKSIZE >= 256) { if (tid < 128) { sdata[tid] = max(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
#     if (BLOCKSIZE >= 128) { if (tid < 64) { sdata[tid] = max(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
#     if (tid < 32) warpReduce(sdata, tid);
#     if (tid == 0) atomicMax(g_odata, sdata[0]);
# }


# // extern "C"
# __global__ void bigReduce(float* lzp, float* lzn, float* nwt, int n, 
#                                             float* kpval, float* knval, float* wval, float maxval) {
#     __shared__ float w_shared[BLOCKSIZE];
#     __shared__ float kp_shared[BLOCKSIZE];
#     __shared__ float kn_shared[BLOCKSIZE];

#     int tid = threadIdx.x;
#     int i = blockIdx.x * BLOCKSIZE * 2 + tid;
#     int gridSize = BLOCKSIZE * 2 * gridDim.x;
#     w_shared[tid] = 0;
#     kp_shared[tid] = 0;
#     kn_shared[tid] = 0;
#     while (i < n) {
#         float w = nwt[i] - maxval;
#         w_shared[tid] += exp(w);
#         kp_shared[tid] += exp(lzp[i] + w);
#         kn_shared[tid] += exp(lzn[i] + w);
#         if (i + BLOCKSIZE < n) {
#             w = nwt[i + BLOCKSIZE] - maxval;
#             w_shared[tid] += exp(w);
#             kp_shared[tid] += exp(lzp[i + BLOCKSIZE] + w);
#             kn_shared[tid] += exp(lzn[i + BLOCKSIZE] + w);
#         }
#         i += gridSize;
#     }
#     __syncthreads();

#     if (BLOCKSIZE >= 512) { if (tid < 256 ) {
#         w_shared[tid] += w_shared[tid + 256]; 
#         kp_shared[tid] += kp_shared[tid + 256]; 
#         kn_shared[tid] += kn_shared[tid + 256]; 
#     } __syncthreads(); }
#     if (BLOCKSIZE >= 256) { if (tid < 128 ) {
#         w_shared[tid] += w_shared[tid + 128]; 
#         kp_shared[tid] += kp_shared[tid + 128]; 
#         kn_shared[tid] += kn_shared[tid + 128]; 
#     } __syncthreads(); }
#     if (BLOCKSIZE >= 128) { if (tid < 64 ) {
#         w_shared[tid] += w_shared[tid + 64]; 
#         kp_shared[tid] += kp_shared[tid + 64]; 
#         kn_shared[tid] += kn_shared[tid + 64]; 
#     } __syncthreads(); }

#     if (tid < 32) warpReduce(w_shared, kp_shared, kn_shared, tid);
#     if (tid == 0) {
#         atomicAdd(wval, w_shared[0]);
#         atomicAdd(kpval, kp_shared[0]);
#         atomicAdd(knval, kn_shared[0]);
#     }
# }

# extern "C"
# __global__ void topKernel(float* lzp, float* lzn, float* weights, float* nwt_all, int n_kt, float* kp, float* kn, float* w,
#         int* wtx, int* wty, bool* matches, float* Zp, int* ktx, int* kty, int* maxes, int w_n_rows, int w_n_cols, int n_wt) {
    
#     int idx = BLOCKSIZE * blockIdx.x + threadIdx.x;

#     if (idx < n_wt) {

#         // float pos_inf = __int_as_float(0x7f800000);
#         float neg_inf = __int_as_float(0xff800000);

#         int x = wtx[idx];
#         int y = wty[idx];

#         if (matches[x*w_n_rows + y]) {

#             float zval = Zp[x*w_n_rows + y];

#             if (zval > 0) {
#                 kp[idx] = zval;
#                 kn[idx] = 0;
#             } else {
#                 kp[idx] = 0;
#                 kn[idx] = abs(zval);
#             }

#         } else {
#             float* xwt = &weights[x*w_n_cols];
#             float* ywt = &weights[y*w_n_cols];
#             float* nwt = &nwt_all[idx*n_kt];
#             int nblocks = __float_as_int(ceilf(n_kt / BLOCKSIZE));
#             outerTriagSum<<<nblocks,BLOCKSIZE>>>(xwt, ywt, nwt, ktx, kty, n_kt);
#             // thrust::device_ptr<float> nwt_t = thrust::device_pointer_cast(nwt);
#             // thrust::device_vector<float> iter = thrust::max_element(nwt_t, nwt_t + n_kt);
#             // float nwtmax = *iter;
#             int* mptr = &maxes[idx];
#             arrmax<<<nblocks,BLOCKSIZE>>>(nwt, mptr, n_kt);
#             float* kpval = &kp[idx]; float* knval = &kn[idx]; float* wval = &w[idx];
#             bigReduce<<<nblocks,BLOCKSIZE>>>(lzp, lzn, nwt, n_kt, kpval, knval, wval, *mptr);
#         }
#     }
# }


# extern "C"
# __global__ void logAndAdd(float* kp, float* kn, float* w, float* maxvals, int n) {
#     int i = blockIdx.x * BLOCKSIZE + threadIdx.x;
#     if (i < n) {
#         float maxval = maxvals[i];
#         kp[i] = log(kp[i]) + maxval;
#         kn[i] = log(kn[i]) + maxval;
#         if (w[i] != 0.) w[i] = log(w[i]) + maxval;
#     }
# }
# '''

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