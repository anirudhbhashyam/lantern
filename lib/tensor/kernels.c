#include <math.h>
#include <stdint.h>
#include <stddef.h>
#ifdef __APPLE__
#include <dlfcn.h>
#include <pthread.h>

typedef void (*Sgemm)(int, int, int, int, int, int, float, const float *,
                      int, const float *, int, float, float *, int);
static Sgemm sgemm;
static pthread_once_t blas_once = PTHREAD_ONCE_INIT;
static void load_blas(void) {
    void *lib = dlopen("/System/Library/Frameworks/Accelerate.framework/Accelerate", RTLD_LAZY | RTLD_LOCAL);
    if (lib) sgemm = (Sgemm)dlsym(lib, "cblas_sgemm");
}
#endif

typedef struct { const float *data; int64_t len; } Input;
typedef struct { float *data; int64_t len; } Output;

// Goose validates dimensions and spans before crossing the ABI boundary.
void goose_tensor_gemm(Input a, Input b, Output c, int32_t m, int32_t n,
                       int32_t k, int32_t lda, int32_t ldb, int32_t ta, int32_t tb) {
#ifdef __APPLE__
    pthread_once(&blas_once, load_blas);
    if (sgemm) {
        // CBLAS row major (101), no transpose (111), transpose (112).
        sgemm(101, ta ? 112 : 111, tb ? 112 : 111, m, n, k, 1.f,
              a.data, lda, b.data, ldb, 0.f, c.data, n);
        return;
    }
#endif
    for (int32_t i = 0; i < m; ++i) {
        for (int32_t j = 0; j < n; ++j) c.data[(size_t)i*n+j] = 0.f;
        for (int32_t p = 0; p < k; ++p) {
            float x = a.data[ta ? (size_t)p*lda+i : (size_t)i*lda+p];
            for (int32_t j = 0; j < n; ++j)
                c.data[(size_t)i*n+j] += x * b.data[tb ? (size_t)j*ldb+p : (size_t)p*ldb+j];
        }
    }
}


void goose_tensor_adam(Output p, Input g, Output m, Output v, Output ema,
                       float rate, float eps, float decay) {
    for (int64_t i = 0; i < p.len; ++i) {
        float grad = g.data[i];
        m.data[i] = 0.9f * m.data[i] + 0.1f * grad;
        v.data[i] = 0.999f * v.data[i] + 0.001f * grad * grad;
        p.data[i] -= rate * m.data[i] / (sqrtf(v.data[i]) + eps);
        ema.data[i] += (1.f - decay) * (p.data[i] - ema.data[i]);
    }
}
