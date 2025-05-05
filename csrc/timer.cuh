#include <cuda_runtime.h>
#include <mutex>

class CUDAKernelTimer {
 public:
  static CUDAKernelTimer& getInstance() {
    static CUDAKernelTimer timer;
    return timer;
  }

  ~CUDAKernelTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  CUDAKernelTimer(const CUDAKernelTimer&) = delete;
  void operator=(const CUDAKernelTimer&) = delete;

  void begin(const cudaStream_t& stream = nullptr) {
    checkCudaError(cudaEventRecord(start, stream),
                   "Failed to record start event");
  }

  void end(const cudaStream_t& stream = nullptr) {
    checkCudaError(cudaEventRecord(stop, stream),
                   "Failed to record stop event");
  }

  float elapse() {
    float time;
    checkCudaError(cudaEventSynchronize(stop),
                   "Failed to synchronize event stop");
    checkCudaError(cudaEventElapsedTime(&time, start, stop),
                   "Failed to get elapsed time");
    return time;
  }

 private:
  cudaEvent_t start;
  cudaEvent_t stop;

  CUDAKernelTimer() {
    checkCudaError(cudaEventCreate(&start), "Failed to create cuda event.");
    checkCudaError(cudaEventCreate(&stop), "Failed to create cuda event.");
  }

  static void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string(msg) + ": " +
                               cudaGetErrorString(err));
    }
  }
};