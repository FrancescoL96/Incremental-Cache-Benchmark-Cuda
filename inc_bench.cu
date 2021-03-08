/*
 * Apologies to whoever will have to read this code, I just discovered precompiler macros and I went crazy with it..
 */
#include <chrono>
#include <iostream>
#include <random>
#include <cmath>
#include <atomic>
#include <cstdlib>

#include <stdio.h>

#include "Timer.cuh"
#include "CheckError.cuh"

#include <omp.h>

using namespace timer;

// Set PRINT to 1 for debug output
#define PRINT 0
#define FROM_debug 0
#define TO_debug 16

// Set ZEROCOPY to 1 to use Zero Copy Memory Mode, UNIFIED to 1 to use Unified Memory, COPY to 1 to use Copy
#define ZEROCOPY 0
#define UNIFIED 0
#define COPY 1

// Set RESULTCHECK to 1 to verify the result with a single CPU thread DO NOT ENABLE, results are non-deterministic
#define RESULTCHECK 0

// Set CPU to 1 to use the CPU concurrently
#define CPU 1
// Set OPENMP to 1 to use more than 1 thread for the CPU
#define OPENMP 1

unsigned int N = 2;
const int POW = 14;			 // Maximum is 30, anything higher and the system will use swap, making the Cuda kernels crash
const int RUNS = 10;
const int SUMS = 2;
const int BLOCK_SIZE_X = 32;


__global__
void sum_gpu_left(float* matrix, const int N, const int SPLIT) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < N) {
		for (int k = 0; k < N/SPLIT; k++) {
			matrix[row] = 2.0 * matrix[row+k];
			float temp = 2.0;
			for (int f = 0; f < 2; f++) {
				temp /= float(f) + sqrt(3.14159265359 * temp)/0.7;
				temp *= 1.6;
			}
			matrix[row] += temp;
		}
	}
}

__global__
void sum_gpu_right(float* matrix, const int N, const int SPLIT) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < N) {
		for (int k = 0; k < N/SPLIT; k++) {
			matrix[row] = 2.0 * matrix[row+k];
			float temp = 2.0;
			for (int f = 0; f < 2; f++) {
				temp /= float(f) + sqrt(3.14159265359 * temp)/0.7;
				temp *= 1.6;
			}
			matrix[row] += temp;
		}
	}
}
/*
__global__
void sum_gpu_right(float* matrix, const int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= N/2 && row < N) {
		if (row % 2 == 0) {		
			for (int k = N/2; k < (((SPLIT/2)+1)*N)/SPLIT; k++) {
				matrix[row] = 2.0 * matrix[row-k];
				float temp = 2.0;
				for (int f = 0; f < 2; f++) {
					temp /= float(f) + sqrt(3.14159265359 * temp)/0.7;
					temp *= 1.6;
				}
				matrix[row] += temp;
			}
		}
	}
}
*/
void sum_cpu_only(float * matrix){
	#if CPU
	for (int i = 0; i < SUMS; i++) {
		if (i % 2 != 0) {
	        for (int j = 0; j < N/2; j++) {
			    if (j % 2 != 0) {
			    	float temp = 2.0 * sqrt(matrix[j] + matrix[j+N/2]);
			    	for (int f = 0; f < 2; f++) {
		    			temp /= float(f) + sqrt(3.14159265359 * temp)/0.7;
				    	temp *= 1.6;
				    }
					matrix[j] = temp;
				}
	        }
			for (int j = N/2; j < N; j++) {
				if (j % 2 == 0) {
					for (int r = 0; r < 2; r++) {
						matrix[j] = sqrt(matrix[j]*(matrix[j] / 2.3));
					}
				}
			}
		} else {
   	        for (int j = N/2; j < N; j++) {
   	        	if (j % 2 == 0) {
			    	float temp = 2.0 * sqrt(matrix[j] + matrix[j-N/2]);
					for (int f = 0; f < 2; f++) {
						temp /= float(f) + sqrt(3.14159265359 * temp)/0.7;
				    	temp *= 1.6;
					}
					matrix[j] = temp;
				}
	        }
			for (int j = 0; j < N/2; j++) {
				if (j % 2 != 0) {
					for (int r = 0; r < 2; r++) {
						matrix[j] = sqrt(matrix[j]*(matrix[j] / 2.3));
					}
				}
			}
		}
		#if PRINT
		printf("RUN %d\n", i);
		printf("Values from index %d to %d\n", FROM_debug, TO_debug);
		printf("H: ");
		for (int i = FROM_debug; i < TO_debug; i++) {
			if (i % (N/2) == 0) printf("| ");
			printf("%.2f ", matrix[i]);
		}
 		printf("\n");
 		#endif
	}
	#else
	for (int i = 0; i < SUMS; i++) {
		for (int j = 0; j < N/2; j++) {
	        if (j % 2 != 0) {
				float temp = 2.0 * sqrt(matrix[j] + matrix[j+N/2]);
				for (int f = 0; f < 2; f++) {
					temp /= float(f) + sqrt(3.14159265359 * temp)/0.7;
					temp *= 1.6;
				}
				matrix[j] = temp;
			}			
        }
		for (int j = N/2; j < N; j++) {
        	if (j % 2 == 0) {
				float temp = 2.0 * sqrt(matrix[j] + matrix[j+N/2]);
				for (int f = 0; f < 2; f++) {
					temp /= float(f) + sqrt(3.14159265359 * temp)/0.7;
					temp *= 1.6;
				}
				matrix[j] = temp;
			}
        }
	}
	#endif
}

int main(int argc, char **argv) {
	int SPLIT = atoi(argv[1]);
    N = (unsigned int) pow(N, POW);
    int grid = N / BLOCK_SIZE_X;
    // -------------------------------------------------------------------------
    // DEVICE INIT
    dim3 DimGrid(grid, 1, 1);
    if (N % grid) DimGrid.x++;
    dim3 DimBlock(BLOCK_SIZE_X, 1, 1);

    // -------------------------------------------------------------------------
    // HOST MEMORY ALLOCATION
    float * h_matrix = new float[N];
 
    std::vector<float> results; 	// Stores computation times for CPU+GPU
    std::vector<float> cpu_results; // Stores CPU (only) computation times
    std::vector<float> gpu_results; // Stores GPU (only) computation times

    // -------------------------------------------------------------------------
    #if ZEROCOPY
    cudaSetDeviceFlags(cudaDeviceMapHost);
    #endif
    for (int z = 0; z < RUNS; z++) {
        std::cout << "Run " << z << " --------------------------- ";
       	if (ZEROCOPY) std::cout << "ZC";
		else if(UNIFIED) std::cout << "UM";
		else if(COPY) std::cout << "CP ";
		std::cout << "- Array section length: 1/" << SPLIT << std::endl;
		
        Timer<HOST> TM;
        Timer<HOST> TM_host;

        // -------------------------------------------------------------------------
        // DEVICE MEMORY ALLOCATION
        float * d_matrix_host;
        float * d_matrix;
        #if ZEROCOPY
        // Zero Copy Allocation
		SAFE_CALL(cudaHostAlloc((void **)&d_matrix_host, N * sizeof(float), cudaHostAllocMapped));
        SAFE_CALL(cudaHostGetDevicePointer((void **)&d_matrix, (void *) d_matrix_host , 0));
        #elif UNIFIED
        // Unified Memory Allocation
        SAFE_CALL(cudaMallocManaged(&d_matrix, N * sizeof(float)));
        #elif COPY
        // Standard Copy
        float * d_matrix_device;
        SAFE_CALL(cudaMalloc(&d_matrix_device, N * sizeof(float)));
        d_matrix = new float[N];
		#endif
        // -------------------------------------------------------------------------
        // MATRIX INITILIZATION
        std::cout << "Starting Initialization..." << std::endl;

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator(seed);
        std::uniform_int_distribution<int> distribution(1, 100);

		#if PRINT
		int count = 1;
		printf("Progress: 0 \%\t");
  		fflush(stdout);
		#endif
        for (int i = 0; i < N; i++) {
			#if PRINT
			float cur_prog = (float) i / (float) N;
			if ( cur_prog >= 0.1 * (float) count) {
				printf("\rProgress: %.0f \%\t", cur_prog * (float) 100);
				fflush(stdout);
				count++;
			}
			#endif
			int temp = distribution(generator);
			h_matrix[i] = temp;
			d_matrix[i] = temp;
        }
        #if PRINT
        printf("\r							\r");
        #endif
        
        // -------------------------------------------------------------------------
        // INITILIZATION PRINT (DEBUG)
		#if PRINT
		printf("Values from index %d to %d\n", FROM_debug, TO_debug);
		printf("H: ");
	    for (int i = FROM_debug; i < TO_debug; i++) {
	    	if (i % (N/2) == 0) printf("| ");
	    	printf("%.2f ", h_matrix[i]);
	    }
		printf("\n");
		printf("D: ");
	    for (int i = FROM_debug; i < TO_debug; i++) {
	    	if (i % (N/2) == 0) printf("| ");
			printf("%.2f ", d_matrix[i]);
	    }
   		printf("\n");
		#endif
        std::cout << "Initialization Finished" << std::endl;

        // -------------------------------------------------------------------------
        // CPU ONLY EXECUTION
        #if RESULTCHECK
        std::cout << "Starting computation (1T - NO GPU)..." << std::endl;
        sum_cpu_only(h_matrix);
        #endif
        // -------------------------------------------------------------------------
        // DEVICE EXECUTION
        std::cout << "Starting computation (GPU+CPU)..." << std::endl;
		TM.start();
		
	    #if CPU
		for (int i = 0; i < SUMS; i++) {
			if (i % 2 != 0) {
				#if COPY
				SAFE_CALL(cudaMemcpy(d_matrix_device, d_matrix, N * sizeof(int), cudaMemcpyHostToDevice));
		        sum_gpu_left << < DimGrid, DimBlock >> > (d_matrix_device, N, SPLIT);
		        CHECK_CUDA_ERROR
   		        SAFE_CALL(cudaMemcpy(d_matrix, d_matrix_device, N * sizeof(int), cudaMemcpyDeviceToHost));
				#else
		        sum_gpu_left << < DimGrid, DimBlock >> > (d_matrix, N, SPLIT);
		        #endif
		        #if UNIFIED
		        // This macro includes cudaDeviceSynchronize(), which makes the program work on the data in lockstep
		        CHECK_CUDA_ERROR
		        #endif
		        TM_host.start();
		        #if OPENMP
				#pragma omp parallel for
				#endif
				for (int j = N/2; j < N; j++) {
					if (j % 2 == 0) {
						//__sync_fetch_and_add(&d_matrix[j], 1);
						for (int r = 0; r < 2; r++) {
							d_matrix[j] = sqrt(d_matrix[j]*(d_matrix[j] / 2.3));
						}
						//printf("cpu right: %d\n", j);
					}
				}
		        TM_host.stop();
			} else {				
				#if COPY
				SAFE_CALL(cudaMemcpy(d_matrix_device, d_matrix, N * sizeof(int), cudaMemcpyHostToDevice));
	   	        sum_gpu_right << < DimGrid, DimBlock >> > (d_matrix_device, N, SPLIT);
   		        CHECK_CUDA_ERROR
		        SAFE_CALL(cudaMemcpy(d_matrix, d_matrix_device, N * sizeof(int), cudaMemcpyDeviceToHost));
				#else
	   	        sum_gpu_right << < DimGrid, DimBlock >> > (d_matrix, N, SPLIT);
	   	        #endif
   		        #if UNIFIED
   		        CHECK_CUDA_ERROR
   		        #endif
   		        TM_host.start();
	   	        #if OPENMP
				#pragma omp parallel for
				#endif
				for (int j = 0; j < N/2; j++) {
					if (j % 2 != 0) {
						//__sync_fetch_and_add(&d_matrix[j], 1);
						for (int r = 0; r < 2; r++) {
							d_matrix[j] = sqrt(d_matrix[j]*(d_matrix[j] / 2.3));
						}
						//printf("cpu left: %d\n", j);
					}
				}
				TM_host.stop();
			}
			// Synchronization needed to avoid race conditions (after the CPU and GPU have done their sides, we need to sync)
			#if ZEROCOPY
			CHECK_CUDA_ERROR
			#endif
			// -------------------------------------------------------------------------
    	    // PARTIAL RESULT PRINT (DEBUG)
			#if PRINT
			printf("RUN %d\n", i);
			printf("Values from index %d to %d\n", FROM_debug, TO_debug);
			printf("D: ");
			for (int i = FROM_debug; i < TO_debug; i++) {
				if (i % (N/2) == 0) printf("| ");
				printf("%.2f ", d_matrix[i]);
			}
	 		printf("\n");
	 		#endif
			// -------------------------------------------------------------------------
		}
        #else
        #if COPY
		SAFE_CALL(cudaMemcpy(d_matrix_device, d_matrix, N * sizeof(int), cudaMemcpyHostToDevice));
		#endif
        for (int i = 0; i < SUMS; i++) {
			#if COPY
	        sum_gpu_left << < DimGrid, DimBlock >> > (d_matrix_device, N);
   	        sum_gpu_right << < DimGrid, DimBlock >> > (d_matrix_device, N);
	        #else
	        sum_gpu_left << < DimGrid, DimBlock >> > (d_matrix, N);
   	        sum_gpu_right << < DimGrid, DimBlock >> > (d_matrix, N);
   	        #endif
	    }
        #endif
        #if COPY && !CPU
        SAFE_CALL(cudaMemcpy(d_matrix, d_matrix_device, N * sizeof(int), cudaMemcpyDeviceToHost));
        #endif
        CHECK_CUDA_ERROR
        TM.stop();
	
		// -------------------------------------------------------------------------
        // RESULT PRINT (DEBUG)
		#if PRINT
		printf("Values from index %d to %d\n", FROM_debug, TO_debug);
		printf("H: ");
	    for (int i = FROM_debug; i < TO_debug; i++) {
	    	if (i % (N/2) == 0) printf("| ");
	    	printf("%.2f ", h_matrix[i]);
	    }
		printf("\n");
		printf("D: ");
	    for (int i = FROM_debug; i < TO_debug; i++) {
	    	if (i % (N/2) == 0) printf("| ");
			printf("%.2f ", d_matrix[i]);
	    }
 		printf("\n");
 		#endif
 		
        cpu_results.push_back(TM_host.total_duration());
        results.push_back(TM.total_duration());

        // -------------------------------------------------------------------------
        // RESULT CHECK
        #if RESULTCHECK
        for (int i = 0; i < N; i++) {
            if (h_matrix[i] != d_matrix[i]) {
                std::cerr << ">< wrong result at: "
                            << (i)
                            << "\n\thost:   " << h_matrix[i]
                            << "\n\tdevice: " << d_matrix[i] << "\n";       
                            
                #if PRINT
  				int err_min = i-5;
				int err_max = i+5;
				if (err_min < 0) err_min = 0;
				if (err_max > N) err_max = N;
				printf("Values from index %d to %d\n", err_min, err_max);
				printf("\tH: ");
				for (int j = err_min; j < err_max; j++) {
					printf("%.2f ", h_matrix[j]);
				}
				printf("\n");
				printf("\tD: ");
				for (int j = err_min; j < err_max; j++) {
					printf("%.2f ", d_matrix[j]);
				}
		 		printf("\n\n");
		 		#endif
                
                cudaDeviceReset();
                std::exit(EXIT_FAILURE);
            }
        }
        std::cout << "<> Correct\n\n";
        #endif

        // -------------------------------------------------------------------------
        // DEVICE MEMORY DEALLOCATION
        #if ZEROCOPY
        SAFE_CALL(cudaFreeHost(d_matrix));
        #elif UNIFIED
        SAFE_CALL(cudaFree(d_matrix));
        #elif COPY
        SAFE_CALL(cudaFree(d_matrix_device));
	    // HOST MEMORY DEALLOCATION
        delete(d_matrix);
        #endif
    }
    // -------------------------------------------------------------------------
    cudaDeviceReset();
    delete(h_matrix);

    // -------------------------------------------------------------------------
    std::cout << "Average ";
	if (ZEROCOPY) std::cout << "ZC";
	else if(UNIFIED) std::cout << "UM";
	else if(COPY) std::cout << "CP";
	std::cout << " Run time: " << std::accumulate(results.begin(), results.end(), 0) / float(RUNS) << " ms - ";
    std::cout << "CPU time only " << std::accumulate(cpu_results.begin(), cpu_results.end(), 0) / float(RUNS) << " ms" << std::endl;

}
