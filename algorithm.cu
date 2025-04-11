#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include "helper_cuda.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <cfloat>

#define COLOR_CHANNELS 0

__global__ void copy_image(const unsigned char *imageIn, unsigned char *imageOut, const int width, const int height, const int cpp) {
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (gidx == 0 & gidy == 0) {
        printf("DEVICE: START COPY\n");
    }
    
    for (int i = gidx; i < height; i += blockDim.x * gridDim.x) {
        for (int j = gidy; j < width; j += blockDim.y * gridDim.y) {
            for (int c = 0; c < cpp; c += 1) {
                imageOut[(i * width + j) * cpp + c] = imageIn[(i * width + j) * cpp + c];
            }
        }
    }

}

// RGB to YUV conversion device function
__device__ void rgb_to_yuv_cuda(int R, int G, int B, int *Y, int *U, int *V) {
    *Y = (int)roundf(0.299f * R + 0.587f * G + 0.114f * B);
    *U = (int)roundf(-0.168736f * R - 0.331264f * G + 0.5f * B) + 128;
    *V = (int)roundf(0.5f * R - 0.418688f * G - 0.081312f * B) + 128;
}

// CUDA kernel for RGB to YUV conversion
__global__ void rgbToYuvKernel(const unsigned char *image_in, unsigned char *image_out, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check if the thread corresponds to a valid pixel
    if (row < height && col < width) {
        int index = (row * width + col) * 3;
        
        // Get RGB values
        int R = image_in[index];
        int G = image_in[index + 1];
        int B = image_in[index + 2];
        
        // Convert to YUV
        int Y, U, V;
        rgb_to_yuv_cuda(R, G, B, &Y, &U, &V);
        
        // Store YUV values
        image_out[index] = Y;
        image_out[index + 1] = U;
        image_out[index + 2] = V;
    }
}

__global__ void computeHistogramKernelSimple(const unsigned char *image, int *histogram, int width, int height) {
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int numPixels = width * height;

    for (int i = globalId; i < numPixels; i += stride) {
        unsigned char y_value = image[i * 3];
        atomicAdd(&histogram[y_value], 1);
    }
}


__global__ void computeHistogramKernel(const unsigned char *image, int *histogram, int width, int height) {

    __shared__ int shared_histogram[256];
    
    int threadId = threadIdx.x;
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int numPixels = width * height;
    
    // Initialize shared memory histogram to zeros
    // Each thread initializes a portion of the histogram
    for (int i = threadId; i < 256; i += blockDim.x) {
        shared_histogram[i] = 0;
    }
    
    __syncthreads();
    
    // Each thread processes pixels with a stride pattern
    for (int i = globalId; i < numPixels; i += stride) {
        unsigned char y_value = image[i * 3];
        atomicAdd(&shared_histogram[y_value], 1);
    }
    
    // Wait for all threads in the block to finish updating shared histogram
    __syncthreads();
    
    // Merge partial histogram from shared memory to global histogram
    for (int i = threadId; i < 256; i += blockDim.x) {
        if (shared_histogram[i] > 0) { // Skip empty bins for efficiency
            atomicAdd(&histogram[i], shared_histogram[i]);
        }
    }
}

__global__ void computeCumulativeHistogramSimple(const int *histogram, int *cdf) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < 256; ++i) {
            sum += histogram[i];
            cdf[i] = sum;
        }
    }
}

__global__ void computeCumulativeHistogramAnother(const int *histogram, int *cdf) {
    extern __shared__ int temp[];  // Dynamically allocated shared memory
    
    int tid = threadIdx.x;
    
    // Load input into shared memory
    if (tid < 256) {
        temp[tid] = histogram[tid];
    }
    __syncthreads();
    
    // Upsweep (reduction phase)
    int offset = 1;
    for (int d = 128; d > 0; d >>= 1) {  // Start from 256/2
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            if (bi < 256) {  // don't access beyond array bounds
                temp[bi] += temp[ai];
            }
        }
        offset *= 2;
    }
    
    // Set last element to 0 for exclusive scan
    if (tid == 0) {
        int t = temp[255];  // Save last element value
        temp[255] = 0;      // Set to 0 for exclusive scan
    }
    __syncthreads();
    
    // Downsweep phase
    for (int d = 1; d < 256; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            if (bi < 256) {  // Boundary check
                int t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] += t;
            }
        }
    }
    __syncthreads();
    
    // Write result to output CDF with conversion to inclusive scan by adding the original histogram value
    if (tid < 256) {
        cdf[tid] = temp[tid] + histogram[tid];  // Convert exclusive to inclusive
    }
}

__global__ void calculateLuminanceLookupKernel(int *histogram_cumulative, int *luminance_lookup_table, int pixels) {
    __shared__ int min_value;

    int tid = threadIdx.x;

    // Initialize min_value to INT_MAX
    if (tid == 0) {
        min_value = INT_MAX;
    }
    __syncthreads();

    // Each thread checks its value
    if (tid < 256 && histogram_cumulative[tid] > 0 && histogram_cumulative[tid] < min_value) {
        atomicMin(&min_value, histogram_cumulative[tid]);
    }
    __syncthreads();

    // Calculate new luminance values
    if (tid < 256) {
        // Apply histogram equalization formula
        luminance_lookup_table[tid] = floor(((float)(histogram_cumulative[tid] - min_value) / (pixels - min_value)) * 255);
    }
}

__global__ void applyLuminanceAndConvertToRgbKernel(unsigned char *image, const int *luminance_lookup_table, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        int index = (row * width + col) * 3;

        // Get YUV values
        int Y = image[index];
        int U = image[index + 1];
        int V = image[index + 2];

        // Apply new luminance from lookup table
        Y = luminance_lookup_table[Y];

        // Convert back to RGB
        int R, G, B;
        U -= 128;
        V -= 128;

        // YUV to RGB conversion with correct coefficients
        R = (int)roundf(Y + 1.402f * V);
        G = (int)roundf(Y - 0.344136f * U - 0.714136f * V);
        B = (int)roundf(Y + 1.772f * U);

        // Clamp values to [0, 255]
        //R = max(0, min(255, R));
        //G = max(0, min(255, G));
        //B = max(0, min(255, B));

        R = R < 0 ? 0 : (R > 255 ? 255 : R);
        G = G < 0 ? 0 : (G > 255 ? 255 : G);
        B = B < 0 ? 0 : (B > 255 ? 255 : B);

        // Store RGB values
        image[index] = R;
        image[index + 1] = G;
        image[index + 2] = B;
    }
}

void rgb_to_yuv(int R, int G, int B, int *Y, int *U, int *V) {
    *Y = (int)roundf(0.299f * R + 0.587f * G + 0.114f * B);
    *U = (int)roundf(-0.168736f * R - 0.331264f * G + 0.5f * B) + 128;
    *V = (int)roundf(0.5f * R - 0.418688f * G - 0.081312f * B) + 128;
}

void yuv_to_rgb(int Y, int U, int V, int *R, int *G, int *B) {
    U -= 128;
    V -= 128;

    *R = (int)roundf(Y + 1.402f * V);
    *G = (int)roundf(Y - 0.344136f * U - 0.714136f * V);
    *B = (int)roundf(Y + 1.772f * U);

    // Ensuring that the values stay in range [0,255]
    *R = *R < 0 ? 0 : (*R > 255 ? 255 : *R);
    *G = *G < 0 ? 0 : (*G > 255 ? 255 : *G);
    *B = *B < 0 ? 0 : (*B > 255 ? 255 : *B);
}

void sequential(const unsigned char *image_in, unsigned char *image_out, int width, int height) {
    printf("Sequential execution time\n");
    // 1. Transform the image from RGB to YUV space
    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            int index = (i * width + j) * 3;

            int R, G, B;
            int Y, U, V;

            R = image_in[index];
            G = image_in[index + 1];
            B = image_in[index + 2];

            rgb_to_yuv(R, G, B, &Y, &U, &V);

            image_out[index] = Y;
            image_out[index + 1] = U;
            image_out[index + 2] = V;
        }
    }

    // 2. Compute the luminance histogram
    int histogram[256] = {0};
    int y_value;

    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            int index = (i * width + j) * 3;

            y_value = image_out[index];
            histogram[y_value]++;
        }
    }

    // 3. Calculate the cumulative histogram
    int histogram_cumulative[256] = {0};

    histogram_cumulative[0] = histogram[0];

    for (int i = 1; i < 256; i++){
        histogram_cumulative[i] = histogram_cumulative[i - 1] + histogram[i];
    }

    int luminance[256] = {0};
    int min_cumulative = INT_MAX;

    for (int i = 0; i < 256; i++) {
        if (histogram_cumulative[i] < min_cumulative) {
            if (histogram_cumulative[i] != 0) {
                min_cumulative = histogram_cumulative[i];
            }
        }
    }

    int pixels = width * height;

    for (int i = 0; i < 256; i++){
        luminance[i] = floor(((float)(histogram_cumulative[i] - min_cumulative) / (pixels - min_cumulative)) * 255);
    }

    // 5. Assign new luminance to each pixel
    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            int index = (i * width + j) * 3;

            image_out[index] = luminance[image_out[index]];
        }
    }

    // 6. Convert the image back to RGB colour space
    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            int index = (i * width + j) * 3;

            int Y, U, V;
            int R, G, B;

            Y = image_out[index];
            U = image_out[index + 1];
            V = image_out[index + 2];

            yuv_to_rgb(Y, U, V, &R, &G, &B);

            image_out[index] = R;
            image_out[index + 1] = G;
            image_out[index + 2] = B;
        }
    }
}

struct KernelTimings {
    float rgbToYuvTime;
    float histogramTime;
    float cdfTime;
    float lookupTableTime;
    float applyConvertTime;
};

KernelTimings parallel(unsigned char *image_in, unsigned char *image_out, int width, int height, int blockSizeInput, int histogramThreads) {
    cudaError_t error;
    int imageSize = width * height * 3 * sizeof(unsigned char);

    // Set up execution configuration
    dim3 blockSize(blockSizeInput, blockSizeInput);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    int histogramBlocks = min(16, (width * height + histogramThreads - 1) / histogramThreads);

    // Kernel timings
    float rgbToYuvTime = 0.0f;
    float histogramTime = 0.0f;
    float cdfTime = 0.0f;
    float lookupTableTime = 0.0f;
    float applyConvertTime = 0.0f;

    // Allocate device memory
    unsigned char *d_image_in, *d_image_out;
    error = cudaMalloc(&d_image_in, imageSize);
    if (error != cudaSuccess) {
        printf("Error allocating device memory for input image: %s\n", cudaGetErrorString(error));
        return {0, 0, 0, 0, 0}; // Return zeros on error
    }

    error = cudaMalloc(&d_image_out, imageSize);
    if (error != cudaSuccess) {
        printf("Error allocating device memory for output image: %s\n", cudaGetErrorString(error));
        cudaFree(d_image_in);
        return {0, 0, 0, 0, 0}; // Return zeros on error
    }

    // Copy input image to device
    cudaMemcpy(d_image_in, image_in, imageSize, cudaMemcpyHostToDevice);

    int *d_histogram;
    int* d_cdf;
    int *d_luminance_lookup_table;

    cudaMalloc(&d_luminance_lookup_table, 256 * sizeof(int));
    cudaMalloc(&d_cdf, 256 * sizeof(int));
    cudaMalloc(&d_histogram, 256 * sizeof(int));
    cudaMemset(d_histogram, 0, 256 * sizeof(int));

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // 1. Launch RGB to YUV conversion kernel
    rgbToYuvKernel<<<gridSize, blockSize>>>(d_image_in, d_image_out, width, height);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    rgbToYuvTime += milliseconds;

    cudaEventRecord(start);
    
    computeHistogramKernel<<<histogramBlocks, histogramThreads>>>(d_image_out, d_histogram, width, height);
    //computeHistogramKernelSimple<<<histogramBlocks, histogramThreads>>>(d_image_out, d_histogram, width, height);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    histogramTime += milliseconds;

    // 2. Calculate cumulative histogram (Bleloch scan)
    cudaEventRecord(start);

    computeCumulativeHistogramAnother<<<1, 256, 256 * sizeof(int)>>>(d_histogram, d_cdf);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cdfTime += milliseconds;

    // 3. Calculate new pixel luminances
    cudaEventRecord(start);

    calculateLuminanceLookupKernel<<<1, 256>>>(d_cdf, d_luminance_lookup_table, width * height);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    lookupTableTime += milliseconds;

    // 4. Apply new luminance to each pixel and convert back from YUV to RGB
    cudaEventRecord(start);

    applyLuminanceAndConvertToRgbKernel<<<gridSize, blockSize>>>(d_image_out, d_luminance_lookup_table, width, height);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    applyConvertTime += milliseconds;

    // Copy result back to host
    error = cudaMemcpy(image_out, d_image_out, imageSize, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        printf("Error copying output image to host: %s\n", cudaGetErrorString(error));
    }

    // Free device memory
    // Clean up timing events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_image_in);
    cudaFree(d_image_out);
    cudaFree(d_histogram);
    cudaFree(d_cdf);
    cudaFree(d_luminance_lookup_table);

    return {rgbToYuvTime, histogramTime, cdfTime, lookupTableTime, applyConvertTime};
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("USAGE: sample -seq|-par input_image output_image\n");
        exit(EXIT_FAILURE);
    }

    // Check execution flag
    bool runSequential = false;
    bool runParallel = false;
    
    if (strcmp(argv[3], "-seq") == 0) {
        runSequential = true;
    } else if (strcmp(argv[3], "-par") == 0) {
        runParallel = true;
    } else {
        printf("Invalid flag. Use -seq for sequential or -par for parallel execution.\n");
        exit(EXIT_FAILURE);
    }

    char szImage_in_name[255];
    char szImage_out_name[255];

    snprintf(szImage_in_name, 255, "%s", argv[1]);
    snprintf(szImage_out_name, 255, "%s", argv[2]);

    // Load image from file and allocate space for the output image
    int width, height, cpp;
    unsigned char *h_imageIn = stbi_load(szImage_in_name, &width, &height, &cpp, COLOR_CHANNELS);

    if (h_imageIn == NULL) {
        printf("Error reading loading image %s!\n", szImage_in_name);
        exit(EXIT_FAILURE);
    }
    printf("Loaded image %s of size %dx%d.\n", szImage_in_name, width, height);
    const size_t datasize = width * height * cpp * sizeof(unsigned char);
    unsigned char *h_imageOut = (unsigned char *)malloc(datasize);

    if (runSequential) {
        clock_t begin, end;
        float elapsed_ms;

        begin = clock();

        sequential(h_imageIn, h_imageOut, width, height);

        end = clock();
        elapsed_ms = ((float)(end - begin) / CLOCKS_PER_SEC) * 1000.0;

        printf("Sequential method time: %.3f milliseconds\n", elapsed_ms);
    }

    
    // Setup Thread organization
    //dim3 blockSize(16, 16);
    //dim3 gridSize((height-1)/blockSize.x+1,(width-1)/blockSize.y+1);
    //dim3 gridSize(1, 1);

    

    if (runParallel) {

        // For storing best configurations
        typedef struct {
            int blockSize;
            int histogramThreads;
            float bestTime;
        } OptimalConfig;

        // Parallel CUDA implementation
        printf("[BENCHMARK] Parallel execution time\n");

        OptimalConfig bestConfigs[5]; // One for each image size
        for (int i = 0; i < 5; i++) {
            bestConfigs[i].bestTime = FLT_MAX;
        }

        // Define histogram thread counts to test (powers of 2)
        int histogramThreads[] = {128, 256, 512, 1024};
        int numHistogramThreads = sizeof(histogramThreads) / sizeof(histogramThreads[0]);

        cudaEvent_t start, stop;
        float milliseconds = 0;
        float total_milliseconds = 0;
        int iterations = 100;

        for (int block = 8; block <= 32; block *= 2) {

            for (int t = 0; t < numHistogramThreads; t++) {

                int histogramThread = histogramThreads[t];
                //printf("Block size %i, Threads: %i\n", block * block, histogramThread);

                float total_rgbToYuvTime = 0.0f;
                float total_histogramTime = 0.0f;
                float total_cdfTime = 0.0f;
                float total_lookupTableTime = 0.0f;
                float total_applyConvertTime = 0.0f;

                for (int i = 0; i < iterations; i++) {
                    cudaEventCreate(&start);
                    cudaEventCreate(&stop);

                    cudaEventRecord(start);

                    KernelTimings timings = parallel(h_imageIn, h_imageOut, width, height, block, histogramThread);

                    total_rgbToYuvTime += timings.rgbToYuvTime;
                    total_histogramTime += timings.histogramTime;
                    total_cdfTime += timings.cdfTime;
                    total_lookupTableTime += timings.lookupTableTime;
                    total_applyConvertTime += timings.applyConvertTime;

                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);

                    // Print time
                    cudaEventElapsedTime(&milliseconds, start, stop);
                    total_milliseconds += milliseconds;
                }

                float avgRgbToYuv = total_rgbToYuvTime / iterations;
                float avgHistogram = total_histogramTime / iterations;
                float avgCdf = total_cdfTime / iterations;
                float avgLookup = total_lookupTableTime / iterations;
                float avgApply = total_applyConvertTime / iterations;
                float avgTotal = total_milliseconds / iterations;
                
                // For each kernel, determine if we need to update with best time
                if (avgRgbToYuv < bestConfigs[0].bestTime) {
                    bestConfigs[0].bestTime = avgRgbToYuv;
                    bestConfigs[0].blockSize = block * block;
                    bestConfigs[0].histogramThreads = histogramThread;
                }
                
                if (avgHistogram < bestConfigs[1].bestTime) {
                    bestConfigs[1].bestTime = avgHistogram;
                    bestConfigs[1].blockSize = block * block;
                    bestConfigs[1].histogramThreads = histogramThread;
                }
                
                if (avgCdf < bestConfigs[2].bestTime) {
                    bestConfigs[2].bestTime = avgCdf;
                    bestConfigs[2].blockSize = block * block;
                    bestConfigs[2].histogramThreads = histogramThread;
                }
                
                if (avgLookup < bestConfigs[3].bestTime) {
                    bestConfigs[3].bestTime = avgLookup;
                    bestConfigs[3].blockSize = block * block;
                    bestConfigs[3].histogramThreads = histogramThread;
                }
                
                if (avgApply < bestConfigs[4].bestTime) {
                    bestConfigs[4].bestTime = avgApply;
                    bestConfigs[4].blockSize = block * block;
                    bestConfigs[4].histogramThreads = histogramThread;
                }

                //printf("Kernel 1: RGB to YUV conversion       - Avg time: %.3f ms\n", total_rgbToYuvTime / iterations);
                //printf("Kernel 2: Histogram Computation       - Avg time: %.3f ms\n", total_histogramTime / iterations);
                //printf("Kernel 3: Cumulative Histogram        - Avg time: %.3f ms\n", total_cdfTime / iterations);
                //printf("Kernel 4: Luminance Lookup Table      - Avg time: %.3f ms\n", total_lookupTableTime / iterations);
                //printf("Kernel 5: Apply Luminance & RGB conv  - Avg time: %.3f ms\n", total_applyConvertTime / iterations);
                printf("[Block size %i, Threads %i] Total time: %.3f milliseconds\n", block * block, histogramThread, avgTotal);
                printf("--------------------------------------------------------------------------------\n");

                total_milliseconds = 0;
            }
        }

        const char* kernelNames[5] = {
            "RGB to YUV Conversion",
            "Histogram Computation",
            "Cumulative Histogram (CDF)",
            "Luminance Lookup Table",
            "Apply Luminance & Convert to RGB"
        };
    
        printf("\n========== BEST CONFIGURATIONS PER KERNEL ==========\n");
        for (int i = 0; i < 5; i++) {
            printf("Kernel %d (%s):\n", i + 1, kernelNames[i]);
            printf("  Best Time: %.3f ms\n", bestConfigs[i].bestTime);
            printf("  Block Size: %d\n", bestConfigs[i].blockSize);
            printf("  Histogram Threads: %d\n", bestConfigs[i].histogramThreads);
            printf("------------------------------------------------------\n");
        }

        // Clean-up events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    // Write the output file
    char szImage_out_name_temp[255];
    strncpy(szImage_out_name_temp, szImage_out_name, 255);
    char *token = strtok(szImage_out_name_temp, ".");
    char *FileType = NULL;
    while (token != NULL) {
        FileType = token;
        token = strtok(NULL, ".");
    }

    if (!strcmp(FileType, "png"))
        stbi_write_png(szImage_out_name, width, height, cpp, h_imageOut, width * cpp);
    else if (!strcmp(FileType, "jpg"))
        stbi_write_jpg(szImage_out_name, width, height, cpp, h_imageOut, 100);
    else if (!strcmp(FileType, "bmp"))
        stbi_write_bmp(szImage_out_name, width, height, cpp, h_imageOut);
    else
        printf("Error: Unknown image format %s! Only png, bmp, or bmp supported.\n", FileType);

    // Free device memory
    //checkCudaErrors(cudaFree(d_imageIn));
    //checkCudaErrors(cudaFree(d_imageOut));

    // Free host memory
    free(h_imageIn);
    free(h_imageOut);

    return 0;
}
