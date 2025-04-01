#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include "helper_cuda.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

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

void rgb_to_yuv(int R, int G, int B, int *Y, int *U, int *V) {
    *Y = (0.299 * R + 0.587 * G + 0.114 * B);
    *U = (-0.168736 * R - 0.331264 * G + 0.5 * B) + 128;
    *V = (0.5 * R - 0.418688 * G - 0.081312 * B) + 128;
}

void yuv_to_rgb(int Y, int U, int V, int *R, int *G, int *B) {
    U -= 128;
    V -= 128;

    *R = (Y + 1.402 * V);
    *G = (Y - 0.344136 * U - 0.714136 * V);
    *B = (Y + 1.772 * U);

    // Ensuring that the values stay in range [0,255]
    *R = *R < 0 ? 0 : (*R > 255 ? 255 : *R);
    *G = *G < 0 ? 0 : (*G > 255 ? 255 : *G);
    *B = *B < 0 ? 0 : (*B > 255 ? 255 : *B);
}

void sequential(const unsigned char *image_in, unsigned char *image_out, int width, int height) {
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

    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            int index = (i * width + j) * 3;

            histogram[image_out[index]]++;
        }
    }    

    // 3. Calculate the cumulative histogram
    int histogram_cumulative[256] = {0};

    histogram_cumulative[0] = histogram[0];
    
    for (int i = 1; i < 256; i++){
        histogram_cumulative[i] = histogram_cumulative[i - 1] + histogram[i];
    }    

    // 4. Calculate new pixel luminances from original luminances based on the histogram equalization formula
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

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("USAGE: sample input_image output_image\n");
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

    sequential(h_imageIn, h_imageOut, width, height);

    // Setup Thread organization
    dim3 blockSize(16, 16);
    dim3 gridSize((height-1)/blockSize.x+1,(width-1)/blockSize.y+1);
    //dim3 gridSize(1, 1);

    unsigned char *d_imageIn;
    unsigned char *d_imageOut;

    // Allocate memory on the device
    checkCudaErrors(cudaMalloc(&d_imageIn, datasize));
    checkCudaErrors(cudaMalloc(&d_imageOut, datasize));

    // Use CUDA events to measure execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Copy image to device and run kernel
    /*cudaEventRecord(start);
    checkCudaErrors(cudaMemcpy(d_imageIn, h_imageIn, datasize, cudaMemcpyHostToDevice));
    copy_image<<<gridSize, blockSize>>>(d_imageIn, d_imageOut, width, height, cpp);
    checkCudaErrors(cudaMemcpy(h_imageOut, d_imageOut, datasize, cudaMemcpyDeviceToHost));
    getLastCudaError("copy_image() execution failed\n");
    cudaEventRecord(stop);
    */
    cudaEventSynchronize(stop);

    // Print time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel Execution time is: %0.3f milliseconds \n", milliseconds);

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
    checkCudaErrors(cudaFree(d_imageIn));
    checkCudaErrors(cudaFree(d_imageOut));

    // Clean-up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free host memory
    free(h_imageIn);
    free(h_imageOut);

    return 0;
}
