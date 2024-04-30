#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Define the convolution kernel for upscaling
float kernel[3][3] = {
    {0.0625, 0.125, 0.0625},
    {0.125,  0.25,  0.125},
    {0.0625, 0.125, 0.0625}
};

// Function to perform 2D convolution
void convolve(int width, int height, int channels, unsigned char* in, unsigned char* out) {
    #pragma omp parallel for collapse(2)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            for (int c = 0; c < channels; c++) {
                float sum = 0.0;
                for (int j = -1; j <= 1; j++) {
                    for (int i = -1; i <= 1; i++) {
                        sum += in[((y + j) * width + (x + i)) * channels + c] * kernel[j + 1][i + 1];
                    }
                }
                out[(y * width + x) * channels + c] = (unsigned char)sum;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s input_image output_image\n", argv[0]);
        return 1;
    }

    char *input_image_path = argv[1];
    char *output_image_path = argv[2];

    FILE *file = fopen(input_image_path, "rb");
    if (!file) {
        printf("Failed to open image file %s\n", input_image_path);
        return 1;
    }

    int width, height, channels;
    unsigned char *image = stbi_load_from_file(file, &width, &height, &channels, 0);
    fclose(file);
    if (!image) {
        printf("Failed to load image %s\n", input_image_path);
        return 1;
    }

    // Allocate memory for the output image
    int new_width = width * 2;
    int new_height = height * 2;
    unsigned char *output_image = (unsigned char *)malloc(new_width * new_height * channels);

    // Apply 2D convolution to upscale the image
    convolve(width, height, channels, image, output_image);

    // Write the output image
    stbi_write_png(output_image_path, new_width, new_height, channels, output_image, new_width * channels);

    // Free memory
    stbi_image_free(image);
    free(output_image);

    printf("Image upscaled and saved to %s\n", output_image_path);

    return 0;
}

