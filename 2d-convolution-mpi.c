#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>


float kernel[3][3] = {
    {0.0625, 0.125, 0.0625},
    {0.125,  0.25,  0.125},
    {0.0625, 0.125, 0.0625}
};


void convolve(int width, int height, int channels, unsigned char* in, unsigned char* out) {
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
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) {
            printf("Usage: %s input_image output_image\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    char *input_image_path = argv[1];
    char *output_image_path = argv[2];

    int width, height, channels;
    unsigned char *image;

    if (rank == 0) {

        int status = stbi_load(input_image_path, &width, &height, &channels, 0);
        if (!status) {
            printf("Failed to load image %s\n", input_image_path);
            MPI_Finalize();
            return 1;
        }
        image = (unsigned char *)malloc(width * height * channels * sizeof(unsigned char));
        stbi_image_free(image);
    }


    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int portion_height = height / size;
    int portion_size = portion_height * width * channels;

    unsigned char *portion = (unsigned char *)malloc(portion_size * sizeof(unsigned char));

    MPI_Scatter(image, portion_size, MPI_UNSIGNED_CHAR, portion, portion_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    unsigned char *output_portion = (unsigned char *)malloc(portion_size * sizeof(unsigned char));
    convolve(width, portion_height, channels, portion, output_portion);
    MPI_Gather(output_portion, portion_size, MPI_UNSIGNED_CHAR, image, portion_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Write the output image on rank 0
    if (rank == 0) {
        stbi_write_png(output_image_path, width, height, channels, image, width * channels);
        printf("Image upscaled and saved to %s\n", output_image_path);
    }

    // Free allocated memory
    free(portion);
    free(output_portion);

    MPI_Finalize();
    return 0;
}
