#include <stdio.h>
#include <string.h>
#include <png++/png.hpp>

int THREAD_PER_DIM = 32;

struct
{
    size_t width;
    size_t height;
    unsigned char *data;
} typedef Image;

/**
 * @brief Reads an image from a file and returns an Image object.
 *
 * @param filename The path to the image file.
 * @return Image* A pointer to the Image object containing the image data.
 */
Image *read_image(const char *filename)
{
    png::image<png::rgb_pixel> image(filename);

    size_t width = image.get_width();
    size_t height = image.get_height();

    auto pixbuf = image.get_pixbuf();

    unsigned char *data = (unsigned char *)malloc(width * height * 3);

    for (size_t row = 0; row < height; row++)
    {
        for (size_t col = 0; col < width; col++)
        {
            auto pixel = pixbuf[row][col];
            data[(row * width + col) * 3] = pixel.red;
            data[(row * width + col) * 3 + 1] = pixel.green;
            data[(row * width + col) * 3 + 2] = pixel.blue;
        }
    }

    Image *img = (Image *)malloc(sizeof(Image));
    img->width = width;
    img->height = height;
    img->data = data;

    return img;
}

/**
 * @brief Initializes a grayscale image with the specified width and height.
 *
 * @param width The width of the image.
 * @param height The height of the image.
 * @return A pointer to the initialized Image structure.
 */
Image *initialize_grayscale_image(size_t width, size_t height)
{
    Image *img = (Image *)malloc(sizeof(Image));
    img->width = width;
    img->height = height;
    img->data = (unsigned char *)malloc(width * height);

    for (size_t i = 0; i < width * height; i++)
    {
        img->data[i] = 0;
    }

    return img;
}

/**
 * Prints the RGB values and corresponding gray value of each pixel in the given images.
 *
 * @param colour The input image containing RGB values.
 * @param gray The output image containing gray values.
 */
void print_image(const Image *colour, const Image *gray)
{
    for (size_t i = 0; i < colour->width * colour->height; i++)
    {
        printf("R: %u, G: %u, B: %u, Gray Value: %u\n", colour->data[i * 3], colour->data[i * 3 + 1], colour->data[i * 3 + 2], gray->data[i]);
    }
}

__global__ void grayscale(const unsigned char *input, unsigned char *output, size_t width, size_t height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height)
    {
        int index = row * width + col;
        unsigned char r = input[index * 3];
        unsigned char g = input[index * 3 + 1];
        unsigned char b = input[index * 3 + 2];

        output[index] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}

void grayscale_cpu(const unsigned char *input, unsigned char *output, size_t width, size_t height)
{
    for (size_t row = 0; row < height; row++)
    {
        for (size_t col = 0; col < width; col++)
        {
            int index = row * width + col;
            unsigned char r = input[index * 3];
            unsigned char g = input[index * 3 + 1];
            unsigned char b = input[index * 3 + 2];

            output[index] = 0.21f * r + 0.71f * g + 0.07f * b;
        }
    }
}

int main(int argc, char **argv)
{
    bool cuda = false;
    if (argc > 1)
    {
        std::string arg1 = argv[1];
        if (arg1 == "cuda")
        {
            cuda = true;
        }
        else
        {
            cuda = false;
        }
    }
    Image *lenna_h = read_image("lenna.png");
    Image *gray_h = initialize_grayscale_image(lenna_h->width, lenna_h->height);

    if (cuda)
    {
        printf("Running on GPU\n");
        unsigned char *lenna_data_d;
        unsigned char *gray_data_d;
        cudaError_t error;

        error = cudaMalloc((void **)&lenna_data_d, lenna_h->width * lenna_h->height * 3);
        if (error != cudaSuccess)
        {
            printf("@ line %d : error allocating memory for lenna data: %s\n", __LINE__, cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }

        error = cudaMalloc((void **)&gray_data_d, lenna_h->width * lenna_h->height);
        if (error != cudaSuccess)
        {
            printf("@ line %d : error allocating memory for gray data: %s\n", __LINE__, cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }

        cudaMemcpy(lenna_data_d, lenna_h->data, lenna_h->width * lenna_h->height * 3, cudaMemcpyHostToDevice);
        cudaMemcpy(gray_data_d, gray_h->data, lenna_h->width * lenna_h->height, cudaMemcpyHostToDevice);

        dim3 blocksPerGrid(ceil(lenna_h->width) / THREAD_PER_DIM, ceil(lenna_h->height) / THREAD_PER_DIM);
        dim3 threadsPerBlock(THREAD_PER_DIM, THREAD_PER_DIM);

        grayscale<<<blocksPerGrid, threadsPerBlock>>>(lenna_data_d, gray_data_d, lenna_h->width, lenna_h->height);

        cudaMemcpy(gray_h->data, gray_data_d, lenna_h->width * lenna_h->height, cudaMemcpyDeviceToHost);
        cudaFree(lenna_data_d);
        cudaFree(gray_data_d);

    }
    else
    {
        printf("Running on CPU\n");
        grayscale_cpu(lenna_h->data, gray_h->data, lenna_h->width, lenna_h->height);
    }
    return 0;
}