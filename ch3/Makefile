CC = nvcc
CFLAGS = -std=c++11
OUTPUT_DIR = build

all: grayscale

grayscale: 
	mkdir -p $(OUTPUT_DIR)
	$(CC) $(CFLAGS) -c grayscale.cu `libpng-config --cflags` -o $(OUTPUT_DIR)/grayscale.o
	$(CC) $(CFLAGS) -o $(OUTPUT_DIR)/grayscale $(OUTPUT_DIR)/grayscale.o `libpng-config --ldflags`

clean:
	rm -rf $(OUTPUT_DIR)
