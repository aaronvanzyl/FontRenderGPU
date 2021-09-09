#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <string>
#include <vector>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
using namespace cv;

using namespace cooperative_groups;
using namespace std;

cudaError_t sumWithCuda(int* sum, const int* arr, unsigned int size);
cudaError_t bitmapMatch(int* total_overlap, const char* patch, const char* bitmap, unsigned int patchWidth, unsigned int patchHeight);
cudaError_t multiMapMatch(int* total_overlap, const char* image, const char* bitmaps,
	unsigned int n_bitmaps, unsigned int patchWidth, unsigned int patchHeight, unsigned int patchCountX, unsigned int patchCountY);

__global__ void matchKernel(int* total_overlap, const char* patch, const char* bitmap, unsigned int patchHeight)
{
	int x = threadIdx.x;
	int overlap = 0;

	for (int y = 0; y < patchHeight; y++) {
		char bitmap_val = bitmap[y * blockDim.x + x];
		char patch_val = patch[y * blockDim.x + x];
		overlap += bitmap_val == patch_val;
	}

#define FULL_MASK 0xffffffff
	for (int offset = 16; offset > 0; offset /= 2) {
		overlap += __shfl_down_sync(FULL_MASK, overlap, offset);
	}

	if (x == 0) {
		*total_overlap = overlap;
	}
}

// total_overlap[patch_idx, bitmap]
__global__ void multiMatchKernel(int* total_overlap, const char* image, const char* bitmaps,
	unsigned int n_bitmaps, unsigned int patchWidth, unsigned int patchHeight, unsigned int patchCountX, unsigned int patchCountY)
{
	int patchCount = patchCountX * patchCountY;
	int patchSize = patchWidth * patchHeight;
	int x = threadIdx.x;
	int bitmapi = blockIdx.y;

	for (int patchi = blockIdx.x; patchi < patchCount; patchi += gridDim.x) {
		int overlap = 0;
		int patchx = patchi % patchCountX;
		int patchy = patchi / patchCountX;

		for (int y = 0; y < patchHeight; y++) {
			char bitmap_val = bitmaps[bitmapi * patchSize + y * patchWidth + x];
			char patch_val = image[(patchy * patchHeight + y) * (patchCountX * patchWidth) + (patchx * patchWidth + x)];
			overlap += bitmap_val == patch_val;
		}

#define FULL_MASK 0xffffffff
		for (int offset = 16; offset > 0; offset /= 2) {
			overlap += __shfl_down_sync(FULL_MASK, overlap, offset);
		}

		if (x == 0) {
			total_overlap[patchi * n_bitmaps + bitmapi] = overlap;
		}
	}
}

__global__ void sumKernel(int* sum, const int* arr)
{
	int i = threadIdx.x;
	int val = arr[i];
#define FULL_MASK 0xffffffff
	for (int offset = 16; offset > 0; offset /= 2) {
		val += __shfl_down_sync(FULL_MASK, val, offset);
	}
	if (i == 0) {
		*sum = val;
	}
}


//void memory_test(vector<int>& arr) {
//    arr.resize(10);
//    arr[7] = 2000;
//
//    char* test_bytes = new char[4]{ 1,2,3,4 };
//
//    //memcpy(&arr[0], test_bytes, 3);
//    copy(test_bytes, test_bytes + 3, arr.begin());
//
//    cout << "test:";
//    for (int x : arr) {
//        cout << x << " ";
//    }
//    cout << "\n";
//}


static void load_font_bitmaps(string font_file, int width, int height, vector<char>& bitmaps, vector<int>& unicode_vals, int& num_chars) {
	string numline;
	string bytesline;
	ifstream myfile(font_file);
	int size = width * height;

	if (myfile.is_open())
	{
		char* int_bytes = new char[4];

		myfile.read(int_bytes, 4);
		std::memcpy(&num_chars, int_bytes, 4);

		bitmaps.resize(size * num_chars);
		unicode_vals.resize(num_chars);

		cout << num_chars << "\n";

		for (int char_idx = 0; char_idx < num_chars; char_idx++) {
			myfile.read(int_bytes, 4);
			std::memcpy(&unicode_vals[char_idx], int_bytes, 4);

			//cout << unicode_val << "\n";

			myfile.read(&bitmaps[size * char_idx], size);
		}
		myfile.close();
	}

	else cout << "Unable to open file";
}

static void render(vector<char>& bitmaps, int width, int height, int index) {
	int offset = width * height * index;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int val = int(bitmaps[offset + y * width + x]);
			cout << (val ? "0" : ".");
		}
		cout << "\n";
	}
	cout << "\n";
}

void render_mat(Mat mat) {

	for (int y = 0; y < mat.rows; y++) {
		for (int x = 0; x < mat.cols; x++) {
			int val = int(mat.data[y * mat.cols + x]);
			cout << (val ? "0" : ".");
		}
		cout << "\n";
	}
	cout << "\n";
}


int main() {

	// --- SET UP OUTPUT STREAMS

	std::locale::global(std::locale(""));

	std::cin.imbue(std::locale());
	std::cout.imbue(std::locale());
	std::cerr.imbue(std::locale());
	std::wcin.imbue(std::locale());
	std::wcout.imbue(std::locale());
	std::wcerr.imbue(std::locale());

	// --- LOAD CHARS

	int patchWidth = 18;
	int patchHeight = 32;
	int patchSize = patchWidth * patchHeight;

	string font_file = "font_33-126_18x32";

	vector<char> bitmaps;
	vector<int> unicode_val;
	int num_chars;
	load_font_bitmaps(font_file, patchWidth, patchHeight, bitmaps, unicode_val, num_chars);
	cout << "Read " << num_chars << " chars\n";



	//for (int i = 0; i < num_chars; i++) {/*
	//	render(bitmaps, patchWidth, patchHeight, i);
	//}*/
	// 
	// 
	char* bitmaps_arr = new char[bitmaps.size()];
	std::copy(bitmaps.begin(), bitmaps.end(), bitmaps_arr);

	// --- LOAD IMAGE
	String imageName("C:/Users/aaron/Pictures/textrender/dragon3.png");
	Mat src_gray, scaled_img, src_binary, final_img;
	src_gray = imread(imageName, IMREAD_GRAYSCALE);
	if (src_gray.empty())
	{
		cout << "Cannot read the image: " << imageName << std::endl;
		return -1;
	}
	
	//cvtColor(src, src_gray, COLOR_BGR2GRAY); // Convert the image to Gray
	threshold(src_gray, src_binary, 0, 255, 1 + THRESH_OTSU);
	cout << src_binary.type() << " " << src_binary.channels() << "\n";

	double upscale = 1;
	resize(src_binary, scaled_img, Size(src_binary.cols * upscale, src_binary.rows * upscale));
	//Mat scaled_img = src_binary;

	int imgWidth = scaled_img.cols;
	int imgHeight = scaled_img.rows;

	int patchCountX = imgWidth / patchWidth;
	int patchCountY = imgHeight / patchHeight;
	int patchCount = patchCountX * patchCountY;
	//render_mat(scaled_img);
	Rect cropRegion(0, 0, patchCountX * patchWidth, patchCountY * patchHeight);

	// Crop the full image to that image contained by the rectangle myROI
	// Note that this doesn't copy the data
	//Mat cropped = scaled_img(cropRegion);

	//render_mat(final_img(cropRegion));

	Mat roiImage = Mat(scaled_img, cropRegion);
	roiImage.copyTo(final_img);
	//render_mat(final_image);


	char* img_arr = new char[final_img.total()];
	memcpy(img_arr, final_img.data, final_img.total());
	cout << "---\n";
	//cout << final_img.total() << "\n";
	//cout << patchCount * patchSize << "\n";
	cout << patchCountX << "x" << patchCountY << "\n";
	cout << patchWidth * patchCountX << "?=" << final_img.cols << "\n";
	cout << patchHeight * patchCountY << "?=" << final_img.rows << "\n";
	cout << patchSize * patchCount << "?=" << final_img.total() << "\n";
	cout << "---\n";

	//char* img_arr = new char[bitmaps.size()];
	//std::copy(bitmaps.begin(), bitmaps.end(), img_arr);

	//wchar_t c = wchar_t(225);
	//wcout << "---" << endl;
	//wcout << c << endl;
	//wcout << "---" << endl;

	// --- CALCULATE OVERLAPS

	int* totalOverlap = new int[patchCount * num_chars];
	cudaError_t cudaStatus = multiMapMatch(totalOverlap, img_arr, bitmaps_arr, num_chars, patchWidth, patchHeight, patchCountX, patchCountY);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	// --- DISPLAY RESULT

	for (int patchi = 0; patchi < patchCount; patchi++) {
		int bestMapi = 0;
		int bestOverlap = 0;
		for (int mapi = 0; mapi < num_chars; mapi++) {
			int overlap = totalOverlap[patchi * num_chars + mapi];
			if (overlap > bestOverlap) {
				bestMapi = mapi;
				bestOverlap = overlap;
			}
		}
		//cout << patchi << " " << bestMapi << " " << bestOverlap << "\n";
		wchar_t c = wchar_t(unicode_val[bestMapi]);
		wcout << c;
		if ((patchi + 1) % patchCountX == 0) {
			wcout << "\n";
		}
	}

	//cout << *totalOverlap << "\n";


	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t sumWithCuda(int* sum, const int* arr, unsigned int size)
{
	int* dev_sum = 0;
	int* dev_arr = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_sum, 1 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_arr, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_arr, arr, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	sumKernel << <1, size >> > (dev_sum, dev_arr);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(sum, dev_sum, 1 * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "return cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_arr);
	cudaFree(dev_sum);

	return cudaStatus;
}

cudaError_t bitmapMatch(int* total_overlap, const char* patch, const char* bitmap, unsigned int patchWidth, unsigned int patchHeight)
{
	char* dev_patch = 0;
	char* dev_bitmap = 0;
	int* dev_total_overlap = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_patch, patchWidth * patchHeight * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_bitmap, patchWidth * patchHeight * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_total_overlap, 1 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_patch, patch, patchWidth * patchHeight * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 1 failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_bitmap, bitmap, patchWidth * patchHeight * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 2 failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	matchKernel << <1, patchWidth >> > (dev_total_overlap, dev_patch, dev_bitmap, patchHeight);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(total_overlap, dev_total_overlap, 1 * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "return cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_total_overlap);
	cudaFree(dev_patch);
	cudaFree(dev_bitmap);

	return cudaStatus;
}

cudaError_t multiMapMatch(int* total_overlap, const char* image, const char* bitmaps,
	unsigned int n_bitmaps, unsigned int patchWidth, unsigned int patchHeight, unsigned int patchCountX, unsigned int patchCountY)
{
	char* dev_image = 0;
	char* dev_bitmaps = 0;
	int* dev_total_overlap = 0;
	cudaError_t cudaStatus;
	int patchSize = patchWidth * patchHeight;
	int patchCount = patchCountX * patchCountY;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_image, patchSize * patchCount * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_bitmaps, patchSize * n_bitmaps * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_total_overlap, n_bitmaps * patchCount * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_image, image, patchSize * patchCount * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 1 failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_bitmaps, bitmaps, patchSize * n_bitmaps * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 2 failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	multiMatchKernel << <dim3(500, n_bitmaps), patchWidth >> > (dev_total_overlap, dev_image, dev_bitmaps, n_bitmaps, patchWidth, patchHeight, patchCountX, patchCountY);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(total_overlap, dev_total_overlap, n_bitmaps * patchCount * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "return cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_total_overlap);
	cudaFree(dev_image);
	cudaFree(dev_bitmaps);

	return cudaStatus;
}
