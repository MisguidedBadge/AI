#include "ConvolutionFilter.h"

/* ConvolutionFilter
	- Take in inputs, activation function, and learningrate
	- Run a random assignment for the kernel
*/
ConvolutionFilter::ConvolutionFilter(vector<vector<float>> input,
	int num_kernels,
	int kernel_size,
	float(*Activation)(float x),
	float learningrate)	{
	// intermidiate variable
	int i, tsize;
	// Intialize class variables
	this->num_kernels = num_kernels;
	this->learning_rate = learningrate;
	this->kernel_size = kernel_size;
	// Preallocate vector size
	this->kernel_array.resize(num_kernels, kernel(kernel_size));
	

}

ConvolutionFilter::~ConvolutionFilter()
{
	// dissolve vectors
	this->kernel_array.shrink_to_fit();
}

// initialize random weights to kernels
void ConvolutionFilter::InitializeKernel(int num_kernels, int kernel_size)
{
	// kernel selection
	for (int i = 0; i < this->num_kernels; i++)
	{
		for (int j = 0; j < this->kernel_size; j++)
		{
			for (int k = 0; k < this->kernel_size; k++)
			{
				this->kernel_array[i][j][k] = (float)((rand() % 100 / 100));
			}
		}

	}

}