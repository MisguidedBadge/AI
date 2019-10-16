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
	this->error = 0;
	// Preallocate vector size
	this->kernel_array.resize(num_kernels);
	for (int i = 0; i < num_kernels; i++)
		this->kernel_array[i].resize(kernel_size, vector<float>(kernel_size, 0));
	// Activation function
	this->Activate = Activation;
	
}

ConvolutionFilter::~ConvolutionFilter()
{
	// dissolve vectors
	this->kernel_array.shrink_to_fit();
}

/* initialize random weights to kernels

*/
void ConvolutionFilter::InitializeKernel(int num_kernels, int kernel_size)
{
	// Kernel selection
	for (int i = 0; i < this->num_kernels; i++)
	{
		// Kernel row
		for (int j = 0; j < this->kernel_size; j++)
		{
			// Kernel column
			for (int k = 0; k < this->kernel_size; k++)
			{
				// Random kernel weights
				this->kernel_array[i][j][k] = (float)((rand() % 100 / 100.00));
			}
		}

	}
}