#ifndef CONV_FILTER_H
#define CONV_FILTER_H

#include "libraries.h"
using namespace std;
/* Convolutional Filter Class
	- Main thing is that this class will create a functioning convolutional filter
	- Convolutions start with Randomized kernels N matrices of MxM size
	- 
*/
typedef vector<vector<float>> kernel;	// Kernel matrix
class ConvolutionFilter
{
public:
	// Class properties
	vector<kernel>	kernel_array;		// Array of kernels
	vector<vector<float>>	outputs;	// Activation Function of convolutional filter
	vector<vector<float>>	input;		// Input Matrix
	int		num_kernels;				// 
	float	learning_rate;				// 
	int		kernel_size;
	float	error;						// Error value computed
	// Constructor
	ConvolutionFilter(vector<vector<float>> input,
					int num_kernels,
					int kernel_size,
					float(*Activation)(float x),
					float learningrate);
	// Destructor
	~ConvolutionFilter();
	// Initialize Kernel Values to random values
	void InitializeKernel(int num_kernels, int kernel_size);
private:
	// Properties
	float (*Activate)(float x);

	// Activation
	//void Convolve();

	// backpropagation
	// Error

};

#endif
