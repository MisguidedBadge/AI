#ifndef CONV_FILTER_H
#define CONV_FILTER_H

#include "libraries.h"
using namespace std;
/* Convolutional Filter Class
	- Main thing is that this class will create a functioning convolutional filter
	- Convolutions start with Randomized kernels N matrices of MxM size
	- 
*/
typedef vector<vector<float>>	kernel_array;		// Array of kernels
class ConvolutionFilter
{
public:
	// Class properties
	vector<kernel_array> kernels;
	// Input[batch][channel][pixels]
	vector<vector<vector<float>>>	output;			// Activation Function of convolutional filter
	vector<vector<vector<float>>>	*input;			// Input Matrix
	vector<vector<vector<float>>>	layer_error;	// Error for that layer [batch][channel][value]
	vector<vector<vector<float>>>	dcz;
	vector<vector<vector<float>>>	dw;				// delta w to change weights after backpropagation
	int		channels;					// number of channels for the input
	int		input_size;
	int		channel_size;
	int		batch_size;
	int		height;						// image height
	int		width;						// image width
	int		num_filters;				//
	int		kernel_tsize;				// total kernel size Ex. 3x3 = 9
	int		kernel_size;				// Kernel size Ex. 3
	int		stride_x;
	int		stride_y;
	int		border;						// border size
	float	learning_rate;				// 
	float	error;						// Error value computed
	// Constructor
	ConvolutionFilter(
					int batch,
					int channels,
					int height,
					int width,
					int num_kernels,
					int kernel_size,
					int stride_x,
					int stride_y,
					float(*Activation)(float x),
					float learningrate);
	// Destructor
	~ConvolutionFilter();
	// Initialize Kernel Values to random values
	void InitializeKernel();
	// Load Input image into class
	void LoadImage(vector<vector<vector<float>>> *input);
	// Feed network values forward
	void FeedForward();
	// Backpropagation
	void Backpropagation(vector<vector<vector<float>>> &Error);
	// Return Output
	vector<vector<vector<float>>> Output();
	// Update Weights
	void UpdateWeights();
private:
	// Properties
	float (*Activate)(float x);
	// Perform Zero padding on the image
	vector<float> ZeroPad(vector<float> image);
	// Perform Convolution on the image
	vector<float> Convolve(int batch, int filter);
	// Dot Product
	float Dot(int batch, int filter, int channel, int height, int width);
	// Deconvolve
	void DeConvolve(int batch, int filters, vector<vector<vector<float>>>& Error);
	// Backwards Dot to change the dw
	void InDot(int batch, int filter, int channel, int height, int width, vector<vector<vector<float>>>& Error);
};

#endif
