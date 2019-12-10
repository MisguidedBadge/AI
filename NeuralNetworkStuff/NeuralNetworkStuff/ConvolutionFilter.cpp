#include "ConvolutionFilter.h"

/* ConvolutionFilter(Number Kernels, Activation Function, LearningRate)
	- Take in inputs, activation function, and learningrate
	- Run a random assignment for the kernel
	- Image and kernel are a 1 dimensional contigous array
*/
ConvolutionFilter::ConvolutionFilter(
									int batch,
									int channels,
									int height,
									int width,
									int num_filters,
									int kernel_size,
									int stride_x,
									int stride_y,
									float(*Activation)(float x),
									float learningrate)	{
	// intermidiate variable
	int i, tsize, j;
	// Intialize class variables
	this->channels = channels;
	this->num_filters = num_filters;
	this->learning_rate = learningrate;
	this->kernel_tsize = kernel_size * kernel_size;
	this->kernel_size = kernel_size;
	this->stride_x = stride_x;
	this->stride_y = stride_y;
	this->border = kernel_size / 2;			// Use odd numbered Kernel size
	this->height = height;					// Height and Width set in Load function
	this->width = width;
	this->error = 0;
	// Preallocate vector sizes
	// Input
	// Kernel and Delta Weights [channels][filters per channel][Kernel Total Size]
	this->kernels.resize(this->channels);
	this->dw.resize(this->channels);
	for (i = 0; i < this->channels; i++)
	{
		this->dw[i].resize(num_filters);
		this->kernels[i].resize(num_filters);
		for (j = 0; j < num_filters; j++)
		{
			this->kernels[i][j].resize(kernel_tsize);
			this->dw[i][j].resize(kernel_tsize);
		}
	}
	// Output
	this->output.resize(batch);
	for(int i = 0; i < this->output.size(); i++)
		this->output[i].resize(this->num_filters);
	// Activation function
	this->Activate = Activation;
	// BackPropagation Stuff
	this->layer_error.resize(batch);
	for (int i = 0; i < batch; i++)
	{
		this->layer_error[i].resize(channels);
		for (int j = 0; j < channels; j++)
			this->layer_error[i][j].resize(height * width);	// input size;
	}

	this->dcz.resize(batch);	// DCZ = [batch][channels][data] error matrix""
	for (int i = 0; i < batch; i++)
	{
		this->dcz[i].resize(num_filters);
		for (int j = 0; j < num_filters; j++)
			this->dcz[i][j].resize(height * width);
	}
}

/* Destructor
- Shrink vectors back to nothing
*/
ConvolutionFilter::~ConvolutionFilter()
{
	// dissolve vectors
	this->kernels.shrink_to_fit();
}

/* Initialize Kernels
- Setup the CNN weights to random values
*/
void ConvolutionFilter::InitializeKernel()
{
	for (int i = 0; i < this->channels; i++)
	{
		// Kernel selection
		for (int j = 0; j < this->num_filters; j++)
		{
			// Kernel matrix
			for (int k = 0; k < this->kernel_tsize; k++)
			{
				this->kernels[i][j][k] = (float)((rand() % 10000 / 10000.00));
				//this->kernels[i][j][k] = 1;
			}

		}
	}
}

/* Load Image
- Input: Image (pass by reference *lowers memory usage by 35%*)
*/
void ConvolutionFilter::LoadImage(vector<vector<vector<float>>> *input)
{
	this->input = input;
	batch_size = this->input->size();
	channel_size = this->input[0][0].size();
	input_size = this->input[0][0][0].size();
}

/* Zero Pad
- Pad image before applying the filter
- Pad in reverse order to reduce complications with array indexing
*/
vector<float> ConvolutionFilter::ZeroPad(vector<float> image)
{
	int i = 0;
	int k = 0;
	// Vector of zeroes - Maximum size is the size of image width
	vector<float> padw  (this->width + 2*this->border, (float)0.0);
	vector<float> pad	(this->border, (float)0.0);

	// insert stuff at the beginnning
	for (i = 0; i < border; i++)
		image.insert(image.end(), padw.begin(), padw.end());

	// Insert stuff on the left and right
	for (i = this->height; i > 0; i--)
	{
		image.insert((image.begin() + i * width), pad.begin(), pad.end());
		image.insert((image.begin() + (i - 1) * width), pad.begin(), pad.end());
	}

	// Insert stuff at the beginning
	for (i = 0; i < border; i++)
		image.insert(image.begin(), padw.begin(), padw.end());

	return image;
}

/* Feed Forward
- Follow procedures to do a successful feed forward
- [Convolve -> Activate]
*/
void ConvolutionFilter::FeedForward()
{
	int con_i;
	// Perform convolution through each filter
	concurrency::parallel_for(0, (int)this->input->size(), [&](float con_i) {
		/*for(int i = 0; i < this->input->size(); i++)*/
		for (int j = 0; j < this->num_filters; j++)
			this->output[con_i][j] = Convolve(con_i, j);
		});


	// Activate output values
	for(int k = 0; k < this->input->size(); k++)
		for(int i = 0; i < this->num_filters; i++)
			for (int j = 0; j < this->output[0][0].size(); j++)
				this->output[k][i][j] = (float)this->Activate(this->output[k][i][j]);
	// Normalize the output
	Normalize(this->output, 1, 0);
}

/* Convolve the Input
- Input is the kernel/filter
- Convolve with respect to each filter
-- Start from top left (non padded)
*/
vector<float> ConvolutionFilter::Convolve(int batch, int filter)
{
	// local variables
	int i, j, k;
	float sum;
	vector<float> output;
	//std::cout << this->input[0][0][0].size() << std::endl;
		// loop through the height starting after the border
		for (j = 0; j < (this->height); j += this->stride_y)
		{
			// loop through the width
			for (k = 0; k < (this->width); k += this->stride_x)
			{
				sum = 0;
				// for each channel
				for (i = 0; i < this->channels; i++)
				{
					// Centered Perform Dot
					sum += Dot(batch, filter, i, j, k);

				}
				output.insert(output.end(), sum);
			}
			
		}
	return output;
}


/* Dot Product
- Input: filter number, channel to be filtered, (x and y) of center
- Output: float weighted sum
*/
float ConvolutionFilter::Dot(int batch, int filter, int channel, int height, int width)
{
	float weighted_sum = 0;
	float tval = 0;

	int i, j, k;
	// increment through the kernel/filter indices
	k = 0;
	// height
	for (i = -border; i <= border; i++)
	{
		// width
		for (j = -border; j <= border; j++)
		{
			if (i + height < 0 || i + height > this->height - 1)
				tval = 0;
			else if (j + width < 0 || j + width > this->width - 1)
				tval = 0;
			else
				tval = this->input[0][batch][channel][(width + j) + ((height + i) * (this->width))] * this->kernels[channel][filter][k++];
			weighted_sum += tval;
		}
		
	}

	return weighted_sum;
}

/* Backpropagation Function
	- Perform backpropagation through a weird convoluted way of decovoluting
*/
void ConvolutionFilter::Backpropagation(vector<vector<vector<float>>> &Error)
{
	// (Error) [batch][channels][values]
	// Zero the delta weight values out
	for (int i = 0; i < channels; i++)
		for (int j = 0; j < num_filters; j++)
			for (int k = 0; k < kernel_tsize; k++)
				this->dw[i][j][k] = 0;	// Zero the values out
	// Reset This layer's error
	for (int i = 0; i < this->batch_size; i++)
		for (int j = 0; j < this->channel_size; j++)
			for (int k = 0; k < this->input_size; k++)
				this->layer_error[i][j][k] = 0;
	// Activation function first
	// DRelu
	for (int i = 0; i < Error.size(); i++)
		for (int k = 0; k < Error[0].size(); k++)
			for(int j = 0; j < Error[0][0].size(); j++)
				this->dcz[i][k][j] = Error[i][k][j] * DRelu(output[i][k][j]);		// Error component * by the Z value (before activation)			
	// Deconvolve
	// Perform de-convolution through each filter
	concurrency::parallel_for(0, (int)batch_size, [&](float con_i) {
		//for (int i = 0; i < this->input->size(); i++)
		for (int j = 0; j < this->num_filters; j++)
			DeConvolve(con_i, j, Error); });

	// Normalize Delta weights to prevent explosion in big images
	// Mess around with normalization to see if we can get better output
	//
	Normalize(this->layer_error, 1, 0);
	Normalize(this->dw, 1, 0);
	//for(int i = 0; i < this->dw.size(); i++)
	//	for(int j = 0; j < this->dw[i].size(); j++)
	//		for(int k = 0; k < this->dw[i][j].size(); k++)
	//			this->dw[i][j][k] = this->dw[i][j][k] * 10;


} 

/* Backpropagation Deconvolution
	-- Select the center pixel
*/
void ConvolutionFilter::DeConvolve(int batch, int filter, vector<vector<vector<float>>>& Error)
{
	// local variables
	int i, j, k;
	//std::cout << this->input[0][0][0].size() << std::endl;
		// loop through the height starting after the border
	for (j = 0; j < (this->height); j += this->stride_y)
	{
		// loop through the width
		for (k = 0; k < (this->width); k += this->stride_x)
		{
			// for each channel
			for (i = 0; i < this->channels; i++)
			{
				// Centered Perform Dot
				InDot(batch, filter, i, j, k, Error);

			}
		}

	}

}

/* BackDot

*/
void ConvolutionFilter::InDot(int batch, int filter, int channel, int height, int width, vector<vector<vector<float>>>& Error)
{
	int i, j, k;
	// increment through the kernel/filter indices
	// height
	k = 0;
	for (i = -border; i <= border; i++)
	{
		// width
		for (j = -border; j <= border; j++)
		{
			// dz/dw
			if( i + height >= 0 && i + height < this->height)
				if (j + width >= 0 && j + width < this->width)
				{
					// Multiply the pixel value of the convolution by the error value to get dz/dw
					this->dw[channel][filter][k] += this->input[0][batch][channel][(width + j) + ((height + i) * (this->width))]		// 
						* Error[batch][filter][((width) / this->stride_x) + ((height) * (this->width) / stride_y)];	//
					// DC/DZ multiplied by weight and accumulate
					this->layer_error[batch][channel][(width + j) + ((height + i) * (this->width))] += this->kernels[channel][filter][k] 
											* Error[batch][filter][((width) / this->stride_x) + ((height) * (this->width) / stride_y)];
				}
			k++;
		}

	}


}

/* Update the convolutional filter weights
*/
void ConvolutionFilter::UpdateWeights()
{
	// (Error) [batch][channels][values]
	// Zero the delta weight values out
	for (int i = 0; i < this->channels; i++)
		for (int j = 0; j < this->num_filters; j++)
			for (int k = 0; k < this->kernel_tsize; k++)
				this->kernels[i][j][k] = this->kernels[i][j][k] - this->learning_rate * this->dw[i][j][k];

}

/* Return the Output Image
*/
vector<vector<vector<float>>> ConvolutionFilter::Output()
{
	return this->output;
}