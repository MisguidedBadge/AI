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
	/*this->input.resize(this->channels);*/
	this->kernels.resize(this->channels);
	for (i = 0; i < this->channels; i++)
	{
		this->kernels[i].resize(num_filters);
		for (j = 0; j < num_filters; j++)
			this->kernels[i][j].resize(kernel_tsize);
	}
	// Output
	this->output.resize(batch);
	for(int i = 0; i < this->output.size(); i++)
		this->output[i].resize(this->num_filters);
	// Activation function
	this->Activate = Activation;
	
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
				this->kernels[i][j][k] = (float)((rand() % 100 / 10000.00));
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

	// First value is 0 always
	/*float val = this->input[0][1][0][0];*/
	//for(int i = 0; i < this->channels; i++)
	//	this->input[i] = ZeroPad(input[i]);
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
	// Perform convolution through each filter
	for(int i = 0; i < this->input->size(); i++)
		for (int j = 0; j < this->num_filters; j++)
			this->output[i][j] = Convolve(i, j);

	// Activate output values
	for(int k = 0; k < this->input->size(); k++)
		for(int i = 0; i < this->num_filters; i++)
			for (int j = 0; j < this->output.size(); j++)
				this->output[k][i][j] = (float)this->Activate(this->output[k][i][j]);

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
					if(j == 479 && k == 639)
						std::cout << "J: " << j << "K: " << k << "i: " << i << std::endl;
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
				tval = this->input[0][batch][channel][(width + j) + ((height + i) * (this->width))];/** this->kernels[channel][filter][k++];*/
			weighted_sum += tval;
		}
		
	}

	return weighted_sum;
}

/* Return the Output Image
*/
vector<vector<vector<float>>> ConvolutionFilter::Output()
{
	return this->output;
}