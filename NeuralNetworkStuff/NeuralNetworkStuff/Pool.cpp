
#include "Pool.h"


/* Max Pool (Details in Header)
-- Takes in the largest value of the subset
*/
vector<vector<vector<float>>> MaxPool(vector<vector<vector<float>>> &image, int size, int height, int width)
{
	// output image
	vector<vector<vector<float>>> output;
	// temp value to be compared and stored
	float temp = 0;
	output.resize(image.size());
	for (int i = 0; i < image.size(); i++)
		output[i].resize(image[i].size());
	// for each image batch
	for (int batch = 0; batch < image.size(); batch++)
	{
		// for each image channel
		for (int channel = 0; channel < image[batch].size(); channel++)
		{
			// get the bottom right of the subset
			for (int i = (size - 1); i < height; i += size)
			{
				for (int j = (size - 1); j < width; j += size)
				{
					temp = 0;
					// Pooling function of the subset
					for (int k = (i - (size - 1)); k <= (i); k++)
					{
						for (int l = (j - (size - 1)); l <= (j); l++)
						{
							// take the maximum value
							if (image[batch][channel][l + k * width] > temp)
								temp = image[batch][channel][l + k * width];
						}
					}
					// insert value into new vector
					output[batch][channel].insert(output[batch][channel].end(), temp);
				}
			}
		}
	}
	return output;
}

/* Backprop for maxpool
-- Take an error matrix and expand it by a sizexsize subset within the matrix
-- Error matrix consists
*/

vector<vector<float>> BackPropMax(vector<vector<float>> &errormat, int channels, int size, int height, int width)
{
	// output matrix
	int size_tot = size * size;
	vector<vector<float>> output;
	output.resize(channels);	// how many channels there are
	int err_c = 0;

	for (int i = 0; i < channels; i++)	// for each channel
		output[i].resize(errormat[0].size()*size_tot);	// make the channel the size of your error matrix

	for (int i = 0; i < height * size; i = i + size)
		for (int j = 0; j < width * size; j = j + size)
		{
			for (int r = i; r < (i + size); r++)
				for (int c = j; c < (j + size); c++)
					for (int channel = 0; channel < channels; channel++)
						// input value from errormat and increment the value of errormat by 1 (it should go through all values
						output[channel][c + r * (width * size)] = errormat[channel][err_c];
			err_c++;
		}
	return output;
}
