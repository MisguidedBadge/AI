
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
							if (image[batch][channel][l + k * height] > temp)
								temp = image[batch][channel][l + k * height];
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

*/