
#include "Pool.h"


/* Max Pool (Details in Header)
-- Takes in the largest value of the subset
*/
vector<float> MaxPool(vector<float> image, int size, int height, int width)
{
	// output image
	vector<float> output;
	// temp value to be compared and stored
	float temp = 0;

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
					if (image[l + k * height] > temp)
						temp = image[l + k * height];
				}
			}
			// insert value into new vector
			output.insert(output.end(), temp);
		}
	}
	return output;
}