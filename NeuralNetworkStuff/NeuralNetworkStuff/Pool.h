#ifndef POOL_H
#define POOL_H

#include "libraries.h"

// Various pooling functions used in the program
//
/* Max Pool: Take in an image and take a max out of a subset
	- Input: image, pool size, image height, image width
	- Output: Downsampled Image 
*/
vector<vector<vector<float>>> MaxPool(vector<vector<vector<float>>> &image, int size, int height, int width);

/* Max Pool Backpropagation
	- Input: image, pool size, image height, image width
	- Output: Upsampled image matrix of errors
*/
vector<float> BackMax(vector<float> image, int size, int height, int width);

#endif


