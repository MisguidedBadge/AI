#ifndef POOL_H
#define POOL_H

#include "libraries.h"

// Various pooling functions used in the program
//
/* Max Pool: Take in an image and take a max out of a subset
	- Input: image, pool size, image height, image width
	- Output: modified 
*/
vector<float> MaxPool(vector<float> image, int size, int height, int width);



#endif


