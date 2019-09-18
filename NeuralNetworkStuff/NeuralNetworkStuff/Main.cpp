/* Program Main File
	- Use the referenced libraries in libraries.h
	- output the results
*/

#include "libraries.h"

#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

int main()
{
	/* initialize random seed: */
	srand(time(NULL));

	vector<float> inputs = {1.5, 7.0};

	Layer* layer = new Layer(1,inputs,Sigmoid,.04);

	for (int i = 0; i < 1000 ; i++) {
		layer->FeedForward();
		layer->BackPropagation(2);
	}

	return 0;
}