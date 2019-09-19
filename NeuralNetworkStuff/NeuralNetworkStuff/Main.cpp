/* Program Main File
	- Use the referenced libraries in libraries.h
	- output the results
*/

#include "libraries.h"

#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <fstream>

int main()
{
	/* initialize random seed: */
	srand(time(NULL));
	ofstream testfile;
	testfile.open("test.dat");

	vector<float> inputs = {0.24, .46};
	vector<float> targets = { 0.90, 0.31};

	Layer* layer = new Layer(2,inputs,Sigmoid,.1);

	layer->InitializeWeights(2, 2);
	for (int i = 0; i < 10000 ; i++) {
		layer->FeedForward();
		layer->BackPropagation(targets);
		//cout << "layer error: " << layer->error << endl;
		testfile << abs(layer->error) << ",";
		//printf("Test \n");
	}
	testfile.close();

	return 0;
}