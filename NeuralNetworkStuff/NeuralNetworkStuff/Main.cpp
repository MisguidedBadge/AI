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
	testfile.open("test_2Layer.dat");
	float alpha = 0.000001;

	// Input and Output Vectors
	vector<float> inputs = {0.24, .46};
	vector<float> targets = { 0.90, 0.31};

	// Layer Definition
	Layer* hidden2 = new Layer(4, inputs, 18, Relu, alpha);
	Layer* hidden1 = new Layer(18, hidden2->outputs, 2, Relu, alpha);
	Layer* output_layer = new Layer(2, hidden1->outputs, 2, Relu , alpha);
	
	// Weight init
	vector<vector<float>> weights, weights2;
	output_layer->InitializeWeights(18, 2);
	hidden1->InitializeWeights(4, 18);
	hidden2->InitializeWeights(2, 4);
	weights = output_layer->weights;
	
	
	for (int i = 0; i < 300 ; i++) {
		// Feed Forward
		hidden2->FeedForward(inputs);
		hidden1->FeedForward(hidden2->outputs);
		output_layer->FeedForward(hidden1->outputs);
		// Back Propagation
		output_layer->BackPropagation(targets);
		hidden1->BackPropagation(output_layer->weights, output_layer->DCZ);
		hidden2->BackPropagation(hidden1->weights, hidden1->DCZ);
		// Update Layer Weights
		hidden2->UpdateWeights();
		hidden1->UpdateWeights();
		output_layer->UpdateWeights();
		// Print Error

		testfile << output_layer->error << ',' << output_layer->weights[0][0] << "," << output_layer->weights[0][1] << "," << output_layer->weights[1][0] << "," << output_layer->weights[1][1] << std::endl;
		//printf("Test \n");
	}
	cout << "layer error: " << output_layer->error << endl;
	testfile.close();



	return 0;
}