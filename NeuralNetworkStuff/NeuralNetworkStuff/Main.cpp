/* Program Main File
	- Use the referenced libraries in libraries.h
	- output the results
*/

#include "libraries.h"



int main()
{
	// Network Variables
	float alpha	= 0.001;
	int height	= 0;		// Image Height
	int width	= 0;		// Image width
	int nk1		= 0;		// Number of Kernels each cnn layer
	int ks1		= 0;		// Kernel Sizes of each layer

	/* initialize random seed: */
	srand(time(NULL));
	ofstream testfile;
	testfile.open("test_2Layer.dat");
	

	// CNN Initialize
	// example Image
	vector<vector<float>> input;
	input.resize(2);
	vector<float> R = { 1, 1, 1 , 2, 2, 2, 3, 3, 3 };
	vector<float> G = { 2, 2, 2, 1, 1, 1, 4, 4, 4 };
	input[0] = R;
	input[1] = G;
	//vector<float> input = { 1,1,1,1,1,2,2,2,2,2 };
	height	= 3;
	width	= 3;
	nk1		= 1;
	ks1		= 4;

	// CNN RGB
	ConvolutionFilter* cnn = new ConvolutionFilter(2, height, width, 1, 2, Relu, alpha);
	cnn->InitializeKernel();
	cnn->LoadImage(input);

	// Input and Output Vectors
	//vector<float> inputs = {0.24, .46};
	//vector<float> targets = { 1.90, 0.65};
	// Fully Connected Layers
	//// Layer Definition
	//Layer* hidden2 = new Layer(4, inputs, 2, Relu, alpha);
	//Layer* hidden1 = new Layer(2, hidden2->outputs, 2, Relu, alpha);
	//Layer* output_layer = new Layer(2, hidden1->outputs, 2, Relu , alpha);
	//
	//// Weight init
	//vector<vector<float>> weights, weights2;
	//output_layer->InitializeWeights(2, 2);
	//hidden1->InitializeWeights(4, 2);
	//hidden2->InitializeWeights(2, 4);
	//weights = output_layer->weights;
	//
	//
	//for (int i = 0; i < 30000 ; i++) {
	//	// Feed Forward
	//	hidden2->FeedForward(inputs);
	//	hidden1->FeedForward(hidden2->outputs);
	//	output_layer->FeedForward(hidden1->outputs);
	//	// Back Propagation
	//	output_layer->BackPropagation(targets);
	//	hidden1->BackPropagation(output_layer->weights, output_layer->DCZ);
	//	hidden2->BackPropagation(hidden1->weights, hidden1->DCZ);
	//	// Update Layer Weights
	//	hidden2->UpdateWeights();
	//	hidden1->UpdateWeights();
	//	output_layer->UpdateWeights();
	//	// Print Error
	//	cout << "layer error: " << output_layer->error << endl;
	//	testfile << output_layer->error << ',' << output_layer->weights[0][0] << "," << output_layer->weights[0][1] << "," << output_layer->weights[1][0] << "," << output_layer->weights[1][1] << std::endl;
	//	//printf("Test \n");
	//}
	//cout << "layer error: " << output_layer->error << endl;
	//testfile.close();



	return 0;
}