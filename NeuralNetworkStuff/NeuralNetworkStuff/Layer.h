/* Neuron class
	-- Neurons have weights attached with inputs
	-- The Neuron has an activation function attached as well
*/


#ifndef LAYER_H
#define LAYER_H
#include "libraries.h"
using namespace std;


class Layer
{
public:
	// Class properties
	// layer has 2 dimensional view of values
	// - allows for matrix computation
	// - Row represents the selected neuron
	// - Column is neuron index
	// - matrix[ij] i = neuron j = index within
	vector<vector<float>> weights;		// create a vector of weights
	vector<vector<float>> DZW;			// D(Z/W)	partial derivativ
	vector<float> DLZ;					// D(L/Z)
	vector<float> DCL;							// D(C/L)
	vector<float> inputs;
	vector<float> Z;					// sum(weight*input) + bias
	vector<float> outputs;				// outputs of the layer
	int num_inputs;
	int num_neurons;					// number of neurons in the layer
	int num_outputs;
	float learning_rate;
	float error;
	// Class methods (Public)
	// constructor
	Layer  (int num_neurons,
		  vector<float> inputs, 
          float (*Activation)(float x),
          float learningrate);

	// destructor
	~Layer();

	void InitializeWeights(int,int);	
	void FeedForward ();		
	void BackPropagation(vector<float> target);


private:
	// Properties
	float (*Activate)(float x);						//update outputs by activating Z (O)
	//void WeightSum();						// Calculate the weighted sum (Z)

	// activation 
	void ComputeZ();
	void ActivateZ();

	// backpropagation
	void OutputError(vector<float> target);
	void ActivationDerivative();
};


#endif
