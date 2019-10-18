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
	/* Class properties
	 - layer has 2 dimensional view of values
	 - allows for matrix computation
	 - Row represents the selected neuron
	 - Column is neuron index
	 - matrix[ij] i = neuron j = index within
	*/
	vector<vector<float>> weights;		// create a vector of weights
	vector<vector<float>> DCW;			// D(C/W)	Weight derivative
	vector<float> DLZ;					// D(L/Z)
	vector<float> DCZ;					// D(C/Z) = D(C/L) * D(L/Z)
	vector<float> inputs;
	vector<float> Z;					// sum(weight*input) + bias
	vector<float> outputs;				// outputs of the layer
	int num_inputs;
	int num_neurons;					// number of neurons in the layer
	int num_outputs;
	float learning_rate;
	float error;
	/* Class Initial Methods (Public) */
	// constructor
	Layer (int num_neurons,
		  vector<float> inputs,
		  int outputs,
          float (*Activation)(float x),
          float learningrate);

	// destructor
	~Layer();

	/* Class Methods (Public)*/
	// Initialize weights to random values
	void InitializeWeights(int,int);	
	// Load inputs into network
	void LoadInput(vector<float> inputs);
	// Feed input data and go through network
	void FeedForward();
	// Backpropagate to Output and hidden layers
	void BackPropagation(vector<float> target);
	void BackPropagation(vector<vector<float>> weights, vector<float> neuron_error);
	// Update weights after error calculation
	void UpdateWeights();

private:
	// Properties
	float (*Activate)(float x);						//update outputs by activating Z (O)

	/* Class Methods (Private)*/
	// activation 
	void ComputeZ();
	void ActivateZ();

	// backpropagation
	void LayerError(vector<float> target);
	void LayerError(vector<vector<float>> weights, vector<float> neuron_error);

};


#endif
