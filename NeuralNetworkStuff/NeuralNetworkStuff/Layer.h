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
	vector<vector<float>> weights;				// create a vector of weights
	// [batch][value]
	vector<vector<float>> DCW;					// D(C/W)	Weight derivative [batch][
	vector<vector<float>> DCZ;					// D(C/Z) = D(C/L) * D(L/Z)
	vector<vector<float>>& inputs;
	vector<vector<float>> Z;					// sum(weight*input) + bias
	vector<vector<float>> outputs;				// outputs of the layer [Batch][Neuron][Output]
	int num_inputs;
	int num_neurons;							// number of neurons in the layer
	int next_neuron;
	float learning_rate;
	float error;
	/* Class Initial Methods (Public) */
	// constructor
	Layer (int num_neurons,
		  int next_neuron,
		  int input_size,
		  int batch_size,
		  vector<vector<float>>& input,
          float (*Activation)(float x),
          float learningrate);

	// destructor
	~Layer();

	/* Class Methods (Public)*/
	// Initialize weights to random values
	void InitializeWeights(int input,int neurons);	
	// Load inputs into network
	void LoadInput(vector<vector<float>> &inputs);
	// Feed input data and go through network
	void FeedForward();
	// Backpropagate to Output and hidden layers
	void BackPropagation(vector<float> error);
	void BackPropagation(vector<vector<float>> &weights, vector<vector<float>> &neuron_error);
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
	void LayerError(vector<float>& error);
	void LayerError(vector<vector<float>> &weights, vector<vector<float>> &neuron_error);

};


#endif
