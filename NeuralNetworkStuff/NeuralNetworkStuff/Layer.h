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
	// layer has 2 dimenional view of values
	// - allows for matrix computation
	// - Row represents the selected neuron
	// - Column is neuron index
	// - matrix[ij] i = neuron j = index within
	vector<vector<float>> weights;			// create a vector of weights
	vector<vector<float>> Z;				// sum(weight*input) + bias
	vector<vector<float>> outputs;			// outputs of the layer
	vector<vector<float>> DZW;				// D(Z/W)	partial derivativ
	vector<vector<float>> DLZ;				// D(L/Z)
	vector<vector<float>> DCL;				// D(C/L)
	int neurons;							// number of neurons in the layer
	int num_inputs;
	int num_outputs;
	///
	// Class methods (Public)
	void Initialize_Weights();				// Initialize weigths with random values
	///
	// Class constructor and destructor
	Layer();
	~Layer();
	//
private:
	// Properties
	void Activate();						//update outputs by activating Z (O)
	void WeightSum();						// Calculate the weighted sum (Z)
	// methods (Private)

};


#endif
