#include "Layer.h"


/* Constructor
- Input: number of neurons, Input, Outputs, Activation Function, Learning Rate
- Setup vectors adn variables
*/
Layer::Layer(int num_neurons,
			  vector<float> inputs,
			  int outputs,
              float  (*Activation)(float x),
              float learning_rate) {
  // define the number of inputs 
  this->inputs = inputs;
  this->num_inputs = inputs.size();
  this->num_neurons = num_neurons;
  this->num_outputs = outputs;
  this->learning_rate = learning_rate;
  // Preallocate vector size
  this->weights.resize(num_neurons, vector<float>(num_inputs, 0));
  this->Z.resize(num_neurons);
  this->outputs.resize(num_neurons);
  this->error = 0.0;
  this->DCZ.resize(num_neurons);
  this->DCW.resize(num_neurons, vector<float>(num_inputs, 0));
  this->DLZ.resize(num_neurons);
  // set up our activation function for the layer
  this->Activate = Activation;
}

/* Layer Destructor
- Vectors shrink in Heap until gone
*/
Layer::~Layer() {
  this->weights.shrink_to_fit();
  this->DCW.shrink_to_fit();
  this->DLZ.shrink_to_fit();
  this->inputs.shrink_to_fit();
  this->Z.shrink_to_fit();
  this->outputs.shrink_to_fit();
}

/* Initialize Weights
- Initialize fully connected network weights to random values
*/
void Layer::InitializeWeights(int inputs, int neurons) {
  for (int i = 0; i < neurons; i++) {
    for (int j = 0; j < inputs; j++) {
      this->weights[i][j] = ((float)((rand()%100)))/100.00; //rand()%6 + 1;
    }
  }
}

/* Load Input
- Reloads input each time before feed-forward
*/
void Layer::LoadInput(vector<float> inputs)
{
	this->inputs = inputs;
}

/* Feed Forward 
-	instantiate layer workflow from input to output
*/
void Layer::FeedForward () {
  this->ComputeZ();
  this->ActivateZ();
}

/* Compute the Z value
- Z = w*i + w*i + ..... b
*/
void Layer::ComputeZ() {
  // zero Z vector
  for (int i = 0; i < this->num_neurons; i++)
    this->Z[i] = 0.0;

  // cycling through the neurons and the inputs
  // computing Z
  for (int i = 0; i < this->weights.size(); i++) {
    for (int j = 0; j < this->weights[i].size(); j++) {
      this->Z[i] += this->weights[i][j] * this->inputs[j];
    }
  }
}


/* Activation Function
	Input: Nothing
	Output: Nothing
	Process: Pass Z Value into activation function to give output
*/
void Layer::ActivateZ() {
  // run our z array into the activation function 
  for (int i = 0; i < this->num_neurons; i++) {
    this->outputs[i] = this->Activate(this->Z[i]);
  }
}

/* Backpropagation function for the output neurons
	Input: Target values
	Output: None
	Calculate Error and Backpropagate it to change weights
*/
void Layer::BackPropagation(vector<float> target) {
  this->LayerError(target);

  // determining the weight error
  for (int i = 0; i < this->weights.size(); i++)  {
    for (int j = 0; j < this->weights[i].size(); j++) {
      this->DCW[i][j] = this->DCZ[i] * inputs[j];
      // determining what value we need to change the weight
	  //this->weights[i][j] -= this->learning_rate * this->DZW[i][j];
	  //cout << "Weight: " << weights[i][j] << endl;
	}
  }
}

/* Backpropagation function for the Hidden neurons
	Input: Hiddeen Layer values
	Output: None
	Calculate Error and Backpropagate it to change weights
*/
void Layer::BackPropagation(vector<vector<float>> weights, vector<float> neuron_error) {
	this->LayerError(weights, neuron_error);

	// determining the weight error
	for (int i = 0; i < this->weights.size(); i++) {
		for (int j = 0; j < this->weights[i].size(); j++) {
			this->DCW[i][j] = this->DCZ[i] * inputs[j];
			// determining what value we need to change the weight
			//this->weights[i][j] -= this->learning_rate * this->DZW[i][j];
			//cout << "Weight: " << weights[i][j] << endl;
		}
	}
}

/* Update Weights
- Update Layer Weights
*/
void Layer::UpdateWeights(){
	for (int i = 0; i < this->weights.size(); i++) {
		for (int j = 0; j < this->weights[i].size(); j++) {
			this->weights[i][j] -= this->learning_rate * this->DCW[i][j];
		}
	}
}

/* Layer Error for output
- OutputError = Output - Target
- Overall output will have N neurons for N outputs
*/
void Layer::LayerError(vector<float> target) {
  this->error = 0.0;

  for (int i = 0; i < this->num_outputs; i++)
  {
	  // Compute layer error
	  this->DCZ[i] = (this->outputs[i] - target[i]) * DRelu(this->Z[i]);
	  this->error += (this->outputs[i] - target[i]);
  }
}

/* LayerError
- Hidden Layer Error
- Error is determined by (Error = D(C/Z)) (Error * Weight)
*/
void Layer::LayerError(vector<vector<float>> weights, vector<float> neuron_error) {
	this->error = 0.0;
	float sum = 0;
	// Go through each neuron
	for (int i = 0; i < this->num_neurons; i++) {
		// Sum weighted Error for that neuron
		for (int j = 0; j < this->num_outputs; j++)
		{
			// Take the transpose of the Ahead layer's weights
			// Compute Error by multiplying Layer weight corresponding to the Output error
			sum += weights[j][i] * neuron_error[j];
		}
		this->DCZ[i] = sum * DRelu(this->Z[i]);
		sum = 0;
	}
}
