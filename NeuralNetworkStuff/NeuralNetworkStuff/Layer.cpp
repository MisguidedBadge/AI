#include "Layer.h"


/* Constructor
- Input: number of neurons, Input, Outputs, Activation Function, Learning Rate
- Setup vectors adn variables
*/
Layer::Layer(int num_neurons,
			  int next_neuron,
			  int input_size,
			  int batch_size,
			  vector<vector<float>>& input,
              float  (*Activation)(float x),
              float learning_rate) : inputs(input) {
  // define the number of inputs 
  this->num_inputs = input_size;
  this->num_neurons = num_neurons; 
  this->batch = batch_size;
  this->next_neuron = next_neuron;
  this->learning_rate = learning_rate;
  // Preallocate vector size
  this->weights.resize(num_neurons, vector<float>(num_inputs, 0));
  this->Z.resize(batch_size);
  for(int i = 0; i < batch_size; i++)
	this->Z[i].resize(num_neurons);

  this->outputs.resize(batch_size);
  for(int i = 0; i < batch_size; i++)
	this->outputs[i].resize(num_neurons);

  this->error = 0.0;
  this->DCZ.resize(batch_size);
  for (int i = 0; i < batch_size; i++)
  {
	  this->DCZ[i].resize(num_neurons);
  }
  this->DCW.resize(num_neurons, vector<float>(num_inputs, 0));
  // set up our activation function for the layer
  this->Activate = Activation;
}

/* Layer Destructor
- Vectors shrink in Heap until gone
*/
Layer::~Layer() {
  this->weights.shrink_to_fit();
  this->DCW.shrink_to_fit();
  //this->inputs.shrink_to_fit();
  this->Z.shrink_to_fit();
  this->outputs.shrink_to_fit();
}

/* Initialize Weights
- Initialize fully connected network weights to random values
*/
void Layer::InitializeWeights(int inputs, int neurons) {
  for (int i = 0; i < neurons; i++) {
    for (int j = 0; j < inputs; j++) {
      this->weights[i][j] = ((float)((rand()%10000)/10000.00)); //rand()%6 + 1;
    }
  }
}

/* Load Input
- Reloads input each time before feed-forward
*/
void Layer::LoadInput(vector<vector<float>> &inputs)
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
	float max = 0;
	float min = 0;
  // zero Z vector
for(int b = 0; b < this->inputs.size(); b++)
  for (int i = 0; i < this->num_neurons; i++)
    this->Z[b][i] = 0.0;

  // cycling through the neurons and the inputs
  // computing Z
for( int b = 0; b < this->inputs.size(); b++){
	for (int i = 0; i < this->weights.size(); i++) {
		for (int j = 0; j < this->weights[i].size(); j++) {
			this->Z[b][i] += this->weights[i][j] * (inputs)[b][j];
		}
	}
  }
 // Try normalizing data for each Z
for (int b = 0; b < this->inputs.size(); b++)
{
	max = 0;
	min = 1000000;
	for (int i = 0; i < this->Z[b].size(); i++)
	{
		if (max < this->Z[b][i])
			max = this->Z[b][i];
		if (min > this->Z[b][i])
			min = this->Z[b][i];
	}
	for (int i = 0; i < this->Z[b].size(); i++)
		this->Z[b][i] = (this->Z[b][i] - min) / (max - min);

}
}


/* Activation Function
	Input: Nothing
	Output: Nothing
	Process: Pass Z Value into activation function to give output
*/
void Layer::ActivateZ() {
  // run our z array into the activation function 
	for (int b = 0; b < this->inputs.size(); b++) {
		for (int i = 0; i < this->num_neurons; i++) {
			this->outputs[b][i] = this->Activate(this->Z[b][i]);
		}
	}
}

/* Backpropagation function for the output neurons
	Input: Target values
	Output: None
	Calculate Error and Backpropagate it to change weights
*/
void Layer::BackPropagation(vector<vector<float>> error) {
	float avg_err = 0;		// average the error between the batches
	this->LayerError(error);

	// determining the weight error
	for (int i = 0; i < this->weights.size(); i++)  {
		for (int j = 0; j < this->weights[i].size(); j++) {
			avg_err = 0;
			for (int b = 0; b < batch; b++)
				avg_err += (this->DCZ[b][i]) * inputs[b][j];

			avg_err = (avg_err / batch);
			this->DCW[i][j] = avg_err;
	}
  }
}

/* Backpropagation function for the Hidden neurons
	Input: Hiddeen Layer values
	Output: None
	Calculate Error and Backpropagate it to change weights
*/
void Layer::BackPropagation(vector<vector<float>> &weights, vector<vector<float>> &neuron_error) {
	float avg_err = 0;	// average error (batched error)
	this->LayerError(weights, neuron_error);
	// determining the weight error
	//concurrency::parallel_for(0, (int)this->weights.size(), [&](int i) {
	for (int i = 0; i < this->weights.size(); i++) {
		for (int j = 0; j < this->weights[i].size(); j++) {
			avg_err = 0;
			// want to average all batch weighted errors
			for (int b = 0; b < this->batch; b++)
				avg_err += this->DCZ[b][i] * (inputs)[b][j];
			avg_err = avg_err / this->outputs.size();
			// get the weight average error
			this->DCW[i][j] = avg_err;
		}
	}
		//});
}

/* Update Weights
- Update Layer Weights
*/
void Layer::UpdateWeights(){
	int i = 0;
		concurrency::parallel_for(0, (int)this->weights.size(), [&](int i) {
		for (int j = 0; j < this->weights[i].size(); j++) {
			this->weights[i][j] -= this->learning_rate * this->DCW[i][j];
		}
	});
}

/* Layer Error for output
- OutputError = Output - Target
- Overall output will have N neurons for N outputs
*/
void Layer::LayerError(vector<vector<float>> &error) {
	// each batch
	for (int b = 0; b < this->batch; b++) 
	{
		for (int i = 0; i < this->outputs[0].size(); i++)
		{
				// Compute layer error
				this->DCZ[b][i] = error[b][i] * DRelu(this->Z[b][i]);
		}
	}
}

/* LayerError
- Hidden Layer Error
- Error is determined by (Error = D(C/Z)) (Error * Weight)
*/
void Layer::LayerError(vector<vector<float>> &weights, vector<vector<float>> &neuron_error) {
	this->error = 0.0;
	float sum = 0.0;
	// go through each batch
	for (int b = 0; b < this->outputs.size(); b++)
	{
		// Go through each neuron
		for (int i = 0; i < this->num_neurons; i++) {
			// Sum weighted Error for that neuron
			for (int j = 0; j < this->next_neuron; j++)
			{
				// Take the transpose of the Ahead layer's weights
				// Compute Error by multiplying Layer weight corresponding to the Output error
				sum += weights[j][i] * neuron_error[b][j];
			}
			this->DCZ[b][i] = sum * DRelu(this->Z[b][i]);
			sum = 0;
		}
	}
}
