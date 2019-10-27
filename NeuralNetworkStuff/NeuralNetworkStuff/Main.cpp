/* Program Main File
	- Use the referenced libraries in libraries.h
	- output the results
*/

#include "libraries.h"

#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <fstream>
#include <iostream>

int main()
{
	/* initialize random seed: */

	// Input and Output Vectors
    vactor<vector<float>> inputs = {{0.24, .46}, {0.35, 0.78}};
    vector<vector<float>> targets = {{ 0.90, 0.31}, {0.89, 0.25}};

    for(int j = 0; j < 2; j++)
    {
        srand(time(NULL));
        ofstream testfile;
        std::string out_file = "test_2Layer" + std::to_string(i) + ".dat";
        
        testfile.open(out_file);
        float alpha = 0.000001;
        
        // Layer Definition
        Layer* hidden2 = new Layer(4, inputs[j], 18, Relu, alpha);
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
            hidden2->FeedForward(inputs[j]);
            hidden1->FeedForward(hidden2->outputs);
            output_layer->FeedForward(hidden1->outputs);
            // Back Propagation
            output_layer->BackPropagation(targets[j]);
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
    }



	return 0;
}
