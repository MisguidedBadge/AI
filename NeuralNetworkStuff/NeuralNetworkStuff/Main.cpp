/* Program Main File
	- Use the referenced libraries in libraries.h
	- output the results
*/

#include "libraries.h"



int main()
{

	// Network Variables
	int batch = 10;
	float max;
	float sum;
	float min;
	float total;
	float alpha = 0.001;
	int height = 0;		// Image Height
	int width = 0;		// Image width
	int nk1 = 0;		// Number of Kernels each cnn layer
	int ks1 = 0;		// Kernel Sizes of each layer
	int num_channels;
	int num_filters;
	int stride_x;
	int stride_y;
	int unroll_size;
	int outputs = 2;
	vector<vector<float>> error;
	num_channels = 3;
	height = 250;
	width = 120;
	nk1 = 1;
	ks1 = 4;
	num_filters = 4;
	stride_x = 1;
	stride_y = 1;
	unroll_size = 300 * num_filters;
	error.resize(batch);
	// Output Vectors
	vector<vector<float>> targets = { { 1, 0 } , {0, 1}, {1, 0} , {1, 0} , {1, 0} , {1, 0}, {0, 1}, {0, 1}, {0, 1}, {0, 1} };
	for (int i = 0; i < batch; i++)	// batch size
		error[i].resize(outputs);
	// Unroll Vector
	vector<vector<float>> unroll_vec;
	unroll_vec.resize(batch);
	for (int i = 0; i < batch; i++)
		unroll_vec[i].resize(unroll_size);
	// Fully Connected Layers ////////////////////////////////////
	// Layer Definition
	Layer* hidden2 = new Layer(200, 100, unroll_size, batch, unroll_vec, Relu, alpha);
	Layer* hidden1 = new Layer(100, outputs, 200, batch, hidden2->outputs, Relu, alpha);
	Layer* output_layer = new Layer(outputs, outputs, 100, batch, hidden1->outputs, Relu, alpha);
	// CNN Layers////////////////////
	ConvolutionFilter* cnn = new ConvolutionFilter(batch, num_channels, height, width, num_filters, 3, stride_x, stride_y, Relu, alpha);
	
	// Weight init
	vector<vector<float>> weights, weights2;
	output_layer->InitializeWeights(100, outputs);
	hidden1->InitializeWeights(200, 100);
	hidden2->InitializeWeights(unroll_size, 200);
	cnn->InitializeKernel();

	// example Image
	vector<vector<vector<float>>> input;
	vector<vector<vector<float>>> image;
	input.resize(batch);
	for(int i = 0; i < batch; i++)
		input[i].resize(3);

	/* initialize random seed: */
	srand(time(NULL));
	ofstream testfile;
	testfile.open("test_2Layer.dat");

	std::string imageName;

	//images[i] = cv::imread(imageName, cv::IMREAD_COLOR);
	cv::Mat matimage;
	//vector<cv::Mat> matimage;
	cv::Mat imageChannels[3];
	cv::Size size(width, height);
	for (int i = 0; i < batch; i++)
	{
		imageName = "I" + std::to_string(i) + ".jpg";
		matimage = cv::imread(imageName, cv::IMREAD_COLOR);
		cv::resize(matimage, matimage, size);
		cv::split(matimage, imageChannels);
		

		for (int j = 0; j < 3; j++)
		{
			cv::Mat binaryImage(imageChannels[j].size(), imageChannels[j ].type());
			for (int r = 0; r < matimage.rows; r++)
			{
				for (int c = 0; c < matimage.cols; c++)
				{
					float pixel = imageChannels[j].at<uchar>(r, c);

					input[i][j].insert(input[i][j].end(), pixel);

				}
			}
		}
	}

	/* Normalize image channels to make it easier for weight updates
	
	*/
	Normalize(input);
////////// START Layer Stuff //////////////////////////////////////////////////////
	for (int i = 0; i < 100; i++) {
	//////////////// CNN Operations ///////////////////////////////////////
		cnn->LoadImage(&input);
		cnn->FeedForward();

		image = cnn->Output();
		image = MaxPool(image, 10, height, width);


		for (int i = 0; i < image.size(); i++)
		{
			int count = 0;
			for (int k = 0; k < image[i].size(); k++)			// for each channel
				for (int j = 0; j < image[i][k].size(); j++)
					unroll_vec[i][count++] = image[i][k][j];		// update unroll loop
		}
	/////////////// Connected Layers /////////////////////////////////////
		total = 0;
		hidden2->LoadInput(unroll_vec);
		hidden2->FeedForward();

		hidden1->LoadInput(hidden2->outputs);
		hidden1->FeedForward();

		output_layer->LoadInput(hidden1->outputs);
		output_layer->FeedForward();

		// Determine error
		for (int b = 0; b < batch; b++)
		{
			for (int i = 0; i < 2; i++)
			{
				error[b][i] = output_layer->outputs[b][i] - targets[b][i];
				
			}
			total += abs(error[b][0]);
		}
		/////////////// Back Propagation //////////////////////////////////////////////
		output_layer->BackPropagation(error);
		hidden1->BackPropagation(output_layer->weights, output_layer->DCZ);
		hidden2->BackPropagation(hidden1->weights, hidden1->DCZ);
		
		/////////////// Pass error into convolutional Domanin ///////////////
		sum = 0.0;
		for (int b = 0; b < unroll_vec.size(); b++)
			for (int i = 0; i < unroll_vec[b].size(); i++)
			{
				for (int j = 0; j < hidden2->weights.size(); j++)
				{
					sum += hidden2->weights[j][i] * hidden2->DCZ[b][j];
				}
				unroll_vec[b][i] = sum;
				sum = 0;
			}
		// Normalize the unrolling vector
		Normalize(unroll_vec);

		// Zero out image
		for (int i = 0; i < image.size(); i++)
			for (int j = 0; j < image[i].size(); j++)
				for (int k = 0; k < image[i][j].size(); k++)
					image[i][j][k] = 0;

		//reconstruct vector with channels
		for (int i = 0; i < image.size(); i++)
		{
			int count = 0;
			for (int j = 0; j < image[i].size(); j++)			// for each channel
				for (int k = 0; k < image[i][j].size(); k++)
					// TODO NEED TO FIX THIS
					image[i][j][k] = unroll_vec[i][count++];
		}

		image = BackPropMax(image, num_filters, 10, height/10, width/10);
		cnn->Backpropagation(image);
		// Update Layer Weights
		hidden2->UpdateWeights();
		hidden1->UpdateWeights();
		output_layer->UpdateWeights();
		cnn->UpdateWeights();
		// Print Error
		cout << "Epoch: " << i << endl;
		cout << "Output: ";
		for (int i = 0; i < output_layer->outputs.size(); i++)
			cout << output_layer->outputs[i][0] << "; ";
		cout << endl;
		cout << "Error: ";
		for (int i = 0; i < error.size(); i++)
			cout << error[i][0] << "; ";
		cout << endl;
		cout << "Error: " << total << endl;
		cout << "CNN Weight: " << cnn->kernels[0][1][1] << endl;
		cout << "Hidden Layer 1 weight: " << hidden1->weights[0][1] << endl;
		//testfile << output_layer->error << ',' << output_layer->weights[0][0] << "," << output_layer->weights[0][1] << "," << output_layer->weights[1][0] << "," << output_layer->weights[1][1] << std::endl;
		//printf("Test \n");
		//if (total == 0)
		//{
		//	cout << "Finished" << endl;
		//	for (int j = 0; j < cnn->output.size(); j++)
		//	{
		//		for (int i = 0; i < cnn->output[0].size(); i++)
		//		{
		//			for (int k = 0; k < cnn->output[0][0].size(); k++)
		//			{
		//				testfile << cnn->output[j][i][k];
		//				testfile << ";";
		//			}
		//			testfile << endl;
		//		}
		//		
		//	}
		//	break;
		//}
	}
	testfile.close();

	return 0;
}




///* Testing backpropagation*/
//// TEST VECTORS
//vector<vector<vector<float>>> ErrorMat;
//ErrorMat.resize(2);
//vector<vector<float>> ErrorChan = { { 0, 1, 1, 0}, {1, 1, 1, 1} }; // Test Case for BackProp Max
//vector<vector<float>> In_Image = { {1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1,}, {1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1,} };
//vector<vector<vector<float>>> Input;
//Input.resize(2);
//Input[0] = In_Image;
//Input[1] = In_Image;
//ErrorMat[0] = ErrorChan;
//ErrorMat[1] = ErrorChan;

//ConvolutionFilter* cnn = new ConvolutionFilter(2, 2, 4, 4, 2, 3, stride_x, stride_y, Relu, alpha);
//
//cnn->InitializeKernel();
//cnn->LoadImage(&Input);
//cnn->FeedForward();

//ErrorMat = BackPropMax(ErrorMat, 2, 2, 2, 2);

//cnn->Backpropagation(ErrorMat);
//cnn->UpdateWeights();
