/* Program Main File
	- Use the referenced libraries in libraries.h
	- output the results
*/

#include "libraries.h"



int main()
{

	// Network Variables ////////////////////////////////////////////////
	int batch = 12;
	int pool = 50;
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
	int divider;
	float max;
	float sum;
	float min;
	float total;
	float alpha = 0.0005;
	string test_path_ped = "Testing_Data/PedTest/";
	string test_path_non = "Testing_Data/NonPedTest/";
	string train_path_ped = "Training_Data/Peds/";
	string train_path_non = "Training_Data/NonPeds/";
	vector<vector<float>> error;
	num_channels = 3;
	height = 250;
	width = 120;
	nk1 = 1;
	ks1 = 4;
	num_filters = 2;
	stride_x = 1;
	stride_y = 1;
	unroll_size = 300 * num_filters;
	cv::Mat matimage;
	//vector<cv::Mat> matimage;
	cv::Mat imageChannels[3];
	cv::Size size(width, height);

	// Output Vectors /////////////////////////////////////////////
	vector<float> positive = { 1, 0 };
	vector<float> negative = { 0, 1 };
	vector<vector<float>> targets;
	for (int i = 0; i < batch; i++)
	{
		if (i < batch / 2)
			targets.push_back(negative);
		else
			targets.push_back(positive);
	}

	// Error //////////////////////////////////////////////////////
	error.resize(batch);
	for (int i = 0; i < batch; i++)	// batch size
		error[i].resize(outputs);
	// Unroll Vector //////////////////////////////////////////////
	vector<vector<float>> unroll_vec;
	unroll_vec.resize(batch);
	for (int i = 0; i < batch; i++)
		unroll_vec[i].resize(unroll_size);
	// Fully Connected Layers /////////////////////////////////////
	// Layer Definition ///////////////////////////////////////////
	Layer* hidden2 = new Layer(200, 100, unroll_size, batch, unroll_vec, Relu, alpha);
	Layer* hidden1 = new Layer(100, 3, 200, batch, hidden2->outputs, Relu, alpha);
	Layer* hidden = new Layer(3, outputs, 100, batch,  hidden1->outputs, Relu, alpha);
	Layer* output_layer = new Layer(outputs, outputs, 3, batch, hidden->outputs, Relu, alpha);
	// CNN Layers//////////////////////////////////////////////////
	ConvolutionFilter* cnn = new ConvolutionFilter(batch, num_channels, height, width, num_filters, 3, stride_x, stride_y, Relu, alpha);
	ConvolutionFilter* cnn2 = new ConvolutionFilter(batch, num_filters, height/2, width/2, num_filters, 3, stride_x, stride_y, Relu, alpha);
	ConvolutionFilter* cnn3 = new ConvolutionFilter(batch, num_filters, height/2, width/2, num_filters, 3, stride_x, stride_y, Relu, alpha);
	ConvolutionFilter* cnn4 = new ConvolutionFilter(batch, num_filters, height/2, width/2, num_filters, 3, stride_x, stride_y, Relu, alpha);
	// Weight init ////////////////////////////////////////////////
	vector<vector<float>> weights, weights2;
	output_layer->InitializeWeights(3, outputs);
	hidden->InitializeWeights(100, 3);
	hidden1->InitializeWeights(200, 100);
	hidden2->InitializeWeights(unroll_size, 200);
	cnn->InitializeKernel();
	cnn2->InitializeKernel();
	cnn3->InitializeKernel();
	cnn4->InitializeKernel();
	// Image Vectors //////////////////////////////////////////////
	vector<vector<vector<float>>> input;
	vector<vector<vector<float>>> image;
	input.resize(batch);
	for (int i = 0; i < batch; i++)
	{
		input[i].resize(3);
		for (int j = 0; j < 3; j++)
			input[i][j].resize(height * width);
	}
	/* initialize random seed: */
	srand(time(NULL));
	ofstream testfile;
	testfile.open("weights.dat");
	std::string imageName;

	// Generate Random Shuffle for SGD //
	vector<int> RanPos;						// Random Pool Pedestrian
	vector<int> RanNeg;						// Random Pool Non-Pedestrian
											
//////////////////////*		Preprocessing Stuff		*///////////////////////////////////////
	for (int i = 0; i < 200; i++) {
		// 50/50 split of Pedestrian and Non-Pedestrian
		RanPos = RandomSelection(batch / 2, pool);
		RanNeg = RandomSelection(batch / 2, pool);
		//RanPos = { 1, 2, 3, 4, 5, 6 };
		//RanNeg = { 1, 2, 3, 4, 5, 6 };


		for (int i = 0; i < batch; i++)
		{
			// Negative first then Postive
			if (i < batch / 2)
				imageName = train_path_non + "I" + std::to_string(RanNeg[i]) + ".jpg";
			else
				imageName = train_path_ped + "I" + std::to_string(RanPos[i - (batch / 2)]) + ".jpg";

			matimage = cv::imread(imageName, cv::IMREAD_COLOR);
			cv::resize(matimage, matimage, size);
			cv::split(matimage, imageChannels);
			for (int j = 0; j < 3; j++)
			{
				int k = 0;
				cv::Mat binaryImage(imageChannels[j].size(), imageChannels[j ].type());
				for (int r = 0; r < matimage.rows; r++)
				{
					for (int c = 0; c < matimage.cols; c++)
					{
						float pixel = imageChannels[j].at<uchar>(r, c);

						input[i][j][k++] = pixel;
					}
				}
			}
		}
		/* Normalize image channels to make it easier for weight updates
	
	*/
		Normalize(input, 0);
		////////// START Layer Stuff //////////////////////////////////////////////////////
		//////////////// CNN Operations ///////////////////////////////////////
		
		/// CNN 1 ///
		cnn->LoadImage(&input);
		cnn->FeedForward();
		image = cnn->Output();
		image = MaxPool(image, 2, height, width);			// 30,000 / 4
		divider = 2;
		/// CNN 2 ///
		cnn2->LoadImage(&image);
		cnn2->FeedForward();
		image = cnn2->Output();
		/// CNN 3 ///
		cnn3->LoadImage(&image);
		cnn3->FeedForward();
		image = cnn3->Output();
		/// CNN 4 ///
		cnn4->LoadImage(&image);
		cnn4->FeedForward();
		image = cnn4->Output();
		image = MaxPool(image, 5, height / divider, width / divider);		
		divider = divider * 5;												// (30,000 / 64)

		// 300 * number of filters
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

		hidden->LoadInput(hidden1->outputs);
		hidden->FeedForward();

		output_layer->LoadInput(hidden->outputs);
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
		hidden->BackPropagation(output_layer->weights, output_layer->DCZ);
		hidden1->BackPropagation(hidden->weights, hidden->DCZ);
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
		Normalize(unroll_vec, 1);
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
		image = BackPropMax(image, num_filters, 5, height / 10, width / 10);
		cnn4->Backpropagation(image);
		cnn3->Backpropagation(cnn4->layer_error);
		cnn2->Backpropagation(cnn3->layer_error);
		image = BackPropMax(cnn2->layer_error, num_filters, 2, height / 2, width / 2);
		cnn->Backpropagation(image);
		// Update Layer Weights
		hidden->UpdateWeights();
		hidden2->UpdateWeights();
		hidden1->UpdateWeights();
		output_layer->UpdateWeights();
		cnn->UpdateWeights();
		cnn2->UpdateWeights();
		cnn3->UpdateWeights();
		cnn4->UpdateWeights();
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
		cout << "CNN Weight: " << cnn->kernels[0][0][1] << endl;
		cout << "Hidden Layer 1 weight: " << hidden1->weights[0][1] << endl;
		testfile << output_layer->error << ',' << output_layer->weights[0][0] << "," << output_layer->weights[0][1] << "," << output_layer->weights[1][0] << "," << output_layer->weights[1][1] << std::endl;


			
	}
	for (int j = 0; j < cnn->output.size(); j++)
	{
		for (int i = 0; i < cnn->output[0].size(); i++)
		{
			for (int k = 0; k < cnn->output[0][0].size(); k++)
			{
				testfile << cnn->output[j][i][k];
				testfile << ";";
			}
			testfile << endl;
		}

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
