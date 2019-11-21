/* Program Main File
	- Use the referenced libraries in libraries.h
	- output the results
*/

#include "libraries.h"



int main()
{

	// Network Variables
	int batch = 10;
	float alpha = 0.0001;
	int height = 0;		// Image Height
	int width = 0;		// Image width
	int nk1 = 0;		// Number of Kernels each cnn layer
	int ks1 = 0;		// Kernel Sizes of each layer
	int num_channels;
	int num_filters;
	int stride_x;
	int stride_y;
	int unroll_size = 600;
	int outputs = 2;
	vector<vector<float>> error;
	num_channels = 3;
	height = 250;
	width = 120;
	nk1 = 1;
	ks1 = 4;
	num_filters = 2;
	stride_x = 1;
	stride_y = 1;
	error.resize(batch);
	// Output Vectors
	vector<vector<float>> targets = { { 1, 0 } , {0, 1}, {1, 0} , {1, 0} , {1, 0} , {1, 0}, {0, 1}, {0, 1}, {0, 1}, {0, 1} };
	for (int i = 0; i < batch; i++)	// batch size
		error[i].resize(outputs);
	// TEST VECTORS
	// vector<vector<float>> Reverse = { { 0, 1, 1, 0}, {1, 1, 1, 1}, {1, 0 , 1, 0} }; // Test Case for BackProp Max
	//Reverse = BackPropMax(Reverse, 3, 10, 2, 2);
	// Fully Connected Layers ////////////////////////////////////
	// Layer Definition
	vector<vector<float>> temp_out;
	temp_out.resize(batch);


	

	Layer* hidden2 = new Layer(200, 100, unroll_size, batch, temp_out, Relu, alpha);
	Layer* hidden1 = new Layer(100, outputs, 200, batch, hidden2->outputs, Relu, alpha);
	Layer* output_layer = new Layer(outputs, outputs, 100, batch, hidden1->outputs, Relu, alpha);

	// Weight init
	vector<vector<float>> weights, weights2;
	output_layer->InitializeWeights(100, outputs);
	hidden1->InitializeWeights(200, 100);
	hidden2->InitializeWeights(unroll_size, 200);
	// CNN Initialize
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


	// CNN RGB ////////////////////////////////////////////
	ConvolutionFilter* cnn = new ConvolutionFilter(batch, num_channels, height, width, num_filters, 3, stride_x, stride_y, Relu, alpha);
	cnn->InitializeKernel();
	cnn->LoadImage(&input);

	cnn->FeedForward();
	image = cnn->Output();

	image = MaxPool(image, 10, height, width);

	// Unroll vectors
	vector<vector<float>> temp_con;
	temp_con.resize(batch);
	for (int i = 0; i < image.size(); i++)
		for (int k = 0; k < image[i].size(); k++)
			temp_con[i].insert(temp_con[i].end(), image[i][k].begin(), image[i][k].end());

	float total;

	for (int i = 0; i < 100; i++) {
		total = 0;
		// Feed Forward
		temp_out = temp_con;
		hidden2->LoadInput(temp_out);
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
			total += error[b][0];
		}
		// Back Propagation
		output_layer->BackPropagation(error);
		hidden1->BackPropagation(output_layer->weights, output_layer->DCZ);
		hidden2->BackPropagation(hidden1->weights, hidden1->DCZ);
		// Update Layer Weights
		hidden2->UpdateWeights();
		hidden1->UpdateWeights();
		output_layer->UpdateWeights();
		// Print Error
		cout << "Output1: " << output_layer->outputs[0][0] << ":::::" << output_layer->outputs[0][1] << endl;
		cout << "Output2: " << output_layer->outputs[1][0] << ":::::" << output_layer->outputs[1][1] << endl;
		cout << "Output1: " << error[0][0] << ":::::" << error[0][1] << endl;
		cout << "Error: " << total << endl;
		//testfile << output_layer->error << ',' << output_layer->weights[0][0] << "," << output_layer->weights[0][1] << "," << output_layer->weights[1][0] << "," << output_layer->weights[1][1] << std::endl;
		//printf("Test \n");
	}
	testfile.close();



	return 0;
}
