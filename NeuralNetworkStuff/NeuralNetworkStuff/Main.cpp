/* Program Main File
	- Use the referenced libraries in libraries.h
	- output the results
*/

#include "libraries.h"



int main()
{

	// Network Variables
	int batch = 2;
	float alpha = 0.0001;
	int height = 0;		// Image Height
	int width = 0;		// Image width
	int nk1 = 0;		// Number of Kernels each cnn layer
	int ks1 = 0;		// Kernel Sizes of each layer
	int num_channels;
	int num_filters;
	int stride_x;
	int stride_y;
	int unroll_size = 768 * 2;
	float error = 0;
	num_channels = 3;
	height = 480;
	width = 640;
	nk1 = 1;
	ks1 = 4;
	num_filters = 2;
	stride_x = 1;
	stride_y = 1;
	// Output Vectors
	vector<float> targets = { 1, 0 };
	// Fully Connected Layers ////////////////////////////////////
	// Layer Definition
	Layer* hidden2 = new Layer(200, 3, unroll_size , Relu, alpha);
	Layer* hidden1 = new Layer(3, 1, 200, Relu, alpha/10.0);
	Layer* output_layer = new Layer(1, 1, 3, Relu, alpha/100.00);

	// Weight init
	vector<vector<float>> weights, weights2;
	output_layer->InitializeWeights(3, 1);
	hidden1->InitializeWeights(200, 3);
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
	for (int i = 0; i < 2; i++)
	{
		imageName = "I" + std::to_string(i) + ".jpg";
		matimage = cv::imread(imageName, cv::IMREAD_COLOR);
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

	//int k = 0;
	//for (int i = 0; i < image[0][0].size(); i++)
	//{
	//	testfile << image[0][0][i] << ",";
	//	if (k == width - 1)
	//	{
	//		testfile << endl;
	//		k = 0;
	//	}
	//	else
	//	{
	//		k++;
	//	}

	//}
	for(int i = 0; i < image.size(); i++)
		for(int k = 0; k < image[i].size(); k++)
			image[i][k] = MaxPool(image[i][k], 20, height, width);

	vector<vector<float>> temp_con;
	temp_con.resize(batch);
	for (int i = 0; i < image.size(); i++)
		for (int k = 0; k < image[i].size(); k++)
			temp_con[i].insert(temp_con[i].end(), image[i][k].begin(), image[i][k].end());

	vector<vector<float>> temp_out;
	temp_out.resize(batch);

	for (int i = 0; i < 30; i++) {
		error = 0;
		// Feed Forward
		temp_out = temp_con;
		for (int k = 0; k < image.size(); k++)
		{
			hidden2->LoadInput(temp_out[k]);
			hidden2->FeedForward();
			temp_out[k] = hidden2->outputs;
		}
		for (int k = 0; k < image.size(); k++)
		{
			hidden1->LoadInput(temp_out[k]);
			hidden1->FeedForward();
			temp_out[k] = hidden1->outputs;
		}
		for (int k = 0; k < image.size(); k++)
		{
			output_layer->LoadInput(temp_out[k]);
			output_layer->FeedForward();
			temp_out[k] = output_layer->outputs;
		}
		// Determine error
		for (int k = 0; k < targets.size(); k++)
			error += temp_out[k][0] - targets[k];
		error = error / targets.size();
		// Back Propagation
		output_layer->BackPropagation(error);
		hidden1->BackPropagation(output_layer->weights, output_layer->DCZ);
		hidden2->BackPropagation(hidden1->weights, hidden1->DCZ);
		// Update Layer Weights
		hidden2->UpdateWeights();
		hidden1->UpdateWeights();
		output_layer->UpdateWeights();
		// Print Error
		//cout << "layer error: " << output_layer->error << endl;
		//testfile << output_layer->error << ',' << output_layer->weights[0][0] << "," << output_layer->weights[0][1] << "," << output_layer->weights[1][0] << "," << output_layer->weights[1][1] << std::endl;
		//printf("Test \n");
	}
	//cout << "layer error: " << output_layer->error << endl;
	testfile.close();



	return 0;
}
