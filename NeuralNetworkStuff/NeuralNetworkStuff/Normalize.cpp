
#include "Normalize.h"



void Normalize(vector<vector<float>>& input, int option)
{
	float min = 10000000;
	float max = -10000;


	for (int i = 0; i < input.size(); i++)
	{
		float min = 10000000;
		float max = -10000;
		for (int j = 0; j < input[i].size(); j++)
		{
			if (max < input[i][j])
				max = input[i][j];
			if (min > input[i][j])
				min = input[i][j];
		}
		for (int j = 0; j < input[i].size(); j++)
				{
			if (max != 0 && min != 0)
			{
				if (option)
					input[i][j] = (2 * (input[i][j] - min) / (max - min)) - 1;
				else
					input[i][j] = (input[i][j] - min) / (max - min);
			}
				}

	}

}


void Normalize(vector<vector<vector<float>>>& input, int option)
{
	float min = 10000000;
	float max = -10000;

	for (int i = 0; i < input.size(); i++)
	{
		float min = 10000000;
		float max = -10000;
		for (int j = 0; j < input[i].size(); j++)
			for(int k = 0; k < input[i][j].size(); k++)
			{
				if (max < input[i][j][k])
					max = input[i][j][k];
				if (min > input[i][j][k])
					min = input[i][j][k];
			}
			for (int j = 0; j < input[i].size(); j++)
				for (int k = 0; k < input[i][j].size(); k++)
				{
					if (max != 0 && min != 0)
					{
						if (option)
							input[i][j][k] = (2 * (input[i][j][k] - min) / (max - min)) - 1;
						else
							input[i][j][k] = (input[i][j][k] - min) / (max - min);
					}
				}
	}

}



