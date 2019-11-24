
#include "Normalize.h"



void Normalize(vector<vector<float>>& input)
{
	float min = 10000000;
	float max = 0;


	for (int i = 0; i < input.size(); i++)
	{
		float min = 10000000;
		float max = 0;
		for (int j = 0; j < input[i].size(); j++)
		{
			if (max < input[i][j])
				max = input[i][j];
			if (min > input[i][j])
				min = input[i][j];
		}
		if (max != 0)
			for (int j = 0; j < input[i].size(); j++)
				input[i][j] = (input[i][j] - min) / (max - min);

	}

}


void Normalize(vector<vector<vector<float>>>& input)
{
	float min = 10000000;
	float max = 0;

	for (int i = 0; i < input.size(); i++)
	{
		float min = 10000000;
		float max = 0;
		for (int j = 0; j < input[i].size(); j++)
			for(int k = 0; k < input[i][j].size(); k++)
			{
				if (max < input[i][j][k])
					max = input[i][j][k];
				if (min > input[i][j][k])
					min = input[i][j][k];
			}
		if (max != 0)
			for (int j = 0; j < input[i].size(); j++)
				for(int k = 0; k < input[i][j].size(); k++)
					input[i][j][k] = (input[i][j][k] - min) / (max - min);

	}

}



