
#include "Misc.h"

/* Random Selection Function
-- Selection a random combination from a pool
-- Used for stochastic gradient descent minibatch
*/
vector<int> RandomSelection(int elements, int pool_size)
{
	vector<int> Selection;

	vector<int> RanNum;									// Random Image to select from a pool
	for (int i = 0; i < pool_size; i++)
		RanNum.push_back(i);
	random_shuffle(RanNum.begin(), RanNum.end());		// shuffle	
	for (int i = 0; i < elements; i++)
		Selection.push_back(RanNum[i]);

	return Selection;
	
}