#include <cmath>
#include <vector>
#include <assert.h>

#ifndef OPERATIONS_H
#define OPERATIONS_H

using namespace std;

void rmsnorm(vector<float>& o, vector<float>& x, vector<float>& weight);
void softmax(vector<float>& x);
void matmul(vector<float>& xout, vector<float>& x, const vector<vector<float>>& w);

#endif