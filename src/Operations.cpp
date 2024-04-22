#include "Operations.h"

void rmsnorm(vector<float>& o, vector<float>& x, vector<float>& weight) {
    // calculate sum of squares
    float ss = 0.0f;
    for (size_t j = 0; j < x.size(); j++) {
        ss += x[j] * x[j];
    }
    ss /= x.size();
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (size_t j = 0; j < o.size(); j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(vector<float>& x) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (size_t i = 1; i < x.size(); i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (size_t i = 0; i < x.size(); i++) {
        x[i] /= sum;
    }
}

void matmul(vector<float>& xout, vector<float>& x, const vector<vector<float>>& w) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    for (size_t i = 0; i < xout.size(); i++) {
        float val = 0.0f;
        for (size_t j = 0; j < x.size(); j++) {
            val += w[i][j] * x[j];
        }
        xout[i] = val;
    }
}