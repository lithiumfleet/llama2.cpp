#include <string>
#include <vector>
#include <cmath>
#include "Config.h"
#include "Operations.h"

#ifndef TRANSFORMER_H
#define TRANSFORMER_H

using namespace std;

class TransformerWeights {
public:
    // token embedding table
    vector<vector<float>> token_embedding_table; // (vocab_size, dim)
    // weights for rmsnorms
    vector<vector<float>> rms_att_weight; // (n_layers, dim) rmsnorm weights
    vector<vector<float>> rms_ffn_weight; // (n_layers, dim)
    // weights for matmuls. note dim == n_heads * head_size
    vector<vector<vector<float>>> wq; // (n_layers, dim, n_heads * head_size)
    vector<vector<vector<float>>> wk; // (n_layers, dim, n_kv_heads * head_size)
    vector<vector<vector<float>>> wv; // (n_layers, dim, n_kv_heads * head_size)
    vector<vector<vector<float>>> wo; // (n_layers, n_heads * head_size, dim)
    // weights for ffn
    vector<vector<vector<float>>> w1; // (n_layers, hidden_dim, dim)
    vector<vector<vector<float>>> w2; // (n_layers, dim, hidden_dim)
    vector<vector<vector<float>>> w3; // (n_layers, hidden_dim, dim)
    // final rmsnorm
    vector<float> rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    vector<vector<float>> wcls;
};

class  RunState { 
public:
    // current wave of activations
    vector<float> x; // activation at current time stamp (dim,)
    vector<float> xb; // same, but inside a residual branch (dim,)
    vector<float> xb2; // an additional buffer just for convenience (dim,)
    vector<float> hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    vector<float> hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    vector<float> q; // query (dim,)
    vector<float> k; // key (dim,)
    vector<float> v; // value (dim,)
    vector<vector<float>> att; // buffer for scores/attention values (n_heads, seq_len)
    vector<float> logits; // output logits
    // kv cache
    vector<vector<vector<float>>> key_cache;   // (layer, seq_len, dim)
    vector<vector<vector<float>>> value_cache; // (layer, seq_len, dim)
};

class Transformer{
public:
    /* data */
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    vector<float> data; // memory mapped data
    unsigned long long file_size; // size of the checkpoint file in bytes

    /* methods */
    void load_from_path(string model_path, Config config);
    void map_weights();
    vector<float> forward(int token, int pos); // input token and position, return logits vector
};

#endif