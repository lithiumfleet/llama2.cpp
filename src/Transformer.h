#include <string>
#include <vector>
#include "Config.h"

#ifndef TRANSFORMER_H
#define TRANSFORMER_H

class TransformerWeights {
public:
    // token embedding table
    std::vector<float> token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    std::vector<float> rms_att_weight; // (layer, dim) rmsnorm weights
    std::vector<float> rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    std::vector<float> wq; // (layer, dim, n_heads * head_size)
    std::vector<float> wk; // (layer, dim, n_kv_heads * head_size)
    std::vector<float> wv; // (layer, dim, n_kv_heads * head_size)
    std::vector<float> wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    std::vector<float> w1; // (layer, hidden_dim, dim)
    std::vector<float> w2; // (layer, dim, hidden_dim)
    std::vector<float> w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    std::vector<float> rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    std::vector<float> wcls;
};

class  RunState { 
public:
    // current wave of activations
    std::vector<float> x; // activation at current time stamp (dim,)
    std::vector<float> xb; // same, but inside a residual branch (dim,)
    std::vector<float> xb2; // an additional buffer just for convenience (dim,)
    std::vector<float> hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    std::vector<float> hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    std::vector<float> q; // query (dim,)
    std::vector<float> k; // key (dim,)
    std::vector<float> v; // value (dim,)
    std::vector<float> att; // buffer for scores/attention values (n_heads, seq_len)
    std::vector<float> logits; // output logits
    // kv cache
    std::vector<float> key_cache;   // (layer, seq_len, dim)
    std::vector<float> value_cache; // (layer, seq_len, dim)
};

class Transformer{
public:
    /* data */
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes

    /* methods */
    void load_from_path(std::string model_path, Config config);
    void map_weights();
    std::vector<float> forward(std::vector<int> input_ids); // input tokens, return logits vector
};

#endif