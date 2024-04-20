#ifndef CONFIG_H
#define CONFIG_H

#include <fstream>
#include <string>
#include <stdexcept>
#include <cassert>

class Config {
public:
    /* data */
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
    void load_from_path(std::string config_file_path);
};

#endif