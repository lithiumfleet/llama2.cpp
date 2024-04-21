#include "config.h"

void Config::load_from_path(std::string config_file_path) {

    std::printf("[info] loading config from %s\n", config_file_path.c_str());

    std::ifstream fp(config_file_path, std::ios::binary);
    
    if (!fp.is_open()) {
        std::string error_message = "[error] can not open file " + config_file_path + " when loading config.\n";
        std::printf("%s", error_message.c_str()); fflush(stdout);
        throw std::runtime_error(error_message);
    }

    fp.read(reinterpret_cast<char*>(this), sizeof(Config));
    fp.close();

    // TODO:assertions for stories110M.bin

    std::printf("dim:%d\nhidden_dim:%d\nn_layers:%d\nn_heads:%d\nn_kv_heads:%d\nvocab_size:%d\nseq_len:%d\n",
        this->dim, this->hidden_dim, this->n_layers, this->n_heads, this->n_kv_heads, this->vocab_size, this->seq_len);

    std::printf("[info] finish loading config.\n\n");
}