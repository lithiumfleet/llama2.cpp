#include "Transformer.h"

void Transformer::map_weights() {
    int head_size = this->config.dim / this->config.n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = this->config.n_layers;

    int offset = 0;

    offset = this->config.vocab_size * this->config.dim;
    this->weights.token_embedding_table.reserve(offset);
    std::copy(this->data, this->data + offset, this->weights.token_embedding_table.data());
    this->data += offset;

    offset = n_layers * this->config.dim;
    this->weights.rms_att_weight.reserve(offset);
    std::copy(this->data, this->data + offset, this->weights.rms_att_weight.data());
    this->data += offset;

    offset = n_layers * this->config.dim * (this->config.n_heads * head_size);
    this->weights.wq.reserve(offset);
    std::copy(this->data, this->data + offset, this->weights.wq.data());
    this->data += offset;

    offset = n_layers * this->config.dim * (this->config.n_kv_heads * head_size);
    this->weights.wk.reserve(offset);
    std::copy(this->data, this->data + offset, this->weights.wk.data());
    this->data += offset;

    offset = n_layers * this->config.dim * (this->config.n_kv_heads * head_size);
    this->weights.wv.reserve(offset);
    std::copy(this->data, this->data + offset, this->weights.wv.data());
    this->data += offset;

    offset = n_layers * (this->config.n_heads * head_size) * this->config.dim;
    this->weights.wo.reserve(offset);
    std::copy(this->data, this->data + offset, this->weights.wo.data());
    this->data += offset;

    offset = n_layers * this->config.dim;
    this->weights.rms_ffn_weight.reserve(offset);
    std::copy(this->data, this->data + offset, this->weights.rms_ffn_weight.data());
    this->data += offset;

    offset = n_layers * this->config.dim * this->config.hidden_dim;
    this->weights.w1.reserve(offset);
    std::copy(this->data, this->data + offset, this->weights.w1.data());
    this->data += offset;

    offset = n_layers * this->config.hidden_dim * this->config.dim;
    this->weights.w2.reserve(offset);
    std::copy(this->data, this->data + offset, this->weights.w2.data());
    this->data += offset;

    offset = n_layers * this->config.dim * this->config.hidden_dim;
    this->weights.w3.reserve(offset);
    std::copy(this->data, this->data + offset, this->weights.w3.data());
    this->data += offset;

    offset = this->config.dim;
    this->weights.rms_final_weight.reserve(offset);
    std::copy(this->data, this->data + offset, this->weights.rms_final_weight.data());
    this->data += offset;
}

void Transformer::load_from_path(std::string model_path, Config config) {
    printf("[info] loading model.\n");
    this->config = config;
    std::ifstream fp(model_path, std::ios::binary);
    if (!fp.is_open()) {
        std::string error_message = "[error] can not open file " + model_path + " when loading weights.\n";
        std::printf("%s", error_message.c_str()); fflush(stdout);
        throw std::runtime_error(error_message);
    }

    // get file size
    fp.seekg(0, std::ios::end);
    std::streampos _file_size = fp.tellg();
    // skip config
    size_t config_size = sizeof(Config);
    fp.seekg(config_size/sizeof(char));
    this->file_size = (ssize_t)(_file_size - config_size);
    
    // new buffer
    this->data = (float*)malloc(this->file_size);
    fp.read(reinterpret_cast<char*>(this->data), this->file_size);

    // load weights
    map_weights();

    printf("[info] finish loading model.\n");
}


std::vector<float> Transformer::forward(std::vector<int> input_ids) {
    return std::vector<float>();
}
