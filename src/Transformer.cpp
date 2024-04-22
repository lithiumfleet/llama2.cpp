#include "Transformer.h"


void split_matrix_2d(vector<vector<float>>& dest, vector<float>::iterator& beg, int width, int length) {
    for (int i = 0; i < width; i ++) {
        dest.push_back(vector<float>(beg+length*i, beg+length*i+length+1));
    }
    beg += width * length;
}

void split_matrix_3d(vector<vector<vector<float>>>& dest, vector<float>::iterator& beg, int width, int length, int height) {
    vector<vector<float>> temp;
    for (int i = 0; i < width; i ++) {
        split_matrix_2d(temp, beg, length, height);
    }
    dest.push_back(temp);
}

void Transformer::map_weights() {
    int head_size = this->config.dim / this->config.n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    int dim         = this->config.dim; // transformer dimension
    int hidden_dim  = this->config.hidden_dim; // for ffn layers
    int n_layers    = this->config.n_layers; // number of layers
    int n_heads     = this->config.n_heads; // number of query heads
    int n_kv_heads  = this->config.n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size  = this->config.vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len     = this->config.seq_len; // max sequence length

    int offset = 0;
    auto beg = this->data.begin();

    // token_embedding_table
    split_matrix_2d(this->weights.token_embedding_table, beg, vocab_size, dim);
    // rms_att_weight
    split_matrix_2d(this->weights.rms_att_weight, beg, n_layers, dim);
    // wq
    split_matrix_3d(this->weights.wq, beg, n_layers, dim, n_heads * head_size);
    // wk
    split_matrix_3d(this->weights.wk, beg, n_layers, dim, n_kv_heads * head_size);
    // wv
    split_matrix_3d(this->weights.wv, beg, n_layers, dim, n_kv_heads * head_size);
    // wo
    split_matrix_3d(this->weights.wo, beg, n_layers, n_heads * head_size, dim);
    // rms_ffn_weight
    split_matrix_2d(this->weights.rms_ffn_weight ,beg, n_layers, dim);
    // w1
    split_matrix_3d(this->weights.w1, beg, n_layers, hidden_dim, dim);
    // w2
    split_matrix_3d(this->weights.w2, beg, n_layers, hidden_dim, dim);
    // w3
    split_matrix_3d(this->weights.w3, beg, n_layers, hidden_dim, dim);
    // rms_final_weight
    this->weights.rms_final_weight = vector<float>(beg, beg+dim+1);
    // skipped weights
    beg += dim + seq_len * head_size / 2 + seq_len * head_size / 2;
    // wcls
    this->weights.wcls = this->weights.token_embedding_table;
}

void Transformer::load_from_path(string model_path, Config config) {
    printf("[info] loading model.\n");
    this->config = config;
    ifstream fp(model_path, ios::binary);
    if (!fp.is_open()) {
        string error_message = "[error] can not open file " + model_path + " when loading weights.\n";
        printf("%s", error_message.c_str()); fflush(stdout);
        throw runtime_error(error_message);
    }

    // get file size
    fp.seekg(0, ios::end);
    streampos _file_size = fp.tellg();
    // skip config
    size_t config_size = sizeof(Config);
    fp.seekg(config_size/sizeof(char));
    this->file_size = (unsigned long long)(_file_size - config_size);
    
    // read from file
    vector<char> char_buffer(this->file_size);
    fp.read(char_buffer.data(), this->file_size);

    const float* float_buffer = reinterpret_cast<const float*>(char_buffer.data());
    size_t num_floats = this->file_size / sizeof(float);

    this->data = vector<float>(float_buffer, float_buffer+num_floats);

    // load weights
    printf("loading weights.\n");
    map_weights();

    // clear this.data
    this->data.clear();

    printf("[info] finish loading model.\n");
}

vector<float> Transformer::forward(int token, int pos) {
    /*                                                                                                                                       
     * (important) input args:
     *   token: current token id. also refer to the position of the
     *   transformer.k_cache, v_cache
     *
     * Following step:
     *   1. map the token to embedding matrix: in this programe, the author just use "dict" to map this. see https://chat.openai.com/share/855ca067-9c8f-4a1d-be72-19c49bee4ddf
     *   2. rmsnorm
     *   for layer in layers:
     *       3. k = k_cache && v = v_cache
     *       4. qkv *= W_qkv. due to kv_cache, qkv only need compute one token one time
     *       5. add RoPE
     *       for head in heads:
     *           6. init att for this head
     *           for step in time_steps:
     *               7. compute score: att[step] = q*k / sqrt(head_size)
     *           8. softmax att
     *           9. output = att * value. but it can simply use weighted add in this case
     *           10. output *= W_o
     *           11. residual add
     *           12. rmsnorm
     *           13. FFN: w2(F.silu(w1(x)) * w3(x))
     *           14. residual add
     *   15. rmsnorm
     *   16. logits = x * W_cls
     */

    // define convenice varibles
    Config *p = &this->config;
    TransformerWeights *w = &this->weights;
    RunState *s = &this->state;

    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;


}
