#include "Transformer.h"

// for debug usage
string head_vec(string name, vector<float>& vec) {
    string res = name + ": ";
    for (size_t i = 0; i < 5; i ++) {
        res += to_string(vec[i]) + ", ";
    }
    return res;
}

string head_vec(string name, vector<vector<float>>& vec) {
    string res = name + ": ";
    for (size_t i = 0; i < 3; i ++) {
        for (size_t j = 0; j < 3; j ++) {
            res += to_string(vec[i][j]) + ", ";
        }
        res += "| ";
    }
    return res;
}

void print_state(RunState s, int l) {
    cout << "Current layer: " << l << endl;
    cout << head_vec("q          ", s.q             ) << endl;
    cout << head_vec("k          ", s.k             ) << endl;
    cout << head_vec("v          ", s.v             ) << endl;
    cout << head_vec("key_cache  ", s.key_cache[l]  ) << endl;
    cout << head_vec("value_cache", s.value_cache[l]) << endl;
    cout << head_vec("att        ", s.att           ) << endl;
    cout << head_vec("x          ", s.x             ) << endl;
    cout << head_vec("xb         ", s.xb            ) << endl;
    cout << head_vec("xb2        ", s.xb2           ) << endl;
    cout << head_vec("hb         ", s.hb            ) << endl;
    cout << head_vec("hb2        ", s.hb2           ) << endl;
    cout << endl;
}

void split_matrix_2d(vector<vector<float>>& dest, vector<float>::iterator& beg, int width, int length) {
    for (int i = 0; i < width; i ++) {
        dest.push_back(vector<float>(beg+length*i, beg+length*i+length));
    }
    beg += width * length;
}

void split_matrix_3d(vector<vector<vector<float>>>& dest, vector<float>::iterator& beg, int width, int length, int height) {
    for (int i = 0; i < width; i ++) {
        vector<vector<float>> temp;
        split_matrix_2d(temp, beg, length, height);
        dest.push_back(temp);
    }
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

    auto beg = this->data.begin();

    // token_embedding_table
    split_matrix_2d(this->weights.token_embedding_table, beg, vocab_size, dim);
    //assert(this->weights.token_embedding_table.size() == vocab_size && this->weights.token_embedding_table[0].size() == dim);
    // rms_att_weight
    split_matrix_2d(this->weights.rms_att_weight, beg, n_layers, dim);
    //assert(this->weights.rms_att_weight.size() == n_layers && this->weights.rms_att_weight[0].size() == dim);
    // wq
    split_matrix_3d(this->weights.wq, beg, n_layers, dim, n_heads * head_size);
    //assert(this->weights.wq.size() == n_layers && this->weights.wq[0].size() == dim && this->weights.wq[0][0].size() == n_heads*head_size);
    // wk
    split_matrix_3d(this->weights.wk, beg, n_layers, dim, n_kv_heads * head_size);
    //assert(this->weights.wk.size() == n_layers && this->weights.wk[0].size() == dim && this->weights.wk[0][0].size() == n_kv_heads*head_size);
    // wv
    split_matrix_3d(this->weights.wv, beg, n_layers, dim, n_kv_heads * head_size);
    //assert(this->weights.wv.size() == n_layers && this->weights.wv[0].size() == dim && this->weights.wv[0][0].size() == n_kv_heads*head_size);
    // wo
    split_matrix_3d(this->weights.wo, beg, n_layers, n_heads * head_size, dim);
    //assert(this->weights.wo.size() == n_layers && this->weights.wo[0].size() == n_heads*head_size && this->weights.wo[0][0].size() == dim);
    // rms_ffn_weight
    split_matrix_2d(this->weights.rms_ffn_weight ,beg, n_layers, dim);
    //assert(this->weights.rms_ffn_weight.size() == n_layers && this->weights.rms_ffn_weight[0].size() == dim);
    // w1
    split_matrix_3d(this->weights.w1, beg, n_layers, hidden_dim, dim);
    //assert(this->weights.w1.size() == n_layers && this->weights.w1[0].size() == hidden_dim && this->weights.w1[0][0].size() == dim);
    // w2
    split_matrix_3d(this->weights.w2, beg, n_layers, dim, hidden_dim);
    //assert(this->weights.w2.size() == n_layers && this->weights.w2[0].size() == dim && this->weights.w2[0][0].size() == hidden_dim);
    // w3
    split_matrix_3d(this->weights.w3, beg, n_layers, hidden_dim, dim);
    //assert(this->weights.w3.size() == n_layers && this->weights.w3[0].size() == hidden_dim && this->weights.w3[0][0].size() == dim);
    // rms_final_weight
    this->weights.rms_final_weight = vector<float>(beg, beg+dim);
    //assert(this->weights.rms_final_weight.size() == dim);
    // skipped weights
    beg += dim + seq_len * head_size / 2 + seq_len * head_size / 2;
    // wcls
    this->weights.wcls = this->weights.token_embedding_table;
}

void Transformer::init_runtime_status() {

    int dim         = this->config.dim; // transformer dimension
    int hidden_dim  = this->config.hidden_dim; // for ffn layers
    int n_layers    = this->config.n_layers; // number of layers
    int n_heads     = this->config.n_heads; // number of query heads
    int vocab_size  = this->config.vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len     = this->config.seq_len; // max sequence length

    this->state.x           = vector<float>(dim, 0);
    this->state.xb          = vector<float>(dim, 0);
    this->state.xb2         = vector<float>(dim, 0);
    this->state.hb          = vector<float>(hidden_dim, 0);
    this->state.hb2         = vector<float>(hidden_dim, 0);
    this->state.q           = vector<float>(dim, 0);
    this->state.key_cache   = vector<vector<vector<float>>>(n_layers, vector<vector<float>>(seq_len, vector<float>(dim, 0)));
    this->state.value_cache = vector<vector<vector<float>>>(n_layers, vector<vector<float>>(seq_len, vector<float>(dim, 0)));
    this->state.att         = vector<vector<float>>(n_heads, vector<float>(seq_len,0));
    this->state.logits      = vector<float>(vocab_size, 0);
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
    this->file_size = (unsigned long long)_file_size - (unsigned long long)config_size;
    size_t num_floats = this->file_size / sizeof(float);
    
    // read from file
    this->data = vector<float>(num_floats);

    fp.read(reinterpret_cast<char*>(this->data.data()), this->file_size);

    // load weights
    printf("loading weights.\n");
    map_weights();

    // clear this.data
    this->data.clear();

    // init runtiome status
    printf("init runtime status.\n");
    init_runtime_status();

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

    s->x = this->weights.token_embedding_table[token];

    // pass through decode layers
    for (int l = 0; l < p->n_layers; l ++) {
        // rmsnorm
        rmsnorm(s->xb, s->x, w->rms_att_weight[l]);
        // get kvcache
        s->k = vector<float>(dim, 0);
        s->v = vector<float>(dim, 0);

        // qkv matmuls for this position
        matmul(s->q, s->xb, w->wq[l]);
        matmul(s->k, s->xb, w->wk[l]);
        matmul(s->v, s->xb, w->wv[l]);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            // v == 0, vec = s->q
            float v0 = s->q[i];
            float v1 = s->q[i+1];
            s->q[i]   = v0 * fcr - v1 * fci;
            s->q[i+1] = v0 * fci + v1 * fcr;
            // v != 0, vec = s->k
            for (int v = 1; v < rotn; v++) {
                // the vector to rotate (query or key)
                float v0 = s->k[i];
                float v1 = s->k[i+1];
                s->k[i]   = v0 * fcr - v1 * fci;
                s->k[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // update kvcache
        s->key_cache[l][pos] = s->k;
        s->value_cache[l][pos] = s->v;

        // multihead attention. iterate over all heads
        for (int h = 0; h < p->n_heads; h++) {
            // in this structure, one seq/att/q/k/v may looks like this:
            // |<---------------------dim--------------------->|
            // |<--head_size-->|<--head_size-->|<--head_size-->|
            // so we have seq beg and end represents the offset in a seq.
            int seq_offset_beg = h*head_size; 
            
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // the key for this time step: s->keycache[l][t][h*head_size:(h+1)*head_size+1] 
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += s->q[seq_offset_beg+i] * s->key_cache[l][t][seq_offset_beg+i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                s->att[h][t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(s->att[h], pos+1);

            // weighted sum of the values, store back into xb
            fill(s->xb.begin()+seq_offset_beg, s->xb.begin()+seq_offset_beg+head_size, 0);

            for (int t = 0; t <= pos; t++) {
                // value vector for this head at this time step is: s->value_cache[l][t][h*head_size:(h+1)*head_size+1]
                // get the attention weight for this timestep
                float a = s->att[h][t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    s->xb[seq_offset_beg+i] += a * s->value_cache[l][t][seq_offset_beg+i];
                }
            }
        }

        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo[l]);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, s->x, w->rms_ffn_weight[l]);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, w->w1[l]);
        matmul(s->hb2, s->xb, w->w3[l]);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2[l]);

        // residual connection
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb[i];
        }
    }
    // end of layers

    // last rmsnorm
    rmsnorm(s->x, s->x, w->rms_final_weight);

    // classifier into logits
    matmul(s->logits, s->x, w->wcls);
    return s->logits;
}

