#include "Tokenizer.h"

void Tokenizer::load_from_path(std::string tokenizer_file_path, Config config) {
    // i should have written the vocab_size into the tokenizer file... sigh (lithium: why?)
    printf("[info] loading tokenizer from %s\n", tokenizer_file_path.c_str()); fflush(stdout);
    this->vocab_size = config.vocab_size;
    printf("vocab_size:%d\n", this->vocab_size); fflush(stdout);

    for (int i = 0; i < 256; i++) {
        this->byte_pieces[i * 2] = (unsigned char)i;
        this->byte_pieces[i * 2 + 1] = '\0';
    }

    // read in the file
    std::ifstream fp(tokenizer_file_path, std::ios::binary);

    // throw error
    if (!fp.is_open()) {
        std::string error_message = "[error] can not open file " + tokenizer_file_path + " when loading tokenizer.\n";
        std::printf("%s", error_message.c_str()); fflush(stdout);
        throw std::runtime_error(error_message);
    }

    // load max_token_length
    fp.read(reinterpret_cast<char*>(&this->max_token_length), sizeof(unsigned int));
    printf("max_token_length:%d\n", this->max_token_length); fflush(stdout);

    // type:    float_16   int   string
    // segment: score      len   vocab
    float cur_vocab_score;
    char *cur_vocab = (char*)malloc(this->max_token_length+5);
    int cur_len;
    for (int i = 0; i < this->vocab_size; i++) {

        fp.read(reinterpret_cast<char*>(&cur_vocab_score), sizeof(float));
        // NOTE: why load float change to continue negative numbers???
        // fp16 can be warpped in C float(fp32) and increasing, don't worry about it.
        // and fp16 not always better...https://github.com/karpathy/llama2.c/pull/93#issuecomment-1651166353
        this->vocab_scores.push_back(cur_vocab_score);

        fp.read(reinterpret_cast<char*>(&cur_len), sizeof(int));

        fp.read(cur_vocab, cur_len);
        cur_vocab[cur_len] = '\0';
        cur_len ++;
        this->vocab.push_back(std::string(cur_vocab, cur_len));

        // for debug: show few vocabs.
        if (i == 0) printf("[info] show some to the vocab_table exsamples\n");
        if (i > 10 && i % (this->vocab_size/2) < 10) {
            printf("vocab:%s\tvocab_score:%f\tvocab_len:%d\n", cur_vocab, cur_vocab_score, cur_len);
        }

    }
    
    fp.close();
    printf("[info] finish loading tokenizer.\n\n"); 
    fflush(stdout);
}

std::string Tokenizer::decode_token(int prev_token, int token) {
    std::string piece = this->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece = piece.substr(1);}
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    // Explaination by GPT: https://chat.openai.com/share/c37b2ded-bd9e-4d3f-b275-f8bf3bd66325
    unsigned char byte_val;
    if (sscanf(piece.c_str(), "<0x%02hhX>", &byte_val) == 1) {
        piece = std::string((char*)this->byte_pieces + byte_val * 2, 2);
    }
    return piece;
}

std::string Tokenizer::decode(std::vector<int> tokens) {
    if (tokens.size() == 0) return "";
    if (tokens.size() == 1) return this->decode_token(1, tokens[0]);
    std::string res = "";
    for (size_t i = 1; i < tokens.size(); i ++) {
        res += this->decode_token(tokens[i-1], tokens[i]);
    }
    return res;
}

int Tokenizer::encode_char(std::string word) {

}

std::vector<int> Tokenizer::encode(std::string text) {
    ;
}