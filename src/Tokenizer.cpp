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

    // type:     float  int  string
    // segments: score  len  vocab
    float cur_vocab_score;
    char *cur_vocab = (char*)malloc(this->max_token_length+5);
    int cur_len;
    for (int i = 0; i < this->vocab_size; i++) {

        fp.read(reinterpret_cast<char*>(&cur_vocab_score), sizeof(float));
        // FIXME: why load float change to continue numbers???
        this->vocab_scores.push_back(cur_vocab_score);

        fp.read(reinterpret_cast<char*>(&cur_len), sizeof(int));

        fp.read(cur_vocab, cur_len);
        cur_vocab[cur_len] = '\0';
        cur_len ++;
        this->vocab.push_back(std::string(cur_vocab, cur_len));

        // for debug: show few vocabs.
        if (i % (this->vocab_size/100) == 0) {
            if (i == 0) printf("[info] show vocab_table exsamples\n");
            printf("vocab:%s\tvocab_score:%.4f\tvocab_len:%d\n", cur_vocab, cur_vocab_score, cur_len); fflush(stdout);
        }

    }
    
    fp.close();
    printf("[info] finish loading tokenizer.\n"); fflush(stdout);
}

Tokenizer::~Tokenizer() {
    ;
}

std::string Tokenizer::decode_token(int prev_token, int token) {
    return "";
}