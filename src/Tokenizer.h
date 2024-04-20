#ifndef TOKENIZER_H
#define TOKENIZRE_H
#include <string>
#include <unordered_map>
#include "config.h"

class Tokenizer {
private:
    /* data */
    char** vocab;
    float* vocab_scores;
    std::unordered_map<std::string,int> vocab_table; // change from TokenIndex *sorted_vocab
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
public:
    void load_from_path(std::string tokenizer_file_path);
    ~Tokenizer();
};

#endif