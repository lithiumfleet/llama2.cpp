#ifndef TOKENIZER_H
#define TOKENIZRE_H
#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>
#include "config.h"

class Tokenizer {
public:
    /* data */
    std::vector<std::string> vocab;
    std::vector<float> vocab_scores; // this vocab_score list is for saving the weights of each token. Some neighboring tokens can be replaced by a single token, both of which form the same string, but the latter has a higher score. e.g. h el lo -> "hello"
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
    void load_from_path(std::string tokenizer_file_path, Config config);
    std::string decode_token(int prev_token, int token);
    std::string decode(std::vector<int> tokens);
    int Tokenizer::encode_char(std::string word);
    std::vector<int> encode(std::string text);
};

#endif