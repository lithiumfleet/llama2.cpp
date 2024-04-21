#ifndef TOKENIZER_H
#define TOKENIZRE_H
#include <string>
#include <unordered_map>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <sstream>
#include "config.h"

class Tokenizer {
public:

    /* data */
    std::vector<std::string> vocab;
    std::vector<float> vocab_scores; // this vocab_score list is for saving the weights of each token. Some neighboring tokens can be replaced by a single token, both of which form the same string, but the latter has a higher score. e.g. h el lo -> "hello"
    std::unordered_map<std::string, int> vocab_to_index;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings

    /* methods */
    void load_from_path(std::string tokenizer_file_path, Config config);
    void init_vocab_to_index(void);
    std::string decode_token(int token);
    std::string decode(std::vector<int> tokens);
    int lookup_word(std::string word);
    std::vector<int> encode(std::string text);
};

#endif