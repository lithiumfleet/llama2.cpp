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
        std::string key = std::string(cur_vocab, cur_len);
        this->vocab.push_back(key);

        // for debug: show few vocabs.
        if (i == 0) printf("[info] show some to the vocab_table exsamples\n");
        if (i > 2 && i % (this->vocab_size/8) < 2) {
            printf("vocab:%s\tvocab_score:%f\tvocab_len:%d\n", cur_vocab, cur_vocab_score, cur_len);
        }

    }
    
    fp.close();
    printf("[info] finish loading tokenizer.\n\n"); 
    fflush(stdout);
}

std::string Tokenizer::decode_token(int token) {
    std::string piece = this->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
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
    std::string res;
    for (size_t i = 0; i < tokens.size(); i ++) {
        res += this->decode_token(tokens[i]);
    }
    return res;
}

void Tokenizer::init_vocab_to_index() {
    for (int i=0; i < this->vocab_size; i ++) {
        this->vocab_to_index.insert({this->vocab[i], i});
    }
};

int Tokenizer::lookup_word(std::string word) {
    // lazy init
    if (this->vocab_to_index.empty()) {
        this->init_vocab_to_index();
    }

    auto resp = this->vocab_to_index.find(word);
    if (resp == this->vocab_to_index.end()) {
        return -1;
    }
    return resp->second;
}

std::string convert_to_hex_string(char ch) {
    // NOTE: powered by Gemini...
    // Convert the character to its ASCII value
    int ascii_value = static_cast<int>(ch);

    // Create the output stream with string formatting
    std::stringstream ss;
    ss << std::fixed << std::setfill('0') << std::setw(2) << std::uppercase << std::hex << ascii_value;

    // Extract the formatted string from the stream
    std::string hex_string = ss.str();

    // Add the leading 0x
    hex_string = "<0x" + hex_string + ">";

    return hex_string;
}


std::vector<int> Tokenizer::encode(std::string input) {
    // trans string to vector string
    std::vector<std::string> text;
    std::string string_value;
    for (auto c : input) {
        if (c == 0x00) continue; // patch for decode many '\000'
        if (c <= 0xFF) {
            string_value = convert_to_hex_string(c);
        } else{
            string_value = std::string(1, c);
        }
        text.push_back(string_value);
    }

    while (true) {
        float best_score = -1e10;
        int mergedstr_left_index = -1;

        // BPE: pair wise merge
        for (size_t i=1; i < text.size(); i ++) {
            std::string cur_merge = text[i-1] + text[i];
            int merge_id = this->lookup_word(cur_merge);
            if (merge_id != -1 && this->vocab_scores[merge_id] > best_score) {
                best_score = this->vocab_scores[merge_id];
                mergedstr_left_index = i-1;
            }
        }

        // BPE ending condition
        if (mergedstr_left_index == -1) { break; }

        // text vector
        text[mergedstr_left_index] = text[mergedstr_left_index] + text[mergedstr_left_index+1];
        text.erase(text.begin()+mergedstr_left_index+1);
    }

    // for debug usage
    // for (auto piece : text) {
    //     printf("%s| ", piece.c_str());
    // }
    // printf("\n"); fflush(stdout);

    // lookup index
    std::vector<int> res;
    int index;
    for (size_t i=0; i < text.size(); i ++) {
        index = this->lookup_word(text[i]);
        if (index == -1) {
            std::string error_msg = "[error] can not decode " + text[i] + "\n";
            throw std::runtime_error(error_msg);
        }
        res.push_back(index);
    }

    return res;
}