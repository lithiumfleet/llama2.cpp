#include "Tokenizer.h"

void Tokenizer::load_from_path(std::string tokenizer_file_path) {
    Config config;
    config.load_from_path(tokenizer_file_path);
    // i should have written the vocab_size into the tokenizer file... sigh (lithium: why?)
    this->vocab_size = config.vocab_size;

    // =================
    // TODO: from here!!
    // =================



    // malloc space to hold the scores and the strings
    this->vocab = (char**)malloc(vocab_size * sizeof(char*));
    this->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    // this->vocab_table will be nothing. initialized lazily
    for (int i = 0; i < 256; i++) {
        this->byte_pieces[i * 2] = (unsigned char)i;
        this->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_file_path.c_str(), "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_file_path.c_str()); exit(EXIT_FAILURE); }
    if (fread(&this->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(this->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        this->vocab[i] = (char *)malloc(len + 1);
        if (fread(this->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        this->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

Tokenizer::~Tokenizer() {
    for (int i = 0; i < this->vocab_size; i++) { free(this->vocab[i]); }
    free(this->vocab);
    free(this->vocab_scores);
}