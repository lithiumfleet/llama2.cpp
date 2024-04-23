#include "Tinystory.h"

void Tinystory::load_from_path(string tokenizer_path, string model_path, string generate_config_path) {
    this->config.load_from_path(model_path);
    this->model.load_from_path(model_path, this->config);
    this->tokenizer.load_from_path(tokenizer_path, this->config);
    this->sampler.load_from_path("", this->config);
}

void Tinystory::generate(string prompt, int steps) {
    // FIXME: add <BOS>

    // encode the (string) prompt into tokens sequence
    vector<int> input_ids = this->tokenizer.encode(prompt);
    input_ids.insert(input_ids.begin(), 1);

    int next;        // will store the next token in the sequence
    int token = input_ids[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        vector<float> logits = this->model.forward(token, pos);
        next = this->sampler.sample(logits);
        pos++;
        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        string piece = this->tokenizer.decode_token(token);
        cout << piece;

        fflush(stdout);
        token = next;
    }
}
