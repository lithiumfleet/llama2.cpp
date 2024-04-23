#include <vector>
#include <string>
#include <iostream>
#include "Config.h"
#include "Transformer.h"
#include "Tokenizer.h"
#include "Sampler.h"

#ifndef TINYSTORY_H
#define TINYSTORY_H

using namespace std;

class Tinystory {
public:
    /* data */
    Tokenizer tokenizer;
    Sampler sampler;
    Transformer model;
    Config config;

    /* methods */
    void load_from_path(string tokenizer_path, string model_path, string generate_config_path);
    void generate(string prompt, int steps);
};

#endif