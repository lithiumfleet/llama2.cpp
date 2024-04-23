#include <vector>
#include <string>
#include "Operations.h"
#include "Config.h"

#ifndef SAMPLER_H
#define SAMPLER_H

using namespace std;

class ProbIndex{
public:
    float prob;
    int index;
}; // struct used when sorting probabilities during top-p sampling

class Sampler{
public:
    /* data */
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;

    /* methods */
    void load_from_path(string generate_config_path, Config config);
    int sample(vector<float>& logits);
};

#endif