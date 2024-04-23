#include "Sampler.h"

void Sampler::load_from_path(string generate_config_path, Config config) {
    printf("[info] load sampler\n");
    printf("FIXME: THIS IS A FAKE LOADER!!!!!\n");
    this->temperature = 0.0;
    this->topp = 0.0;
    this->vocab_size = config.vocab_size;
    this->rng_state = 0;
    printf("[info] finish loading sampler\n\n");
}

int greedy_sample(vector<float>& logits) {
    // get max value's index of this vector.
    int res = -1;
    float max_prob = 0.0f;
    for (size_t i = 0; i < logits.size(); i ++) {
        if (max_prob < logits[i]) {
            max_prob = logits[i];
            res = i;
        }
    }
    return res;
}

float random_f32(unsigned long long *state) { 
    // random float32 in [0,1)
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    unsigned int random_u32 = (*state * 0x2545F4914F6CDD1Dull) >> 32;
    return (random_u32 >> 8) / 16777216.0f;
}

int sample_mult(vector<float> probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

// FIXME: NOT COMPELETE
int sample_topp(vector<float> probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    assert(0);
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

int Sampler::sample(vector<float>& logits) {
    int next;
    if (this->temperature == 0.0f) {
        next = greedy_sample(logits);
    } else {
        // apply the temperature to the logits
        for (int i=0; i<this->vocab_size; i++) { 
            logits[i] /= this->temperature;
        }

        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, this->vocab_size);

        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&this->rng_state);
        // we sample from this distribution to get the next token
        if (this->topp <= 0 || this->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, this->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, this->vocab_size, this->topp, this->probindex, coin);
        }
    }
    return next;
}
