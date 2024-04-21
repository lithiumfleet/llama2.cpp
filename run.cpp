#include "src/Config.h"
#include "src/Tokenizer.h"

int main() {
    Config config;
    config.load_from_path("./stories110M.bin");
    Tokenizer tokenizer;
    tokenizer.load_from_path("./tokenizer.bin", config);
    for (auto i : tokenizer.vocab_scores) {
        printf("%f\n", i);
        fflush(stdout);  
    }

    
    return 0;
}