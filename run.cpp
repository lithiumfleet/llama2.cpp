#include "Config.h"
#include "Tokenizer.h"
#include "Transformer.h"
#include <iostream>

using namespace std;
int main() {
    Config config;
    config.load_from_path("./stories110M.bin");
    Transformer model;
    model.load_from_path("./stories110M.bin", config);
    auto logits = model.forward(1, 0);
    for (auto item : logits) {
        cout << item << ", ";
    }
    cout << endl;
    
    return 0;
}