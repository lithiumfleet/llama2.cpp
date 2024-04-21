#include "src/Config.h"
#include "src/Tokenizer.h"
#include <iostream>

using namespace std;

int main() {
    Config config;
    config.load_from_path("./stories110M.bin");
    Tokenizer tokenizer;
    tokenizer.load_from_path("./tokenizer.bin", config);
    vector<int> vec_a;
    for (int i = 289; i <= 500; ++i) {
        vec_a.push_back(i);
    }
    cout << tokenizer.decode(vec_a);
    
    return 0;
}