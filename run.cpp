#include "Tinystory.h"
#include <iostream>

using namespace std;
int main() {
    
    Tinystory tinystory;
    tinystory.load_from_path("tokenizer.bin", "stories110M.bin", "");
    string prompt = "Once upon a time";
    cout << "[info] start generate. current prompt: " << prompt << endl;
    tinystory.generate(prompt, 200);
    
    return 0;
}