#include "src/Config.h"
#include "src/Tokenizer.h"

int main() {
    Tokenizer tokenizer;
    tokenizer.load_from_path("./stories110M.bin");
    
    return 0;
}