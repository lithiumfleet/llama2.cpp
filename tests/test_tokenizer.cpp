#include "src/Config.h"
#include "src/Tokenizer.h"
#include <iostream>

using namespace std;

int main() {
    Config config;
    config.load_from_path("./stories110M.bin");
    Tokenizer tokenizer;
    tokenizer.load_from_path("./tokenizer.bin", config);


    vector<string> test_s = {
        "\t",
        "\b",
        "\n",
        "what can i say?",
        "mamba\t out!!?",
        "adsfas\ndf",
        " ",
        "",
        "Once upon a time there are...",
        "<0xAA>",
        "234",
        "-",
        "ADSL",
        "*"
    };

    for (string s : test_s) {
        auto res_1 = tokenizer.encode(s);
        string res_1_str;
        for (auto i : res_1) {
            res_1_str += std::to_string(i)+",";
        }

        string ds = tokenizer.decode(res_1);

        auto res_2 = tokenizer.encode(ds);
        string res_2_str;
        for (auto i : res_2) {
            res_2_str += std::to_string(i)+",";
        }

        cout << "original: " << s << endl;
        cout << "  encode: " << res_1_str << endl;
        cout << "  decode: " << ds << endl;
        cout << "en-again: " << res_2_str << endl;
        cout << endl;
    }
    
    
    return 0;
}