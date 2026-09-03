#include "tokenizer.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
    try {
        const inference::Tokenizer tokenizer("models/Qwen3-0.6B/tokenizer.json");

        const auto basic = tokenizer.encode("hello world");
        expect(basic == std::vector<std::int32_t>({14990, 1879}),
               "byte-level BPE IDs differ for a known phrase");
        expect(tokenizer.decode(basic) == "hello world", "basic decode did not round-trip");

        // The tokenizer declares NFC normalization, so decomposed e + acute
        // must decode as the composed character.
        const std::string decomposed = "cafe\xcc\x81";
        const auto unicode = tokenizer.encode(decomposed);
        expect(tokenizer.decode(unicode) == "caf\xc3\xa9", "NFC normalization failed");

        const std::string multilingual = "Hello, \xe4\xb8\x96\xe7\x95\x8c! \xf0\x9f\x91\x8b\nnext line";
        expect(tokenizer.decode(tokenizer.encode(multilingual)) == multilingual,
               "multilingual byte-level encoding did not round-trip");

        const auto chat_text = inference::format_qwen3_chat_prompt("Hello");
        const auto chat = tokenizer.encode(chat_text);
        expect(!chat.empty() && chat.front() == 151644,
               "chat prompt does not begin with <|im_start|>");
        expect(std::find(chat.begin(), chat.end(), 151668) != chat.end(),
               "non-thinking chat prompt does not contain </think>");
        expect(tokenizer.decode(chat, false) == chat_text, "chat prompt did not round-trip");
        expect(tokenizer.decode(std::vector<std::int32_t>{151644, 9707, 151645}) == "Hello",
               "special-token skipping failed");

        const std::vector<inference::ChatMessage> history{
            {.role = "user", .content = "Hello"},
            {.role = "assistant", .content = "Hi!"},
            {.role = "user", .content = "How are you?"},
        };
        const std::string expected_history =
            "<|im_start|>user\nHello<|im_end|>\n"
            "<|im_start|>assistant\nHi!<|im_end|>\n"
            "<|im_start|>user\nHow are you?<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n";
        const auto formatted_history = inference::format_qwen3_chat_prompt(history);
        expect(formatted_history == expected_history, "multi-turn chat formatting failed");
        expect(tokenizer.decode(tokenizer.encode(formatted_history), false) == expected_history,
               "multi-turn chat prompt did not round-trip");

        std::cout << "tokenizer test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tokenizer test failed: " << error.what() << '\n';
        return 1;
    }
}
