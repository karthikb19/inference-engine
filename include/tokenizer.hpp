#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace inference {

struct ChatMessage {
    std::string role;
    std::string content;
};

class Tokenizer {
public:
    explicit Tokenizer(const std::filesystem::path& tokenizer_path);
    ~Tokenizer();

    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;

    [[nodiscard]] std::vector<std::int32_t> encode(std::string_view text) const;
    [[nodiscard]] std::string decode(
        std::span<const std::int32_t> token_ids,
        bool skip_special_tokens = true) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Qwen3's single-turn user chat format with an assistant generation prompt.
[[nodiscard]] std::string format_qwen3_chat_prompt(
    std::string_view user_prompt,
    bool enable_thinking = false);

// Formats complete chat history and appends the assistant generation prompt.
[[nodiscard]] std::string format_qwen3_chat_prompt(
    std::span<const ChatMessage> messages,
    bool enable_thinking = false);

}  // namespace inference
