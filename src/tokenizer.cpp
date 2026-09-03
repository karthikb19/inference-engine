#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include "tokenizer.hpp"

#include "json.hpp"

#include <unicode/unorm2.h>
#include <unicode/ustring.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace inference {
namespace {

using Json = nlohmann::json;

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Unable to open tokenizer file: " + path.string());
    }
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void append_utf8(std::string& output, std::uint32_t codepoint) {
    if (codepoint <= 0x7f) {
        output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ff) {
        output.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else if (codepoint <= 0xffff) {
        output.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else {
        output.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
}

std::pair<std::uint32_t, std::size_t> next_utf8(std::string_view text, std::size_t offset) {
    const auto first = static_cast<unsigned char>(text[offset]);
    if (first < 0x80) return {first, 1};

    std::size_t length = 0;
    std::uint32_t value = 0;
    if ((first & 0xe0) == 0xc0) {
        length = 2;
        value = first & 0x1f;
    } else if ((first & 0xf0) == 0xe0) {
        length = 3;
        value = first & 0x0f;
    } else if ((first & 0xf8) == 0xf0) {
        length = 4;
        value = first & 0x07;
    } else {
        throw std::runtime_error("Tokenizer encountered invalid UTF-8");
    }
    if (offset + length > text.size()) {
        throw std::runtime_error("Tokenizer encountered truncated UTF-8");
    }
    for (std::size_t index = 1; index < length; ++index) {
        const auto byte = static_cast<unsigned char>(text[offset + index]);
        if ((byte & 0xc0) != 0x80) {
            throw std::runtime_error("Tokenizer encountered invalid UTF-8 continuation byte");
        }
        value = (value << 6) | (byte & 0x3f);
    }
    return {value, length};
}

std::string normalize_nfc(std::string_view input) {
    if (input.empty()) return {};

    UErrorCode status = U_ZERO_ERROR;
    const auto* normalizer = unorm2_getNFCInstance(&status);
    if (U_FAILURE(status)) throw std::runtime_error("ICU could not create the NFC normalizer");

    status = U_ZERO_ERROR;
    std::int32_t utf16_length = 0;
    u_strFromUTF8(nullptr, 0, &utf16_length, input.data(), static_cast<std::int32_t>(input.size()),
                  &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) {
        throw std::runtime_error("Tokenizer input is not valid UTF-8");
    }
    status = U_ZERO_ERROR;
    std::vector<UChar> utf16(static_cast<std::size_t>(utf16_length) + 1);
    u_strFromUTF8(utf16.data(), static_cast<std::int32_t>(utf16.size()), &utf16_length,
                  input.data(), static_cast<std::int32_t>(input.size()), &status);
    if (U_FAILURE(status)) throw std::runtime_error("Tokenizer input is not valid UTF-8");

    status = U_ZERO_ERROR;
    auto normalized_length = unorm2_normalize(normalizer, utf16.data(), utf16_length, nullptr, 0,
                                               &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) {
        throw std::runtime_error("ICU NFC normalization failed");
    }
    status = U_ZERO_ERROR;
    std::vector<UChar> normalized(static_cast<std::size_t>(normalized_length) + 1);
    normalized_length = unorm2_normalize(normalizer, utf16.data(), utf16_length,
                                         normalized.data(), normalized.size(), &status);
    if (U_FAILURE(status)) throw std::runtime_error("ICU NFC normalization failed");

    status = U_ZERO_ERROR;
    std::int32_t utf8_length = 0;
    u_strToUTF8(nullptr, 0, &utf8_length, normalized.data(), normalized_length, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) {
        throw std::runtime_error("ICU UTF-8 conversion failed");
    }
    status = U_ZERO_ERROR;
    std::string result(static_cast<std::size_t>(utf8_length), '\0');
    u_strToUTF8(result.data(), utf8_length, &utf8_length, normalized.data(), normalized_length,
                &status);
    if (U_FAILURE(status)) throw std::runtime_error("ICU UTF-8 conversion failed");
    return result;
}

std::string pair_key(std::string_view left, std::string_view right) {
    std::string key;
    key.reserve(left.size() + right.size() + 1);
    key.append(left);
    key.push_back('\0');
    key.append(right);
    return key;
}

class Regex {
public:
    explicit Regex(const std::string& pattern) {
        int error_code = 0;
        PCRE2_SIZE error_offset = 0;
        code_ = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern.c_str()),
                              PCRE2_ZERO_TERMINATED, PCRE2_UTF | PCRE2_UCP,
                              &error_code, &error_offset, nullptr);
        if (code_ == nullptr) {
            std::array<PCRE2_UCHAR, 256> message{};
            pcre2_get_error_message(error_code, message.data(), message.size());
            throw std::runtime_error("Unable to compile tokenizer regex at offset " +
                                     std::to_string(error_offset) + ": " +
                                     reinterpret_cast<const char*>(message.data()));
        }
    }

    ~Regex() { if (code_ != nullptr) pcre2_code_free(code_); }
    Regex(const Regex&) = delete;

    template <typename Callback>
    void for_each_match(std::string_view text, Callback callback) const {
        pcre2_match_data* match_data = pcre2_match_data_create_from_pattern(code_, nullptr);
        if (match_data == nullptr) throw std::runtime_error("Unable to allocate regex match data");
        std::size_t offset = 0;
        try {
            while (offset < text.size()) {
                const int result = pcre2_match(code_, reinterpret_cast<PCRE2_SPTR>(text.data()),
                                               text.size(), offset, 0, match_data, nullptr);
                if (result == PCRE2_ERROR_NOMATCH) {
                    callback(text.substr(offset));
                    break;
                }
                if (result < 0) throw std::runtime_error("Tokenizer regex matching failed");
                const auto* range = pcre2_get_ovector_pointer(match_data);
                if (range[0] > offset) callback(text.substr(offset, range[0] - offset));
                if (range[1] == range[0]) throw std::runtime_error("Tokenizer regex made no progress");
                callback(text.substr(range[0], range[1] - range[0]));
                offset = range[1];
            }
        } catch (...) {
            pcre2_match_data_free(match_data);
            throw;
        }
        pcre2_match_data_free(match_data);
    }

private:
    pcre2_code* code_ = nullptr;
};

}  // namespace

struct Tokenizer::Impl {
    explicit Impl(const std::filesystem::path& path)
        : document(Json::parse(read_file(path))),
          regex(document.at("pre_tokenizer").at("pretokenizers").at(0)
                    .at("pattern").at("Regex").get<std::string>()) {
        const auto& model = document.at("model");
        if (model.at("type") != "BPE" || document.at("decoder").at("type") != "ByteLevel") {
            throw std::runtime_error("This tokenizer implementation requires byte-level BPE");
        }

        std::int32_t maximum_id = 0;
        for (const auto& [token, value] : model.at("vocab").items()) {
            const auto id = value.get<std::int32_t>();
            vocab.emplace(token, id);
            maximum_id = std::max(maximum_id, id);
        }
        for (const auto& added : document.at("added_tokens")) {
            const auto id = added.at("id").get<std::int32_t>();
            const auto content = added.at("content").get<std::string>();
            added_tokens.emplace_back(content, id);
            vocab.emplace(content, id);
            maximum_id = std::max(maximum_id, id);
        }
        id_to_token.resize(static_cast<std::size_t>(maximum_id) + 1);
        special.resize(id_to_token.size(), false);
        for (const auto& [token, id] : vocab) id_to_token.at(id) = token;
        for (const auto& added : document.at("added_tokens")) {
            special.at(added.at("id").get<std::int32_t>()) = added.at("special").get<bool>();
        }
        std::sort(added_tokens.begin(), added_tokens.end(), [](const auto& left, const auto& right) {
            return left.first.size() > right.first.size();
        });

        std::int32_t rank = 0;
        for (const auto& merge : model.at("merges")) {
            if (merge.is_array() && merge.size() == 2) {
                merge_ranks.emplace(pair_key(merge.at(0).get<std::string>(),
                                             merge.at(1).get<std::string>()), rank++);
            } else {
                throw std::runtime_error("Unsupported tokenizer merge representation");
            }
        }

        std::vector<std::uint32_t> byte_codepoints;
        byte_codepoints.reserve(256);
        for (std::uint32_t value = 33; value <= 126; ++value) byte_codepoints.push_back(value);
        for (std::uint32_t value = 161; value <= 172; ++value) byte_codepoints.push_back(value);
        for (std::uint32_t value = 174; value <= 255; ++value) byte_codepoints.push_back(value);
        std::array<bool, 256> directly_mapped{};
        for (const auto value : byte_codepoints) directly_mapped[value] = true;
        std::uint32_t extra = 0;
        for (std::uint32_t byte = 0; byte < 256; ++byte) {
            const auto codepoint = directly_mapped[byte] ? byte : 256 + extra++;
            byte_to_unicode[byte].clear();
            append_utf8(byte_to_unicode[byte], codepoint);
            unicode_to_byte.emplace(codepoint, static_cast<unsigned char>(byte));
        }
    }

    void encode_piece(std::string_view piece, std::vector<std::int32_t>& output) const {
        std::vector<std::string> symbols;
        symbols.reserve(piece.size());
        for (const auto byte : piece) {
            symbols.push_back(byte_to_unicode[static_cast<unsigned char>(byte)]);
        }

        while (symbols.size() > 1) {
            auto best_rank = std::numeric_limits<std::int32_t>::max();
            std::string best_key;
            for (std::size_t index = 0; index + 1 < symbols.size(); ++index) {
                const auto key = pair_key(symbols[index], symbols[index + 1]);
                if (const auto found = merge_ranks.find(key);
                    found != merge_ranks.end() && found->second < best_rank) {
                    best_rank = found->second;
                    best_key = key;
                }
            }
            if (best_key.empty()) break;

            std::vector<std::string> merged;
            merged.reserve(symbols.size());
            for (std::size_t index = 0; index < symbols.size();) {
                if (index + 1 < symbols.size() &&
                    pair_key(symbols[index], symbols[index + 1]) == best_key) {
                    merged.push_back(symbols[index] + symbols[index + 1]);
                    index += 2;
                } else {
                    merged.push_back(std::move(symbols[index++]));
                }
            }
            symbols = std::move(merged);
        }

        for (const auto& symbol : symbols) {
            const auto found = vocab.find(symbol);
            if (found == vocab.end()) {
                throw std::runtime_error("BPE produced a token absent from the vocabulary");
            }
            output.push_back(found->second);
        }
    }

    void encode_ordinary(std::string_view text, std::vector<std::int32_t>& output) const {
        const auto normalized = normalize_nfc(text);
        regex.for_each_match(normalized, [this, &output](std::string_view piece) {
            encode_piece(piece, output);
        });
    }

    Json document;
    Regex regex;
    std::unordered_map<std::string, std::int32_t> vocab;
    std::vector<std::string> id_to_token;
    std::vector<bool> special;
    std::vector<std::pair<std::string, std::int32_t>> added_tokens;
    std::unordered_map<std::string, std::int32_t> merge_ranks;
    std::array<std::string, 256> byte_to_unicode;
    std::unordered_map<std::uint32_t, unsigned char> unicode_to_byte;
};

Tokenizer::Tokenizer(const std::filesystem::path& tokenizer_path)
    : impl_(std::make_unique<Impl>(tokenizer_path)) {}
Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

std::vector<std::int32_t> Tokenizer::encode(std::string_view text) const {
    std::vector<std::int32_t> output;
    std::size_t offset = 0;
    while (offset < text.size()) {
        std::size_t next_offset = std::string_view::npos;
        const std::pair<std::string, std::int32_t>* next_token = nullptr;
        for (const auto& candidate : impl_->added_tokens) {
            const auto found = text.find(candidate.first, offset);
            if (found < next_offset) {
                next_offset = found;
                next_token = &candidate;
            }
        }
        if (next_token == nullptr) {
            impl_->encode_ordinary(text.substr(offset), output);
            break;
        }
        if (next_offset > offset) {
            impl_->encode_ordinary(text.substr(offset, next_offset - offset), output);
        }
        output.push_back(next_token->second);
        offset = next_offset + next_token->first.size();
    }
    return output;
}

std::string Tokenizer::decode(std::span<const std::int32_t> token_ids,
                              bool skip_special_tokens) const {
    std::string encoded;
    for (const auto id : token_ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= impl_->id_to_token.size() ||
            impl_->id_to_token[id].empty()) {
            throw std::out_of_range("Token ID is outside the tokenizer vocabulary: " +
                                    std::to_string(id));
        }
        if (skip_special_tokens && impl_->special[id]) continue;
        encoded += impl_->id_to_token[id];
    }

    std::string decoded;
    for (std::size_t offset = 0; offset < encoded.size();) {
        const auto [codepoint, length] = next_utf8(encoded, offset);
        if (const auto found = impl_->unicode_to_byte.find(codepoint);
            found != impl_->unicode_to_byte.end()) {
            decoded.push_back(static_cast<char>(found->second));
        } else {
            append_utf8(decoded, codepoint);
        }
        offset += length;
    }
    return decoded;
}

std::string format_qwen3_chat_prompt(std::string_view user_prompt, bool enable_thinking) {
    const std::array messages{ChatMessage{.role = "user", .content = std::string(user_prompt)}};
    return format_qwen3_chat_prompt(messages, enable_thinking);
}

std::string format_qwen3_chat_prompt(std::span<const ChatMessage> messages,
                                     bool enable_thinking) {
    if (messages.empty()) throw std::runtime_error("Chat history cannot be empty");

    std::string result;
    for (const auto& message : messages) {
        if (message.role != "system" && message.role != "user" &&
            message.role != "assistant") {
            throw std::runtime_error("Unsupported chat role: " + message.role);
        }
        result += "<|im_start|>";
        result += message.role;
        result += '\n';
        result += message.content;
        result += "<|im_end|>\n";
    }
    result += "<|im_start|>assistant\n";
    if (!enable_thinking) result += "<think>\n\n</think>\n\n";
    return result;
}

}  // namespace inference
