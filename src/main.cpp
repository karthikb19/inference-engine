#include "inference_engine.hpp"
#include "tokenizer.hpp"

#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

[[nodiscard]] std::uint64_t parse_token_id(std::string_view text) {
    if (text.empty() || text.front() == '-') {
        throw std::runtime_error("Token IDs must be non-negative integers: " + std::string(text));
    }
    std::size_t parsed{};
    const auto value = std::stoull(std::string(text), &parsed);
    if (parsed != text.size()) {
        throw std::runtime_error("Invalid token ID: " + std::string(text));
    }
    return value;
}

void print_usage(const char* program) {
    std::cerr << "Usage: " << program
              << " [--model-dir PATH] [--model PATH] [--tokenizer PATH]"
                 " [--max-new-tokens COUNT] [--chat | --prompt TEXT | --tokens TOKEN_ID ...]\n";
}

void print_shape(const std::vector<std::uint64_t>& shape) {
    std::cout << '[';
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) std::cout << ", ";
        std::cout << shape[index];
    }
    std::cout << ']';
}

void print_tensor(const inference::TensorMetadata& tensor) {
    std::cout << tensor.name << ": dtype=" << tensor.dtype << ", shape=";
    print_shape(tensor.shape);
    std::cout << ", bytes=" << tensor.bytes << '\n';
}

void run_chat(const inference::Tokenizer& tokenizer,
              const inference::InferenceEngine& engine,
              std::int32_t max_new_tokens) {
    std::vector<inference::ChatMessage> history;
    std::cout << "Qwen3 interactive chat\n"
                 "Commands: /clear resets the conversation, /quit exits.\n\n";

    std::string input;
    while (true) {
        std::cout << "You> " << std::flush;
        if (!std::getline(std::cin, input)) {
            std::cout << '\n';
            return;
        }
        if (input == "/quit" || input == "/exit") return;
        if (input == "/clear") {
            history.clear();
            std::cout << "Conversation cleared.\n\n";
            continue;
        }
        if (input.empty()) continue;

        history.push_back({.role = "user", .content = input});
        try {
            const auto formatted = inference::format_qwen3_chat_prompt(history);
            const auto prompt_ids = tokenizer.encode(formatted);
            const auto result = engine.generate(prompt_ids, max_new_tokens);
            const auto response = tokenizer.decode(result.token_ids, true);
            history.push_back({.role = "assistant", .content = response});

            std::cout << "Assistant> " << response;
            if (response.empty() || response.back() != '\n') std::cout << '\n';
            std::cout << '\n';
        } catch (const std::exception& error) {
            history.pop_back();
            std::cerr << "error: " << error.what() << "\n\n";
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path model_directory = "models/Qwen3-0.6B";
    std::filesystem::path model_path = model_directory / "model.safetensors";
    std::filesystem::path tokenizer_path = model_directory / "tokenizer.json";
    std::vector<std::int32_t> token_ids;
    std::optional<std::string> prompt;
    bool chat_mode = false;
    std::int32_t max_new_tokens = 32;

    try {
        for (int index = 1; index < argc; ++index) {
            const std::string_view argument = argv[index];
            if (argument == "--model-dir") {
                if (++index == argc) {
                    print_usage(argv[0]);
                    return 1;
                }
                model_directory = argv[index];
                model_path = model_directory / "model.safetensors";
                tokenizer_path = model_directory / "tokenizer.json";
            } else if (argument == "--model") {
                if (++index == argc) {
                    print_usage(argv[0]);
                    return 1;
                }
                model_path = argv[index];
            } else if (argument == "--tokenizer") {
                if (++index == argc) {
                    print_usage(argv[0]);
                    return 1;
                }
                tokenizer_path = argv[index];
            } else if (argument == "--prompt") {
                if (++index == argc) {
                    print_usage(argv[0]);
                    return 1;
                }
                prompt = argv[index];
            } else if (argument == "--chat") {
                chat_mode = true;
            } else if (argument == "--max-new-tokens") {
                if (++index == argc) {
                    print_usage(argv[0]);
                    return 1;
                }
                const auto count = parse_token_id(argv[index]);
                if (count == 0 || count > std::numeric_limits<std::int32_t>::max()) {
                    throw std::runtime_error("--max-new-tokens must be a positive int32 value");
                }
                max_new_tokens = static_cast<std::int32_t>(count);
            } else if (argument == "--tokens") {
                while (++index < argc) {
                    const auto token_id = parse_token_id(argv[index]);
                    if (token_id > std::numeric_limits<std::int32_t>::max()) {
                        throw std::runtime_error("Token ID exceeds the int32 range");
                    }
                    token_ids.push_back(static_cast<std::int32_t>(token_id));
                }
            } else {
                print_usage(argv[0]);
                return 1;
            }
        }
        const auto selected_modes = static_cast<int>(chat_mode) +
                                    static_cast<int>(prompt.has_value()) +
                                    static_cast<int>(!token_ids.empty());
        if (selected_modes > 1) {
            throw std::runtime_error("Use only one of --chat, --prompt, or --tokens");
        }

        const inference::Tokenizer tokenizer(tokenizer_path);
        const inference::InferenceEngine engine(model_path);
        if (chat_mode || selected_modes == 0) {
            run_chat(tokenizer, engine, max_new_tokens);
            return 0;
        }
        if (prompt.has_value()) {
            token_ids = tokenizer.encode(inference::format_qwen3_chat_prompt(*prompt));
            std::cout << "Input token IDs:\n";
            for (std::size_t index = 0; index < token_ids.size(); ++index) {
                if (index != 0) std::cout << ' ';
                std::cout << token_ids[index];
            }
            std::cout << '\n';
        }
        if (token_ids.empty()) {
            print_usage(argv[0]);
            return 1;
        }

        const auto result = engine.generate(token_ids, max_new_tokens);

        std::cout << "Loaded " << result.model.path << " (" << result.model.tensor_count
                  << " tensors)\n";
        print_tensor(result.model.embedding);
        print_tensor(result.model.first_attention_norm);
        for (const auto& step : result.steps) {
            std::cout << "generated token " << step.index << ": id=" << step.token_id
                      << ", sequence_length=" << step.sequence_length
                      << (step.is_prefill ? ", prefill_gpu_ms=" : ", decode_gpu_ms=")
                      << std::fixed << std::setprecision(3) << step.gpu_milliseconds << '\n';
        }
        if (result.stopped_on_eos) std::cout << "stopped on EOS token 151645\n";

        std::cout << "Generated token IDs:\n";
        for (std::size_t index = 0; index < result.token_ids.size(); ++index) {
            if (index != 0) std::cout << ' ';
            std::cout << result.token_ids[index];
        }
        std::cout << "\nGenerated text:\n" << tokenizer.decode(result.token_ids, true) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
