/*
 * copy from https://github.com/0382/util/blob/main/cpp/argparse/argparse.hpp
 */
#pragma once
#ifndef JSHL_ARGPARSE_HPP
#define JSHL_ARGPARSE_HPP

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <vector>

namespace argsutil
{

// 尽管使用编译器相关的 ABI 函数可以相对比较优雅的实现这个功能
// 但是不同的编译器在某些类型下可能出现一些奇怪的行为
// 最重要的是 std::string 还是免不了要模板特例化
// 因此，不如在这里限定一些类型，免去不可控制的行为

// 我们仅支持 bool, int, int64_t, double, std::string
// 想要其他长度的类型，获取值之后自行转换
// 当然，如果你愿意的话，自己定义模板特例化也不是不可以

template <typename T>
inline std::string type_string()
{
    return "null";
}

template <>
inline std::string type_string<bool>()
{
    return "bool";
}

template <>
inline std::string type_string<int>()
{
    return "int";
}

template <>
inline std::string type_string<int64_t>()
{
    return "int64_t";
}

template <>
inline std::string type_string<double>()
{
    return "double";
}

template <>
inline std::string type_string<std::string>()
{
    return "string";
}

template <typename T>
std::string to_string(const T &value)
{
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

template <typename T>
T parse_value(const std::string &value)
{
    std::istringstream iss(value);
    T result;
    iss >> result;
    return result;
}

struct short_circuit_option
{
    short_circuit_option(std::string sname, std::string lname, std::string help, std::function<void(void)> callback)
        : short_name(std::move(sname)), long_name(std::move(lname)), help(std::move(help)),
          callback(std::move(callback))
    {}
    std::string short_name;
    std::string long_name;
    std::string help;
    std::function<void(void)> callback;
};

struct option
{
    option(std::string sname, std::string lname, std::string help, std::string type, std::string value)
        : short_name(std::move(sname)), long_name(std::move(lname)), help(std::move(help)), type(std::move(type)),
          value(std::move(value))
    {}

    std::string short_name;
    std::string long_name;
    std::string help;
    std::string type;
    std::string value;
};

struct argument
{
    argument(std::string name, std::string help, std::string type)
        : name(std::move(name)), help(std::move(help)), type(std::move(type))
    {}

    std::string name;
    std::string help;
    std::string type;
    std::string value;
};

class argparser
{
  private:
    std::string description;
    std::string program_name;
    std::vector<short_circuit_option> short_circuit_options;
    std::vector<option> options;
    std::unordered_map<char, std::size_t> short_name_index;
    std::vector<argument> named_arguments;
    std::vector<argument> arguments;

  public:
    argparser(std::string description) : description(std::move(description)) {}

    argparser &set_program_name(std::string name)
    {
        program_name = std::move(name);
        return *this;
    }

    void print_usage() const
    {
        std::cout << "usage: " << program_name << " [options]";
        for (const auto &named_arg : named_arguments)
        {
            std::cout << " [=" << named_arg.name << "]";
        }
        for (const auto &arg : arguments)
        {
            std::cout << " [" << arg.name << "]";
        }
        std::cout << std::endl;
    }

    void print_help() const
    {
        print_usage();
        std::cout << "\n" << description << "\n\n";
        std::cout << "Options:\n";
        // calculate the longest option name
        std::size_t max_name_length = 0;
        for (const auto &opt : short_circuit_options)
        {
            std::size_t length = opt.long_name.length();
            if (!opt.short_name.empty())
            {
                length += 4;
            }
            max_name_length = std::max(max_name_length, length);
        }
        for (const auto &opt : options)
        {
            std::size_t length = opt.long_name.length();
            if (!opt.short_name.empty())
            {
                length += 4;
            }
            max_name_length = std::max(max_name_length, length);
        }
        max_name_length = std::max(max_name_length, std::size_t(25));
        // print the options
        for (const auto &opt : short_circuit_options)
        {
            std::cout << "  ";
            std::size_t printed_length = 0;
            if (!opt.short_name.empty())
            {
                std::cout << opt.short_name << ", ";
                printed_length = 4;
            }
            std::cout << opt.long_name;
            printed_length += opt.long_name.length();
            std::cout << std::string(max_name_length - printed_length, ' ');
            std::cout << replace(opt.help, "\n", "\n" + std::string(max_name_length + 2, ' ')) << '\n';
        }
        for (const auto &opt : options)
        {
            std::cout << "  ";
            std::size_t printed_length = 0;
            if (!opt.short_name.empty())
            {
                std::cout << opt.short_name << ", ";
                printed_length = 4;
            }
            std::cout << opt.long_name;
            printed_length += opt.long_name.length();
            std::cout << std::string(max_name_length - printed_length, ' ');
            if (opt.type != "bool")
            {
                std::cout << "(" << opt.type << ") ";
            }
            std::cout << replace(opt.help, "\n", "\n" + std::string(max_name_length + 2, ' ')) << '\n';
        }
        if (named_arguments.size() > 0)
        {
            std::cout << "\nNamed arguments:\n";
            max_name_length = 0;
            for (const auto &arg : named_arguments)
            {
                max_name_length = std::max(max_name_length, arg.name.length());
            }
            max_name_length = std::max(max_name_length, std::size_t(25));
            for (const auto &arg : named_arguments)
            {
                std::cout << "  ";
                std::cout << arg.name;
                std::cout << std::string(max_name_length - arg.name.length(), ' ') << "(" << arg.type << ") ";
                std::cout << replace(arg.help, "\n", "\n" + std::string(max_name_length + 2, ' ')) << '\n';
            }
        }
        if (arguments.size() > 0)
        {
            std::cout << "\nPosition arguments:\n";
            max_name_length = 0;
            for (const auto &arg : arguments)
            {
                max_name_length = std::max(max_name_length, arg.name.length());
            }
            max_name_length = std::max(max_name_length, std::size_t(25));
            for (const auto &arg : arguments)
            {
                std::cout << "  ";
                std::cout << arg.name;
                std::cout << std::string(max_name_length - arg.name.length(), ' ') << "(" << arg.type << ") ";
                std::cout << replace(arg.help, "\n", "\n" + std::string(max_name_length + 2, ' ')) << '\n';
            }
        }
    }

    argparser &add_help_option()
    {
        return add_sc_option("-?", "--help", "show this help message", [this]() { print_help(); });
    }

    // add short circuit option
    argparser &add_sc_option(std::string sname, std::string lname, std::string help, std::function<void(void)> callback)
    {
        // long name must not be empty
        check_add_option_lname(lname);
        // allow short name to be empty
        if (sname != "")
        {
            check_add_option_sname(sname);
            short_name_index[sname.back()] = short_circuit_options.size();
        }
        short_circuit_options.emplace_back(std::move(sname), std::move(lname), std::move(help), std::move(callback));
        return *this;
    }

    template <typename T>
    argparser &add_option(std::string sname, std::string lname, std::string help, T &&default_value)
    {
        if (type_string<T>() == "null")
        {
            std::cerr << "(build error) unsupport type for option: " << typeid(T).name() << std::endl;
            std::exit(-1);
        }
        check_add_option_lname(lname);
        if (sname != "")
        {
            check_add_option_sname(sname);
            short_name_index[sname.back()] = options.size();
        }
        options.emplace_back(std::move(sname), std::move(lname), std::move(help), type_string<T>(),
                             to_string(default_value));
        return *this;
    }

    argparser &add_option(std::string sname, std::string lname, std::string help)
    {
        check_add_option_lname(lname);
        if (sname != "")
        {
            check_add_option_sname(sname);
            short_name_index[sname.back()] = options.size();
        }
        options.emplace_back(std::move(sname), std::move(lname), std::move(help), "bool", "0");
        return *this;
    }

    template <typename T>
    argparser &add_argument(std::string name, std::string help)
    {
        check_add_argument_name<T>(name);
        arguments.emplace_back(std::move(name), std::move(help), type_string<T>());
        return *this;
    }

    template <typename T>
    argparser &add_named_argument(std::string name, std::string help)
    {
        check_add_argument_name<T>(name);
        named_arguments.emplace_back(std::move(name), std::move(help), type_string<T>());
        return *this;
    }

    template <typename T>
    T get_option(const std::string &name) const
    {
        auto pos = find_option_sname(name);
        if (pos == options.cend())
        {
            pos = find_option_lname(name);
        }
        if (pos == options.cend())
        {
            std::cerr << "(get error) option not found: " << name << std::endl;
            std::exit(-1);
        }
        if (pos->type != type_string<T>())
        {
            std::cerr << "(get error) option type mismatch: set '" << pos->type << "' but you try get with '"
                      << type_string<T>() << "'" << std::endl;
            std::exit(-1);
        }
        return parse_value<T>(pos->value);
    }

    // some alias for get_option
    bool has_option(const std::string &name) const { return get_option<bool>(name); }
    bool get_option_bool(const std::string &name) const { return get_option<bool>(name); }
    int get_option_int(const std::string &name) const { return get_option<int>(name); }
    int64_t get_option_int64(const std::string &name) const { return get_option<int64_t>(name); }
    double get_option_double(const std::string &name) const { return get_option<double>(name); }
    std::string get_option_string(const std::string &name) const { return get_option<std::string>(name); }

    template <typename T>
    T get_argument(const std::string &name) const
    {
        auto pos = find_argument(name);
        if (pos == arguments.cend())
        {
            pos = find_named_argument(name);
        }
        if (pos == named_arguments.cend())
        {
            std::cerr << "(get error) argument not found: " << name << std::endl;
            std::exit(-1);
        }
        if (pos->type != type_string<T>())
        {
            std::cerr << "(get error) argument type mismatch: set '" << pos->type << "' but you try get with '"
                      << type_string<T>() << "'" << std::endl;
            std::exit(-1);
        }
        return parse_value<T>(pos->value);
    }

    // some alias for get_argument
    int get_argument_int(const std::string &name) const { return get_argument<int>(name); }
    int64_t get_argument_int64(const std::string &name) const { return get_argument<int64_t>(name); }
    double get_argument_double(const std::string &name) const { return get_argument<double>(name); }
    std::string get_argument_string(const std::string &name) const { return get_argument<std::string>(name); }

    // parse arguments
    argparser &parse(int argc, char *argv[])
    {
        // if not set program name, use argv[0]
        if (program_name == "")
        {
            program_name = argv[0];
        }
        if (argc == 1)
        {
            print_usage();
            std::exit(0);
        }
        std::vector<std::string> tokens;
        for (int i = 1; i < argc; ++i)
        {
            tokens.emplace_back(argv[i]);
        }
        // start parse short circuit options
        for (auto &&sc_opt : short_circuit_options)
        {
            auto pos = std::find_if(tokens.cbegin(), tokens.cend(),
                                    [&sc_opt](const std::string &tok)
                                    { return tok == sc_opt.short_name || tok == sc_opt.long_name; });
            if (pos != tokens.cend())
            {
                sc_opt.callback();
                std::exit(0);
            }
        }
        // start parse options
        for (auto &&opt : options)
        {
            auto pos =
                std::find_if(tokens.cbegin(), tokens.cend(),
                             [&opt](const std::string &tok) { return tok == opt.short_name || tok == opt.long_name; });
            if (pos == tokens.cend())
            {
                continue;
            }
            pos = tokens.erase(pos);
            if (opt.type == "bool")
            {
                opt.value = "1";
            }
            else // other types need to parse next token
            {
                if (pos == tokens.cend())
                {
                    std::cerr << "(parse error) option " << opt.short_name << " " << opt.long_name
                              << " should have value" << std::endl;
                    std::exit(-1);
                }
                opt.value = *pos;
                pos = tokens.erase(pos);
            }
        }
        // if there are short name like options, parse it as aggregation short name options
        {
            auto pos =
                std::find_if(tokens.cbegin(), tokens.cend(), [](const std::string &tok) { return tok.front() == '-'; });
            if (pos != tokens.cend())
            {
                if (pos->length() == 1)
                {
                    std::cerr << "(parse error) bare unexcepted '-'" << std::endl;
                    std::exit(-1);
                }
                if ((*pos)[1] == '-')
                {
                    std::cerr << "(parse error) unrecognized option" << (*pos) << std::endl;
                    std::exit(-1);
                }
                std::string short_names = pos->substr(1);
                for (char ch : short_names)
                {
                    std::size_t index = short_name_index[ch];
                    if (index < short_circuit_options.size() && short_circuit_options[index].short_name.back() == ch)
                    {
                        short_circuit_options[index].callback();
                        std::exit(0);
                    }
                }
                for (char ch : short_names)
                {
                    std::size_t index = short_name_index[ch];
                    if (index < options.size() && options[index].short_name.back() == ch)
                    {
                        if (options[index].type != "bool")
                        {
                            std::cerr << "(parse error) aggregation short name option must be bool" << std::endl;
                            std::exit(-1);
                        }
                        options[index].value = "1";
                    }
                    else
                    {
                        std::cerr << "(parse error) unrecognized short name option '" << ch << "' in " << (*pos)
                                  << std::endl;
                        std::exit(-1);
                    }
                }
                pos = tokens.erase(pos);
            }
        }
        // start parse named arguments
        if (tokens.size() < named_arguments.size())
        {
            std::cerr << "(parse error) not enough named_arguments" << std::endl;
            std::exit(-1);
        }
        for (auto &named_arg : named_arguments)
        {
            for (auto pos = tokens.begin(); pos != tokens.end();)
            {
                if (try_parse_named_argument(*pos, named_arg))
                {
                    pos = tokens.erase(pos);
                    break;
                }
                ++pos;
            }
            if (named_arg.value == "")
            {
                std::cerr << "(parse error) named_argument " << named_arg.name << " should have value" << std::endl;
                std::exit(-1);
            }
        }
        // start parse position arguments
        if (tokens.size() != arguments.size())
        {
            std::cerr << "(parse error) position argument number missmatching, give " << tokens.size() << ", but need "
                      << arguments.size() << '\n';
            std::cerr << "uncaptured command line arguments:\n";
            for (const auto &tok : tokens)
            {
                std::cerr << tok << '\n';
            }
            std::cerr << std::flush;
            std::exit(-1);
        }
        for (std::size_t i = 0; i < tokens.size(); ++i)
        {
            arguments[i].value = tokens[i];
        }
        return *this;
    }

    // print to file
    void print_as_ini(std::ostream &os, bool comments = false) const
    {
        if (options.size() > 0)
        {
            os << "[options]\n";
        }
        for (const auto &opt : options)
        {
            if (comments)
            {
                os << "# " << replace(opt.help, "\n", "\n# ") << "\n";
            }
            if (opt.type == "bool")
            {
                os << opt.long_name.substr(2) << "=" << (opt.value == "1" ? "true" : "false") << "\n";
            }
            else
            {
                os << opt.long_name.substr(2) << "=" << opt.value << '\n';
            }
        }
        if (named_arguments.size() > 0)
        {
            os << "[named_arguments]\n";
        }
        for (const auto &named_arg : named_arguments)
        {
            if (comments)
            {
                os << "# " << replace(named_arg.help, "\n", "\n# ") << "\n";
            }
            os << named_arg.name << "=" << named_arg.value << '\n';
        }
        if (arguments.size() > 0)
        {
            os << "[arguments]\n";
        }
        for (const auto &arg : arguments)
        {
            if (comments)
            {
                os << "# " << replace(arg.help, "\n", "\n# ") << "\n";
            }
            os << arg.name << "=" << arg.value << '\n';
        }
    }

  private:
    bool try_parse_named_argument(const std::string &line, argument &named_arg)
    {
        auto pos = line.find('=');
        if (pos == std::string::npos)
        {
            return false;
        }
        auto name = line.substr(0, pos);
        auto value = line.substr(pos + 1);
        if (name != named_arg.name)
        {
            return false;
        }
        else
        {
            named_arg.value = value;
            return true;
        }
    }

    std::string replace(const std::string &str, const std::string &from, const std::string &to) const
    {
        std::string ret;
        std::size_t pos = 0, pre_pos = 0;
        while ((pos = str.find(from, pre_pos)) != std::string::npos)
        {
            ret += str.substr(pre_pos, pos - pre_pos) + to;
            pre_pos = pos + from.length();
        }
        ret += str.substr(pre_pos);
        return ret;
    }

    using argument_iterator = std::vector<argument>::const_iterator;
    using option_iterator = std::vector<option>::const_iterator;
    using sc_option_iterator = std::vector<short_circuit_option>::const_iterator;

    auto find_argument(const std::string &key) const -> argument_iterator
    {
        return std::find_if(arguments.cbegin(), arguments.cend(),
                            [&key](const argument &arg) { return arg.name == key; });
    }

    auto find_named_argument(const std::string &key) const -> argument_iterator
    {
        return std::find_if(named_arguments.cbegin(), named_arguments.cend(),
                            [&key](const argument &arg) { return arg.name == key; });
    }

    auto find_sc_option_sname(const std::string &key) const -> sc_option_iterator
    {
        return std::find_if(short_circuit_options.cbegin(), short_circuit_options.cend(),
                            [&key](const short_circuit_option &opt) { return opt.short_name == key; });
    }

    auto find_sc_option_lname(const std::string &key) const -> sc_option_iterator
    {
        return std::find_if(short_circuit_options.cbegin(), short_circuit_options.cend(),
                            [&key](const short_circuit_option &opt) { return opt.long_name == key; });
    }

    auto find_option_sname(const std::string &key) const -> option_iterator
    {
        return std::find_if(options.cbegin(), options.cend(),
                            [&key](const option &opt) { return opt.short_name == key; });
    }

    auto find_option_lname(const std::string &key) const -> option_iterator
    {
        return std::find_if(options.cbegin(), options.cend(),
                            [&key](const option &opt) { return opt.long_name == key; });
    }

    void check_add_option_sname(const std::string &key) const
    {
        if (key.size() != 2 || key.front() != '-')
        {
            std::cerr << "(build error) short option name must be `-` followed by one character" << std::endl;
            std::exit(-1);
        }
        char ch = key.back();
        if (short_name_index.find(ch) != short_name_index.end())
        {
            std::cerr << "(build error) short option name " << key << " already exists" << std::endl;
            std::exit(-1);
        }
    }

    void check_add_option_lname(const std::string &key) const
    {
        if (key == "")
        {
            std::cerr << "(build error) long option name cannot be empty" << std::endl;
            std::exit(-1);
        }
        if (key.substr(0, 2) != "--")
        {
            std::cerr << "(build error) long option name must be `--` followed by one or more characters" << std::endl;
            std::exit(-1);
        }
        if (find_option_lname(key) != options.cend() || find_sc_option_lname(key) != short_circuit_options.cend())
        {
            std::cerr << "(build error) long option name " << key << " already exists" << std::endl;
            std::exit(-1);
        }
    }

    template <typename T>
    void check_add_argument_name(const std::string &key) const
    {
        if (type_string<T>() == "null")
        {
            std::cerr << "(build error) argument type is not supported: " << typeid(T).name() << std::endl;
            std::exit(-1);
        }
        if (type_string<T>() == "bool")
        {
            std::cerr << "(build error) argument type cannot be bool" << std::endl;
            std::exit(-1);
        }
        if (key == "")
        {
            std::cerr << "(build error) argument name cannot be empty" << std::endl;
            std::exit(-1);
        }
        if (find_argument(key) != arguments.cend() || find_named_argument(key) != named_arguments.cend())
        {
            std::cerr << "(build error) argument name " << key << " already exists" << std::endl;
            std::exit(-1);
        }
    }
};

}; // namespace util

#endif // JSHL_ARGPARSE_HPP
