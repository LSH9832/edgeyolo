#pragma once
#ifndef PYTHONLIKE_STR_H
#define PYTHONLIKE_STR_H

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#ifndef MIN
  #define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif
#ifndef MAX
  #define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

class pystring {

    std::string str_;

public:

    pystring() {

    }

    pystring(std::string str) {
        str_ = str;
    }

    pystring(const char * str) {
        str_ = str;
    }

    pystring(int str)
    {
        str_ = std::to_string(str);
    }

    pystring(double str)
    {
        str_ = std::to_string(str);
    }

    // pystring(std::vector<int> )

    bool operator==(pystring &str) {
        return str_ == str.str();
    }

    bool operator==(std::string &str) {
        return str_ == str;
    }

    bool operator==(const char str[]) {
        return str_ == str;
    }

    pystring operator+(pystring &str) {
        return pystring(str_ + str.str());
    }

    pystring operator+(std::string str) {
        return pystring(str_ + str);
    }

    pystring operator+(const char * str) {
        return pystring(str_ + str);
    }

    pystring operator+(const char c) {
        return pystring(str_ + c);
    }

    pystring operator+(const int value) {
        return pystring(str_ + std::to_string(value));
    }

    pystring operator+(const double value) {
        return pystring(str_ + std::to_string(value));
    }

    void operator+=(pystring str) {
        str_ += str.str();
    }

    void operator+=(std::string str) {
        str_ += str;
    }

    void operator+=(const char * str) {
        str_ += str;
    }

    void operator+=(const char c) {
        str_ += c;
    }

    char operator[](int idx){
        return str_[idx<0?str_.length()+idx:idx];
    }

    operator int() const {
        return std::atoi(str_.c_str());
    }

    operator float() const {
        return std::atof(str_.c_str());
    }

    operator double() const {
        return std::atof(str_.c_str());
    }

    operator std::string() const {
        return str_;
    }

    operator bool() const {
        auto s = pystring(str_).lower().str();
        if (s == "false" || s == "0")
        {
            return false;
        }
        else if (s == "true" || s == "1")
        {
            return true;
        }
        else
        {
            std::cerr << "can not change string '" << str_ << "' to bool, return false" << std::endl;
            return false;
        }
    }

    pystring operator*(int value) {
        std::ostringstream oss;
        if(str_.size())
        {
            for(int i=0;i<value;i++)
            {
                oss << str_;
            }
        }
        return pystring(oss.str());
    }

    void operator*=(int value) {
        std::ostringstream oss;
        if(str_.size())
        {
            for(int i=0;i<value;i++)
            {
                oss << str_;
            }
        }
        str_ = oss.str();
    }

    bool startswith(const std::string& prefix) {
        size_t str_len = str_.length();
        size_t prefix_len = prefix.length();
        if (prefix_len > str_len) return false;
        return str_.find(prefix) == 0;
    }

    bool endswith(const std::string& suffix) {
        size_t str_len = str_.length();
        size_t suffix_len = suffix.length();
        if (suffix_len > str_len) return false;
        return (str_.find(suffix, str_len - suffix_len) == (str_len - suffix_len));
    }

    bool startswith(pystring &prefix) {
        size_t str_len = str_.length();
        size_t prefix_len = prefix.str().length();
        if (prefix_len > str_len) return false;
        return str_.find(prefix.str()) == 0;
    }

    bool endswith(pystring &suffix) {
        size_t str_len = str_.length();
        size_t suffix_len = suffix.str().length();
        if (suffix_len > str_len) return false;
        return (str_.find(suffix.str(), str_len - suffix_len) == (str_len - suffix_len));
    }

    inline char &at(size_t idx) {
        return str_.at(idx);
    }

    inline void append(std::initializer_list<char> __l) {
        str_.append(__l);
    }

    size_t length() {
        return str_.length();
    }

    inline bool empty(){
        return str_.empty();
    }

    inline char &back() {
        return str_.back();
    }

    inline void pop_back() {
        return str_.pop_back();
    }

    friend std::ostream& operator<<(std::ostream& _os, pystring obj) {
        _os << obj.str_;
        return _os;
    }

    std::vector<pystring> split(std::string delimiter) {
        std::vector<pystring> tokens;
        size_t pos = 0;
        std::string::size_type prev_pos = 0;
        while ((pos = str_.find(delimiter, prev_pos)) != std::string::npos) {
            tokens.push_back(pystring(str_.substr(prev_pos, pos - prev_pos)));
            prev_pos = pos + delimiter.length();
        }
        if (prev_pos < str_.length()) tokens.push_back(pystring(str_.substr(prev_pos)));
        if (endswith(delimiter)) tokens.push_back("");
        return tokens;
    }

    std::vector<pystring> split(pystring delimiter) {
        return this->split(delimiter.str());
    }

    std::vector<pystring> split(const char* delimiter) {
        return this->split(pystring(delimiter).str());
    }

    std::vector<pystring> split() {
        std::vector<pystring> tokens;
        size_t pos = 0;
        int prev_pos = -1;

        for (size_t i=0; i<str_.length();i++) {
            if (i < prev_pos) continue;
            if ((pos = std::min(str_.find("\n", i), std::min(str_.find(" ", i), str_.find("\t", i)))) != std::string::npos) {
                if (i == pos) {
                    prev_pos = i;
                    continue;
                }
                else {
                    pystring substr = str_.substr(prev_pos+1, pos - prev_pos - 1);
                    if (substr.length()) {
                        tokens.push_back(substr);
                        prev_pos = pos;
                    }
                }
            }
        }
        pystring last_substr = str_.substr(prev_pos+1);
        if (prev_pos < str_.length() && last_substr.length()) tokens.push_back(last_substr);
        return tokens;
    }

    pystring replace(std::string old_substr, std::string new_substr) {
        size_t pos = 0;
        while ((pos = str_.find(old_substr, pos)) != std::string::npos) {
            str_.replace(pos, old_substr.length(), new_substr);
            pos += new_substr.length(); // 更新位置以继续搜索
        }
        return str_;
    }

    pystring replace(pystring old_substr, pystring new_substr) {
        return this->replace(old_substr.str(), new_substr.str());
    }

    pystring replace(const char* old_substr, const char* new_substr) {
        return this->replace(pystring(old_substr), pystring(new_substr));
    }

    pystring zfill(int len) {
        std::string ret = str_;
        while (ret.length() < len) {
            ret = "0" + ret;
        }
        return pystring(ret);
    }

    bool isdigit() {
        for (char c : str_) if (!std::isdigit(c)) return false;
        return true;
    }

    pystring ljust(int length) {
        std::string ret = str_;
        if (str_.length()>=length) return str_;
        for (int i=0;i<length-str_.length();i++) ret += " ";
        return ret;
    }

    pystring rjust(int length) {
        std::string ret = str_;
        if (str_.length()>=length) return str_;
        for (int i=0;i<length-str_.length();i++) ret = " " + ret;
        return ret;
    }

    const std::string str() {
        return str_;
    }

    inline const char* c_str() {
        return str_.c_str();
    }

    pystring upper() {
        pystring ret = "";
        for (int i=0;i<str_.length();i++) {
            int assic = str_[i];
            if (assic < 123 && assic > 96) {
                ret += (char)(assic - 32);
            }
            else ret += str_[i];
        }
        return ret;
    }

    pystring lower() {
        pystring ret = "";
        for (int i=0;i<str_.length();i++) {
            int assic = str_[i];
            if (assic < 91 && assic > 64) {
                ret += (char)(assic + 32);
            }
            else ret += str_[i];
        }
        return ret;
    }


};


#endif