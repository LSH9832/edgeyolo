/*
 * str.h
 *
 *  Created on: Mar 6, 2024
 *      Author: LSH9832
 */
#ifndef CPP_PY_STR_H
#define CPP_PY_STR_H


namespace pyStr {

	bool startswith(const std::string& str, const std::string& prefix) {
		size_t str_len = str.length();
		size_t prefix_len = prefix.length();
		if (prefix_len > str_len) return false;
		return str.find(prefix) == 0;
	}

	bool endswith(const std::string& str, const std::string& suffix) {
		size_t str_len = str.length();
		size_t suffix_len = suffix.length();
		if (suffix_len > str_len) return false;
		return (str.find(suffix, str_len - suffix_len) == (str_len - suffix_len));
	}

	bool isdigit(const std::string& str) {
	    for (char c : str) if (!std::isdigit(c)) return false;
	    return true;
	}

	std::string replace(std::string str, std::string old_substr, std::string new_substr) {
		size_t pos = 0;
		while ((pos = str.find(old_substr, pos)) != std::string::npos) {
			str.replace(pos, old_substr.length(), new_substr);
			pos += new_substr.length(); // 更新位置以继续搜索
		}
		return str;
	}

	std::vector<std::string> split(const std::string &s, const std::string &delimiter) {
	    std::vector<std::string> tokens;
	    size_t pos = 0;
	    std::string::size_type prev_pos = 0;
	    while ((pos = s.find(delimiter, prev_pos)) != std::string::npos) {
	        tokens.push_back(s.substr(prev_pos, pos - prev_pos));
	        prev_pos = pos + delimiter.length();
	    }
	    if (prev_pos < s.length()) tokens.push_back(s.substr(prev_pos));
	    return tokens;
	}

	std::vector<std::string> split(const std::string &s) {
		std::vector<std::string> tokens;
		size_t pos = 0;
		int prev_pos = -1;

		for (size_t i=0; i<s.length();i++) {
			if (i < prev_pos) continue;
			if ((pos = std::min(s.find("\n", i), std::min(s.find(" ", i), s.find("\t", i)))) != std::string::npos) {
				if (i == pos) {
					prev_pos = i;
					continue;
				}
				else {
					std::string substr = s.substr(prev_pos+1, pos - prev_pos - 1);
					if (substr.length()) {
						tokens.push_back(substr);
						prev_pos = pos;
					}
				}
			}
		}
		std::string last_substr = s.substr(prev_pos+1);
		if (prev_pos < s.length() && last_substr.length()) tokens.push_back(last_substr);
		return tokens;
	}

	std::string ljust(const std::string &str, int length) {
		std::string ret = str;
		if (str.length()>=length) return str;
		for (int i=0;i<length-str.length();i++) ret += " ";
		return ret;
	}

	std::string rjust(const std::string &str, int length) {
		std::string ret = str;
		if (str.length()>=length) return str;
		for (int i=0;i<length-str.length();i++) ret = " " + ret;
		return ret;
	}

	std::string zfill(const std::string &str, int length) {
		std::string ret = str;
		if (str.length()>=length) return str;
		for (int i=0;i<length-str.length();i++) ret = "0" + ret;
		return ret;
	}

	std::string zfill(const int num, int length) {
		std::string ret = std::to_string(num);
		if (ret.length()>=length) return ret;
		int ori_len = ret.length();
		for (int i=0;i<length-ori_len;i++) ret = "0" + ret;
		return ret;
	}

}

#endif
