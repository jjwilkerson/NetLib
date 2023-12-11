/**
 * @file
 * @brief Defines Tokenizer class, which tokenizes sentences and replaces digit symbols with words.
 *
 */

#include "Tokenizer.h"

#include <iostream>
#include <string>
#include <boost/regex.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/regex.hpp>

using namespace std;

namespace netlib {

Tokenizer::Tokenizer() {
}

Tokenizer::~Tokenizer() {
}

void Tokenizer::tokenize(string sent, vector<string> &tokens) {
	boost::trim(sent);
	if (endsWith(sent, " .")) {
		sent.resize(sent.length() - 2);
	} else if (endsWith(sent, " . \"")) {
		sent.resize(sent.length() - 2);
		sent[sent.length()-1] = '\"';
	} else if (endsWith(sent, " . )")) {
		sent.resize(sent.length() - 2);
		sent[sent.length()-1] = ')';
	} else if (endsWith(sent, " . '")) {
		sent.resize(sent.length() - 2);
		sent[sent.length()-1] = '\'';
	}

	boost::regex re("[|\\\\]+");
	sent = boost::regex_replace(sent, re, " ");

	re.set_expression("\\?+!+");
	sent = boost::regex_replace(sent, re, " ? ! ");

	re.set_expression("!{2,}");
	sent = boost::regex_replace(sent, re, " ! ");

	boost::trim(sent);

	boost::split_regex(tokens, sent, boost::regex("\\s+"));

	unsigned int numTokens = tokens.size();
	for (int k = 0; k < numTokens; k++) {
		string token = tokens[k];

		if (token == "EOS" && k < (numTokens - 1)) {
//			cout << "changing EOS to EOS_" << endl;
			token = "EOS_";
			tokens[k] = token;
		} else if (token == "UNK") {
//			cout << "changing UNK to UNK_" << endl;
			token = "UNK_";
			tokens[k] = token;
		} else if (token == "[" || token == "{") {
			token = "(";
			tokens[k] = token;
		} else if (token == "]" || token == "}") {
			token = ")";
			tokens[k] = token;
		} else {
			token = replaceDigitsSymbolToWord(token);
//				token = replaceDigitsWordToSymbol(token);
			tokens[k] = token;
		}
	}
}

string Tokenizer::detokenize(vector<string>* tokens) {
	if (tokens->size() == 0) {
		return "";
	}
	string s = (*tokens)[0];
	for (int i = 1; i < tokens->size(); i++) {
		s.append(" ");
		s.append((*tokens)[i]);
	}
	return s;
}

string Tokenizer::replaceDigitsSymbolToWord(string token) {
	if (token == "0") {
		return "zero";
	} else if (token == "1") {
		return "one";
	} else if (token == "2") {
		return "two";
	} else if (token == "3") {
		return "three";
	} else if (token == "4") {
		return "four";
	} else if (token == "5") {
		return "five";
	} else if (token == "6") {
		return "six";
	} else if (token == "7") {
		return "seven";
	} else if (token == "8") {
		return "eight";
	} else if (token == "9") {
		return "nine";
	} else {
		return token;
	}
}

string Tokenizer::replaceDigitsWordToSymbol(string token) {
	if (token == "zero") {
		return "0";
	} else if (token == "one") {
		return "1";
	} else if (token == "two") {
		return "2";
	} else if (token == "three") {
		return "3";
	} else if (token == "four") {
		return "4";
	} else if (token == "five") {
		return "5";
	} else if (token == "six") {
		return "6";
	} else if (token == "seven") {
		return "7";
	} else if (token == "eight") {
		return "8";
	} else if (token == "nine") {
		return "9";
	} else {
		return token;
	}

}

bool Tokenizer::endsWith(const string &str, const string &ending) {
	if (str.length() >= ending.length()) {
		return (0 == str.compare(str.length() - ending.length(), ending.length(), ending));
	} else {
		return false;
	}
}

} /* namespace netlib */
