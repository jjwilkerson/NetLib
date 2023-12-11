/**
 * @file
 * @brief Defines WvParser class, which iterates over a corpus of sentences.
 *
 * The WvParser class iterates over a corpus of sentences, encoding each word or punctuation symbol as a Word2Vec vector
 * or "special" vector. It implements the functionality of the WvCorpusIterator class, separated out so it can be used by other
 * classes.
 */

#include "WvParser.h"

#include <boost/regex.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/regex.hpp>
#include "../config/Config.h"
#include "Corrector.h"
#include "Replacer.h"
#include "WordVectors.h"
#include "Tokenizer.h"

namespace netlib {

WvParser::WvParser(Config& config, WordVectors &wv, int maxSentenceLength, bool replace, unsigned int replacePeriod)
			: wv(wv), maxSentenceLength(maxSentenceLength), replace(replace) {
	corrector = new Corrector();
	replacer = new Replacer(config.seed, config.replacerStateFilename, replacePeriod);
}

WvParser::~WvParser() {
	delete corrector;
	delete replacer;
}

void WvParser::inputFor(string *sentences, int count, int **batchIx,
		unsigned int *lengths, bool printTokens, vector<string> *tokensRet) {
	//	cout << endl << "inputFor" << endl;
		for (int j = 0; j < count; j++) {
			string sentence = sentences[j];

			vector<string>* tokens;
			if (tokensRet == NULL) {
				tokens = new vector<string>();
			} else {
				tokens = &tokensRet[j];
			}

			tokens->clear();
			tokenize(sentence, *tokens);
			tokens->push_back("EOS");

			if (printTokens) {
				cout << "t:";
				vector<string>::iterator it;
				for (it = tokens->begin(); it != tokens->end(); it++) {
					cout << " " << *it;
				}
				cout << endl;
			}

			unsigned int numTokens = tokens->size();
			assert (numTokens <= maxSentenceLength);

			for (int k = 0; k < numTokens; k++) {
				string token = (*tokens)[k];

				if (token == "EOS" && k < (numTokens - 1)) {
//					cout << "changing EOS to EOS_" << endl;
					token = "EOS_";
					(*tokens)[k] = token;
				} else if (token == "UNK") {
//					cout << "changing UNK to UNK_" << endl;
					token = "UNK_";
					(*tokens)[k] = token;
				} else if (token == "[" || token == "{") {
					token = "(";
					(*tokens)[k] = token;
				} else if (token == "]" || token == "}") {
					token = ")";
					(*tokens)[k] = token;
				} else {
					token = Tokenizer::replaceDigitsSymbolToWord(token);
	//				token = replaceDigitsWordToSymbol(token);
					(*tokens)[k] = token;
				}

				int ix = wv.indexOf(token);
				if (ix == wv.indexOf("UNK")) {
					(*tokens)[k] = "UNK";
				}

				if (batchIx != NULL) {
					batchIx[k][j] = ix;
				}
			}

			lengths[j] = numTokens;

			if (tokensRet == NULL) {
				delete tokens;
			}
		}
}

void WvParser::tokenize(string sent, vector<string> &tokens) {
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

	corrector->correct(tokens);
	if (replace) {
		replacer->replace(tokens);
	}
}

void WvParser::saveReplacerState(const char *stateFilename) {
	replacer->saveState(stateFilename);
}

bool WvParser::endsWith(const string &str, const string &ending) {
	if (str.length() >= ending.length()) {
		return (0 == str.compare(str.length() - ending.length(), ending.length(), ending));
	} else {
		return false;
	}
}

} /* namespace netlib */
