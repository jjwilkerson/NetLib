/**
 * @file
 * @brief Declares Tokenizer class, which tokenizes sentences and replaces digit symbols with words.
 *
 */

#ifndef TOKENIZER_H_
#define TOKENIZER_H_

#include <string>
#include <vector>

using namespace std;

namespace netlib {

/**
 * @brief Tokenizes sentences and replaces digit symbols with words..
 *
 */
class Tokenizer {
public:
	Tokenizer();
	virtual ~Tokenizer();
	static void tokenize(string sent, vector<string>& tokens);
	static string detokenize(vector<string>* tokens);
	static string replaceDigitsSymbolToWord(string token);
	static string replaceDigitsWordToSymbol(string token);
private:
	static bool endsWith(const string& str, const string& ending);
};

} /* namespace netlib */

#endif /* TOKENIZER_H_ */
