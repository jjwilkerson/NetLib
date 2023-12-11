/**
 * @file
 * @brief Declares WvParser class, which iterates over a corpus of sentences.
 *
 * The WvParser class iterates over a corpus of sentences, encoding each word or punctuation symbol as a Word2Vec vector
 * or "special" vector. It implements the functionality of the WvCorpusIterator class, separated out so it can be used by other
 * classes.
 */

#ifndef WVPARSER_H_
#define WVPARSER_H_

#include <string>
#include <vector>

using namespace std;

namespace netlib {

class Config;
class Corrector;
class Replacer;
class WordVectors;

/**
 * @brief Iterates over a corpus of sentences.
 *
 * Iterates over a corpus of sentences, encoding each word or punctuation symbol as a Word2Vec vector
 * or "special" vector. Implements the functionality of the WvCorpusIterator class, separated out so it can be used by other
 * classes.
 */
class WvParser {
public:
	WvParser(Config& config, WordVectors &wv, int maxSentenceLength, bool replace = false, unsigned int replacePeriod = 200);
	virtual ~WvParser();
	void inputFor(string* sentences, int count, int **batchIx, unsigned int* lengths, bool printTokens = false,
				vector<string>* tokensRet = NULL);
	void tokenize(string sent, vector<string>& tokens);
	void saveReplacerState(const char* stateFilename);

private:
	static bool endsWith(const string& str, const string& ending);
	WordVectors &wv;
	Corrector* corrector;
	Replacer* replacer;
	bool replace;
	int maxSentenceLength;
};

} /* namespace netlib */

#endif /* WVPARSER_H_ */
