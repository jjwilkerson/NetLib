/**
 * @file
 * @brief Declares WvCorpusIterator class, which iterates over a corpus of sentences.
 *
 * WvCorpusIterator class iterates over a corpus of sentences, encoding each word or punctuation symbol as a Word2Vec vector
 * or "special" vector.
 */

#ifndef WVCORPUSITERATOR_H_
#define WVCORPUSITERATOR_H_

#include <string>
#include <map>
#include "CorpusIterator.h"
#include "WvParser.h"

using namespace std;

namespace netlib {

class Config;
class WordVectors;

/**
 * @brief Iterates over a corpus of sentences.
 *
 * Iterates over a corpus of sentences, encoding each word or punctuation symbol as a Word2Vec vector
 * or "special" vector.
 */
class WvCorpusIterator: public CorpusIterator {
public:
	WvCorpusIterator(Config& config, WordVectors &wv, string corpusFilename, int batchSize, int maxSentenceLength,
			bool training, bool replace = false, unsigned int replacePeriod = 200, string ixFilename = "", ix_type ix = 0);
	virtual ~WvCorpusIterator();
	void inputFor(string* sentences, int count, int **batchIx, unsigned int* lengths, bool printTokens = false,
				vector<string>* tokensRet = NULL);
	void tokenize(const string sent, vector<string>& tokens);
	void saveReplacerState(const char* stateFilename);

private:
	WvParser* wvParser;
};

} /* namespace netlib */

#endif /* WVCORPUSITERATOR_H_ */
