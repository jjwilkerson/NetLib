/**
 * @file
 * @brief Defines WvCorpusIterator class, which iterates over a corpus of sentences.
 *
 * WvCorpusIterator class iterates over a corpus of sentences, encoding each word or punctuation symbol as a Word2Vec vector
 * or "special" vector.
 */

#include "WvCorpusIterator.h"

namespace netlib {

WvCorpusIterator::WvCorpusIterator(Config& config, WordVectors &wv, string corpusFilename, int batchSize, int maxSentenceLength,
		bool training, bool replace, unsigned int replacePeriod, string ixFilename, ix_type ix)
			: CorpusIterator(corpusFilename, batchSize, maxSentenceLength, training, ixFilename, ix) {
	wvParser = new WvParser(config, wv, maxSentenceLength, replace, replacePeriod);
}

WvCorpusIterator::~WvCorpusIterator() {
	delete wvParser;
}

void WvCorpusIterator::inputFor(string* sentences, int count, int **batchIx, unsigned int* lengths,
		bool printTokens, vector<string>* tokensRet) {
	wvParser->inputFor(sentences, count, batchIx, lengths, printTokens, tokensRet);
}

void WvCorpusIterator::tokenize(string sent, vector<string>& tokens) {
	wvParser->tokenize(sent, tokens);
}

void WvCorpusIterator::saveReplacerState(const char *stateFilename) {
	wvParser->saveReplacerState(stateFilename);
}

} /* namespace netlib */
