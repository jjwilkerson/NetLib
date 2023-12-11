/**
 * @file
 * @brief Declares CorpusIterator class, a base class for objects that iterate over an input corpus.
 *
 * CorpusIterator class is a base class for objects that iterator over an input corpus. They provide batches
 * of samples for training.
 *
 */

#ifndef CORPUSITERATOR_H_
#define CORPUSITERATOR_H_

#include "../NetLib.h"
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <ctime>
#include <cassert>
#include <boost/regex.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/regex.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>

using namespace std;

namespace netlib {

/**
 * @brief A base class for objects that iterate over an input corpus.
 *
 * A base class for objects that iterator over an input corpus. They provide batches
 * of samples for training.
 *
 */
class CorpusIterator {
public:
	CorpusIterator(string corpusFilename, int batchSize, int maxSentenceLength,
			bool training, string ixFilename = "", ix_type ix = 0);
	virtual ~CorpusIterator();

	virtual bool hasNext();
	virtual void next(int **batchIx, unsigned int* lengths, vector<string>* tokensRet = NULL);
	virtual void inputFor(string* sentences, int count, int **batchIx, unsigned int* lengths, bool printTokens = false,
			vector<string>* tokensRet = NULL) = 0;
	void shuffle();
	void loadIxs(string filename);
	void saveIxs(string filename);
	ix_type currentIx();
	void reset();
	void resetTo(ix_type newIx);
	ix_type size();
	void get(ix_type selectedIx, string& sentence);
	string getIxFilename();
	void setIxFilename(string filename);
	unsigned getBatchSize();

protected:
	string corpusFilename;
	unsigned batchSize;
	unsigned maxSentenceLength;
	bool training;
	string ixFilename;
	ix_type ix;
	unsigned long currentBatch;
	ix_type *sentenceIxs;
	ix_type corpusSize;
	unsigned long totalBatches;
	ifstream corpusFile;

	void initCorpus();
	bool endsWith(const string& str, const string& ending);
};

} /* namespace netlib */

#endif /* CORPUSITERATOR_H_ */
