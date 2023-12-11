/**
 * @file
 * @brief Declares VecIterator, a CorpusIterator that iterates over a set of vectors.
 *
 */

#ifndef VECITERATOR_H_
#define VECITERATOR_H_

#include "../NetLib.h"
#include <string>
#include <fstream>
#include <iostream>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>

using namespace std;

namespace netlib {

/**
 * @brief A CorpusIterator that iterates over a set of vectors.
 *
 */
class VecIterator {
public:
	VecIterator(string corpusFilename, int vecSize, int batchSize, bool training,
			string ixFilename = "", ix_type ix = 0);
	virtual ~VecIterator();
	virtual bool hasNext();
	virtual void next(dtype2* input);
	virtual void inputFor(dtype2 *input, int count);
	void get(ix_type selectedIx, dtype2* d);
	void shuffle();
	void loadIxs(string filename);
	void saveIxs(string filename);
	ix_type currentIx();
	void reset();
	void resetTo(ix_type newIx);
	ix_type size();
	string getIxFilename();
	void setIxFilename(string filename);
	unsigned getBatchSize();

private:
	void initCorpus();

	string corpusFilename;
	int vecSize;
	int batchSize;
	bool training;
	string ixFilename;
	ix_type ix;
	unsigned long currentBatch;
	ix_type *vecIxs;
	ix_type corpusSize;
	unsigned long totalBatches;
	ifstream corpusFile;
	dtype2* h_input;
};

} /* namespace netlib */

#endif /* VECITERATOR_H_ */
