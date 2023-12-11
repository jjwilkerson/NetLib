/**
 * @file
 * @brief Declares VecInputSource class, which serves vectors as input to a network.
 *
 */

#ifndef VECINPUTSOURCE_H_
#define VECINPUTSOURCE_H_

#include <string>
#include "InputSource.h"

using namespace std;

namespace netlib {

class VecIterator;

/**
 * @brief Serves vectors as input to a network.
 *
 */
class VecInputSource: public InputSource {
public:
	VecInputSource(VecIterator& corpus, int batchSize, int vecLength, int maxSeqLength, int maxNumBatch,
			ix_type startIx);
	virtual ~VecInputSource();
	void toFirstBatch();
	bool hasNextBatchSet();
	void toNextBatchSet();
	bool hasNext();
	void nextBatch(int batchNum, vector<string>* tokensRet = NULL);
	void next();
	dtype2** getRevInputs();
	dtype2** getFwdInputs();
	unsigned int** getDInputLengths();
	unsigned int* getHInputLengths();
	dtype2** getTargets();
	void computeMatchMasks(int batchNum, dtype2** outputs);
	dtype2** matchMasks(int batchNum);
	void reset();
	void shuffle();
	void loadIxs(string filename);
	void saveIxs(string filename);
	string getIxFilename();
	int getBatchSize();
	int getVecLength();
	int maxNumBatch;
	ix_type firstBatchIx;
private:
	void initMem();
	void freeMem();
	VecIterator& corpus;
	int batchSize;
	int vecLength;
	int maxSeqLength;
	int numBatchSaved;
	dtype2 **d_inputs_rev = NULL;
	dtype2 **d_targets = NULL;
	unsigned int **d_inputLengths = NULL;
	unsigned int *h_inputLengths = NULL;
};

} /* namespace netlib */

#endif /* VECINPUTSOURCE_H_ */
