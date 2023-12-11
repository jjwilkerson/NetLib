/**
 * @file
 * @brief Declares SedInputSource class, which serves sentences as input to a network.
 *
 */

#ifndef SEDINPUTSOURCE_H_
#define SEDINPUTSOURCE_H_

#include <string>
#include <vector>
#include "InputSource.h"

using namespace std;

namespace netlib {

class WordVectors;
class WvCorpusIterator;

/**
 * @brief Serves sentences as input to a network.
 *
 */
class SedInputSource: public InputSource {
public:
	SedInputSource(WvCorpusIterator& corpus, WordVectors& wv, int batchSize, int maxSeqLength, int maxNumBatch,
			bool doForward, ix_type startIx, bool dictGpu1);
	virtual ~SedInputSource();
	void toFirstBatch();
	bool hasNextBatchSet();
	void toNextBatchSet();
	bool hasNext();
	void nextBatch(int batchNum = 0, vector<string>* tokensRet = NULL);
	void next(vector<string>* tokensRet = NULL);
	void computeMatchMasks(int batchNum, dtype2** outputs);
	dtype2** matchMasks(int batchNum);
	void reset();
	dtype2** getRevInputs();
	dtype2** getFwdInputs();
	unsigned int** getDInputLengths();
	unsigned int* getHInputLengths();
	dtype2** getTargets();
	int getBatchSize();
	WvCorpusIterator& getCorpus();
	ix_type size();
	void get(ix_type selectedIx, string& sentence);
	void inputFor(string* sentences, int count, bool printTokens = false,
			vector<string>* tokensRet = NULL);
	void shuffle();
	void loadIxs(string filename);
	void saveIxs(string filename);
	string getIxFilename();
	void saveReplacerState(const char* stateFilename);
	int maxNumBatch;
	ix_type firstBatchIx;
	unsigned int *h_inputLengths = NULL;

private:
	void initMem();
	void freeMem();
	void copyNextBatch(int batchNum, vector<string>* tokensRet = NULL);
	void copyInputs(int batchNum);
	void fillInputs(int batchNum);
	WvCorpusIterator& corpus;
	WordVectors& wv;
	bool doForward;
	int batchSize;
	int maxSeqLength;
	dtype2 **d_inputs_rev = NULL;
	dtype2 **dd_inputs_rev = NULL;
	dtype2 **d_inputs_fwd = NULL;
	dtype2 **dd_inputs_fwd = NULL;
	dtype2 **d_targets = NULL;
	unsigned int **d_inputLengths = NULL;
	dtype2 **d1_inputs_rev = NULL;
	dtype2 **dd1_inputs_rev = NULL;
	dtype2 **d1_inputs_fwd = NULL;
	dtype2 **dd1_inputs_fwd = NULL;
	dtype2 **d1_targets = NULL;
	unsigned int **d1_inputLengths = NULL;
	int **h_batchIx = NULL;
	int ***d_batchIx = NULL;
	dtype2 ***d_matchMasks = NULL;
	int ***d1_batchIx = NULL;
	dtype2 ***d1_matchMasks = NULL;
	dtype2 *h_matchMask = NULL;
	int numBatchSaved;
	bool dictGpu1;
};

} /* namespace netlib */

#endif /* SEDINPUTSOURCE_H_ */
