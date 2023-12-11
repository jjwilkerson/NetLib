/**
 * @file
 * @brief Declares Config class, which saves, loads and encapsulates configuration parameters.
 */

#ifndef CONFIG_H_
#define CONFIG_H_

#include "../NetLib.h"
#include <string>

using namespace std;

namespace netlib {

class State;

/**
 * @brief Saves, loads and encapsulates configuration parameters.
 */
class Config {
public:
	Config();
	virtual ~Config();
	void save(const char* filename);
	void load(const char* filename);
	static Config* random();
	void updateFromState(State* state);

	int batchSize;

	int numLayers;
	int sentence_vector_width;
	int encoder_rnn_width;
	int decoder_rnn_width;
	int embedding_width;

	int encoder_mod;
	int decoder_mod;

	string activation;

	bool useTwoDevices;
	bool dictGpu1;

	unsigned int seed;

	int numRec;

	dtypeh l2;
	dtypeh structDampCoef;

	dtypeh ffWeightCoef;
	dtypeh recWeightCoef;

	dtypeh stdInitFF;
	dtypeh stdInitRec;

	string weightsFilename;
	string curandStatesFilename; //not loaded/saved
	string replacerStateFilename; //not loaded/saved

	bool pretrain;
	bool train;

	int maxIterCG;
	dtypeh initDamp;
	int numBatchGrad;
	int numBatchG;
	int numBatchError;

	//pretrain
	int k;
	bool persistent;

	//train
	dtypeh *dropouts;

	dtypeh deltaDecayInitial;
	dtypeh deltaDecayRatio;
	dtypeh deltaDecayMax;

	static const int LAST_ITER = 1;
	static const int CHOSEN_ITER = 2;
	int deltaIter;

	dtypeh dampThreshLower;
	dtypeh dampThreshUpper;

	float pDropMatch;

	float learningRate;
	float lrDecay;

	static const string HF;
	static const string ADAM;
	string optimizer;

	static int rndUniform(int i, int j);
	static dtypeh rndUniform(dtypeh d, dtypeh e);
	static dtypeh rndExp(dtypeh d, dtypeh e);
	static bool rndBool(double prob);

	int testPeriod;
	int testMatchingPeriod;
	int demoPeriod;
	int savePeriod;
	int tempSavePeriod;
	int printPeriod;
	int testMaxBatches = 10000;
	int testMatchingMaxBatches = 10000;

	dtypeh gradClipThresh;
	dtypeh gradClipMax;

	bool lossScaleEnabled = false;
	dtypeh lossScaleFac = 1.0;
	dtypeh lossScaleUpFac = 2.0;
	dtypeh lossScaleDnFac = 0.5;
	int lossScalePeriod = 1000;
	int iterNoOverflow = 0;

	float adamP1 = 0.9;
	float adamP2 = 0.999;

	int maxSeqLength = 80;
	int wvLength = 300;

	int maxDocumentSize = 100;
	int maxNumQuestions = 10;
	int documentVectorWidth = 20000;
	int compressedSvWidth = 3000;
};

} /* namespace netlib */

#endif /* CONFIG_H_ */
