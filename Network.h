/**
 * @file
 * @brief Declares Network class, which represents an entire network.
 *
 */

#ifndef NETWORK_H_
#define NETWORK_H_

//#include "Layer.h"
//#include "InputLayer.h"
//#include "OutputLayer.h"
//#include <cuda_runtime.h>
//#include <helper_cuda.h>
#include "NetLib.h"
#include <vector>
#include <string>
#include "cublas_v2.h"

using namespace std;

namespace netlib {

class InputSource;
class Layer;
class InputLayer;
class LossFunction;
class WeightInit;
class Config;

/**
 * @brief Represents an entire network.
 *
 */
class Network {
public:
	Network(cublasHandle_t& handle, int batchSize, int maxSeqLength, Config config, LossFunction& lossFunction,
			WeightInit& ffWeightInit, WeightInit& recWeightInit, bool training);
	virtual ~Network();
//	void addInputSource(InputSource* source);
	void setTrainSource(InputSource* source);
	void addInput(InputLayer* layer);
	void setOutput(Layer* layer);
	void addHidden(Layer* layer);
	dtype2 error(dtype2* params = NULL, bool stochasticDropout = true, bool matchDrop = true, InputSource* source = NULL);
	void forward(int batchNum, dtype2* params = NULL, bool deriv = false, bool stochasticDropout = true,
			InputSource* source = NULL, bool computeMatchMasks = true);
	void forwardNext(vector<string>* tokensRet = NULL);
	void iterInit();
	void copyParams();
	dtype1* getMasterParams();
	dtype2* calcGrad();
	dtype2* calcG(dtype2* v, dtypeh damping = 0, dtype2* out = NULL);
	void Rforward(dtype2* v, int batchNum);
	void Rback(dtype2* Gv, int batchNum);
	void checkGrad(dtype2* gradCalc);
	bool hasParam(int i);
	bool isWeight(int i);
	bool isGradOverflow();
	static bool checkClose(dtype2* a, dtype2* b, unsigned int size, float rtol = 1e-3, float atol = 1e-7);
	void checkG(dtype2* GvCalc, dtype2* v, dtypeh damping = 0);
	void checkJ(int s, dtype2** J);
	dtype2** getRevInputs();
	dtype2** getFwdInputs();
	unsigned int** getDInputLengths();
	unsigned int* getHInputLengths();
	dtype2** getTargets();
	Layer* getOutputLayer();
	dtype2** matchMasks(int batchNum);
	bool isLossScaleEnabled();
	dtypeh getLossScaleFac();
	int getIterNoOverflow();
	void init();
	void saveWeights(const char* weightsFilename);
	void saveCurandStates(const char* filename);
	void setParamOffset(int offset);
	void clearLayers();
	static void cpuSumRows(dtype2* d_mat, dtype2* d_vec, int rows, int cols);
	static void printStatsGpu(string label, half* a, unsigned int size);
	static void printStatsGpu(string label, float* a, unsigned int size);
	static void printStatsGpu(string label, double* a, unsigned int size);
	static void printStatsGpu(string label, unsigned int* a, unsigned int size);
	static void printAll(dtype2* a, unsigned int size);
	static void printAll(unsigned int* a, unsigned int size);
	static void printAll(int* a, unsigned int size);
	static void printAllGpu(dtype2* a, unsigned int size);
	static void printAllGpu(unsigned int* a, unsigned int size);
	static void printAllGpu(int* a, unsigned int size);
	static void printRowGpu(const char* label, dtype2* a, unsigned int rows, unsigned int cols, unsigned int row);
	int batchSize;
	int maxSeqLength;
	int nParams = 0;
	LossFunction& lossFunction;
	dtype2 *W = NULL;
	dtype1 *W1 = NULL;
	dtype2 *h_W = NULL;
	dtype1 *h_W1 = NULL;
	int numBatchError;
	int numBatchGrad;
	int numBatchG;
	dtype2 l2;
	bool optHF;

private:
	void calcOffsets();
	void initMem();
	void freeMem();
	void initWeights();
	void loadWeights(std::string weightsFilename);
	void initCurand();
	void loadCurand(std::string filename);
	void toFirstBatch();
	void toNextBatchSet();
	void nextBatch(int batchNum = 0);
	cublasHandle_t& handle;
	unsigned int rngSeed;
	std::string weightsFilename;
	std::string curandStatesFilename;
	WeightInit& ffWeightInit;
	WeightInit& recWeightInit;
	bool matchDrop;
	bool training;
//	std::vector<InputSource*> inputSources;
	InputSource* trainSource = NULL;
	InputSource* currSource = NULL;
	std::vector<InputLayer*> inputLayers;
	Layer* outputLayer = NULL;
//	std::vector<Layer*> hiddenLayers;
	std::vector<Layer*> layers;
	dtype2 **losses_d = NULL;
	bool lossScaleEnabled = false;
	dtypeh lossScaleFac = 1.0;
	dtypeh lossScaleUpFac = 2.0;
	dtypeh lossScaleDnFac = 0.5;
	int lossScalePeriod = 1000;
	int iterNoOverflow = 0;
	bool gradOverflow = false;
	int paramOffset = 0;
};

} /* namespace netlib */

#endif /* NETWORK_H_ */
