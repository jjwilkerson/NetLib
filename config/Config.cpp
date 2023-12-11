/**
 * @file
 * @brief Defines Config class, which saves, loads and encapsulates configuration parameters.
 */

#include "Config.h"

#include <json/json.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include "../state/State.h"

typedef boost::mt19937 base_generator_type;

base_generator_type generator(static_cast<unsigned int>(std::time(0)));
boost::uniform_real<> uni_dist(0, 1);
boost::variate_generator<base_generator_type&, boost::uniform_real<> > uniRnd(generator, uni_dist);

using namespace std;

namespace netlib {

const string Config::HF = "hf";
const string Config::ADAM = "adam";

Config::Config() {
	testPeriod = 0;
	testMatchingPeriod = 0;
	demoPeriod = 0;
	savePeriod = 0;
	tempSavePeriod = 0;
	printPeriod = 0;
	dictGpu1 = false;
}

Config::~Config() {
}

void Config::save(const char* filename) {
	Json::Value root;
	root["batchSize"] = batchSize;
	root["numLayers"] = numLayers;
	root["sentence_vector_width"] = sentence_vector_width;
	root["encoder_rnn_width"] = encoder_rnn_width;
	root["decoder_rnn_width"] = decoder_rnn_width;
	root["embedding_width"] = embedding_width;
	root["activation"] = activation;
	root["useTwoDevices"] = useTwoDevices;
	root["dictGpu1"] = dictGpu1;
	root["seed"] = seed;
	root["numRec"] = numRec;
	root["l2"] = l2;
	root["structDampCoef"] = structDampCoef;
	root["ffWeightCoef"] = ffWeightCoef;
	root["recWeightCoef"] = recWeightCoef;
	root["stdInitFF"] = stdInitFF;
	root["stdInitRec"] = stdInitRec;
	root["weightsFilename"] = weightsFilename;
	root["pretrain"] = pretrain;
	root["train"] = train;
	root["maxIterCG"] = maxIterCG;
	root["initDamp"] = initDamp;
	root["numBatchGrad"] = numBatchGrad;
	root["numBatchG"] = numBatchG;
	root["numBatchError"] = numBatchError;
	root["k"] = k;
	root["persistent"] = persistent;

	for (int i = 0; i < numLayers; i++) {
		root["dropouts"][i] = dropouts[i];
	}

	root["deltaDecay"]["initial"] = deltaDecayInitial;
	root["deltaDecay"]["ratio"] = deltaDecayRatio;
	root["deltaDecay"]["max"] = deltaDecayMax;
	root["deltaIter"] = deltaIter;

	root["dampThreshLower"] = dampThreshLower;
	root["dampThreshUpper"] = dampThreshUpper;

	root["pDropMatch"] = pDropMatch;

	root["learningRate"] = learningRate;
	root["lrDecay"] = lrDecay;

	root["optimizer"] = optimizer;

	root["trainer"]["testPeriod"] = testPeriod;
	root["trainer"]["testMatchingPeriod"] = testMatchingPeriod;
	root["trainer"]["demoPeriod"] = demoPeriod;
	root["trainer"]["savePeriod"] = savePeriod;
	root["trainer"]["tempSavePeriod"] = tempSavePeriod;
	root["trainer"]["printPeriod"] = printPeriod;
	root["trainer"]["testMaxBatches"] = testMaxBatches;
	root["trainer"]["testMatchingMaxBatches"] = testMatchingMaxBatches;

	root["gradClipThresh"] = gradClipThresh;
	root["gradClipMax"] = gradClipMax;

	root["encoder_mod"] = encoder_mod;
	root["decoder_mod"] = decoder_mod;

	root["lossScale"]["enabled"] = lossScaleEnabled;
	root["lossScale"]["fac"] = lossScaleFac;
	root["lossScale"]["upFac"] = lossScaleUpFac;
	root["lossScale"]["dnFac"] = lossScaleDnFac;
	root["lossScale"]["period"] = lossScalePeriod;

	root["adam"]["p1"] = adamP1;
	root["adam"]["p2"] = adamP2;

	root["maxDocumentSize"] = maxDocumentSize;
	root["maxNumQuestions"] = maxNumQuestions;
	root["documentVectorWidth"] = documentVectorWidth;
	root["compressedSvWidth"] = compressedSvWidth;

	cout << "saving config:" << endl;
	cout << root << endl;

	ofstream cFile(filename, ios::trunc);
	cFile << root << endl;
	cFile.close();
}

void Config::load(const char* filename) {
	Json::Value root;

	ifstream cFile(filename);
	cFile >> root;
	cFile.close();

	cout << "config loaded:" << endl;
	cout << root << endl;

	batchSize = root["batchSize"].asInt();
	numLayers = root["numLayers"].asInt();
	sentence_vector_width = root["sentence_vector_width"].asInt();
	encoder_rnn_width = root["encoder_rnn_width"].asInt();
	decoder_rnn_width = root["decoder_rnn_width"].asInt();
	embedding_width = root["embedding_width"].asInt();

	if (root.isMember("activation")) {
		activation = root["activation"].asString();
	} else {
		activation = "tanh";
	}

	useTwoDevices = root["useTwoDevices"].asBool();

	if (root.isMember("dictGpu1")) {
		dictGpu1 = root["dictGpu1"].asBool();
	} else {
		dictGpu1 = false;
	}

	seed = root["seed"].asUInt();
	numRec = root["numRec"].asInt();
	l2 = root["l2"].asFloat();
	structDampCoef = root["structDampCoef"].asFloat();
	ffWeightCoef = root["ffWeightCoef"].asFloat();
	recWeightCoef = root["recWeightCoef"].asFloat();
	stdInitFF = root["stdInitFF"].asFloat();
	stdInitRec = root["stdInitRec"].asFloat();
	weightsFilename = root["weightsFilename"].asString();
	pretrain = root["pretrain"].asBool();
	train = root["train"].asBool();
	maxIterCG = root["maxIterCG"].asInt();
	initDamp = root["initDamp"].asFloat();
	numBatchGrad = root["numBatchGrad"].asInt();
	numBatchG = root["numBatchG"].asInt();
	numBatchError = root["numBatchError"].asInt();
	k = root["k"].asInt();
	persistent = root["persistent"].asBool();

	dropouts = new dtypeh[numLayers];
	for (int i = 0; i < numLayers; i++) {
		dropouts[i] = root["dropouts"][i].asFloat();
	}

	deltaDecayInitial = root["deltaDecay"]["initial"].asFloat();
	deltaDecayRatio = root["deltaDecay"]["ratio"].asFloat();
	deltaDecayMax = root["deltaDecay"]["max"].asFloat();
	deltaIter = root["deltaIter"].asInt();

	if (root.isMember("dampThreshLower")) {
		dampThreshLower = root["dampThreshLower"].asFloat();
	} else {
		dampThreshLower = 0.1f;
	}

	if (root.isMember("dampThreshUpper")) {
		dampThreshUpper = root["dampThreshUpper"].asFloat();
	} else {
		dampThreshUpper = 0.4f;
	}

	if (root.isMember("pDropMatch")) {
		pDropMatch = root["pDropMatch"].asFloat();
	} else {
		pDropMatch = 0.5f;
	}

	if (root.isMember("optimizer")) {
		optimizer = root["optimizer"].asString();
	} else {
		optimizer = HF;
	}

	if (root.isMember("learningRate")) {
		learningRate = root["learningRate"].asFloat();
	} else {
		learningRate = 0.0f;
	}

	if (root.isMember("lrDecay")) {
		lrDecay = root["lrDecay"].asFloat();
	} else {
		lrDecay = 1.0f;
	}

	if (root.isMember("trainer")) {
		testPeriod = root["trainer"]["testPeriod"].asInt();
		testMatchingPeriod = root["trainer"]["testMatchingPeriod"].asInt();
		demoPeriod = root["trainer"]["demoPeriod"].asInt();
		savePeriod = root["trainer"]["savePeriod"].asInt();
		tempSavePeriod = root["trainer"]["tempSavePeriod"].asInt();
		printPeriod = root["trainer"]["printPeriod"].asInt();
		if (root["trainer"].isMember("testMaxBatches")) {
			testMaxBatches = root["trainer"]["testMaxBatches"].asInt();
		}
		if (root["trainer"].isMember("testMatchingMaxBatches")) {
			testMatchingMaxBatches = root["trainer"]["testMatchingMaxBatches"].asInt();
		}
	}

	if (root.isMember("gradClipThresh")) {
		gradClipThresh = root["gradClipThresh"].asFloat();
	} else {
		gradClipThresh = 0.0f;
	}

	if (root.isMember("gradClipMax")) {
		gradClipMax = root["gradClipMax"].asFloat();
	} else {
		gradClipMax = 0.0f;
	}

	encoder_mod = root["encoder_mod"].asInt();
	decoder_mod = root["decoder_mod"].asInt();

	if (root.isMember("lossScale")) {
		lossScaleEnabled = root["lossScale"]["enabled"].asBool();
		lossScaleFac = root["lossScale"]["fac"].asFloat();
		lossScaleUpFac = root["lossScale"]["upFac"].asFloat();
		lossScaleDnFac = root["lossScale"]["dnFac"].asFloat();
		lossScalePeriod = root["lossScale"]["period"].asInt();
	}

	if (root.isMember("adam")) {
		adamP1 = root["adam"]["p1"].asFloat();
		adamP2 = root["adam"]["p2"].asFloat();
	}

	if (root.isMember("maxDocumentSize")) {
		maxDocumentSize = root["maxDocumentSize"].asInt();
	}

	if (root.isMember("maxNumQuestions")) {
		maxNumQuestions = root["maxNumQuestions"].asInt();
	}

	if (root.isMember("documentVectorWidth")) {
		documentVectorWidth = root["documentVectorWidth"].asInt();
	}

	if (root.isMember("compressedSvWidth")) {
		compressedSvWidth = root["compressedSvWidth"].asInt();
	}
}

Config* Config::random() {
	Config* config = new Config();

	config->numLayers = 7;
	config->sentence_vector_width = 8000;
	config->encoder_rnn_width = 8000;
	config->decoder_rnn_width = 8000;
	config->embedding_width = 300;

	config->optimizer = ADAM;
	config->activation = "tanh";

	config->useTwoDevices = false;

	config->seed = 1234;

	config->numRec = 2;

	config->l2 = rndExp(1e-8, 1e-4);

	config->ffWeightCoef = rndExp(0.005, 0.1);
	config->recWeightCoef = rndExp(0.001, 0.05);

	config->weightsFilename = "";

	config->pretrain = false;
	config->train = true;

	config->structDampCoef = 0;
	config->maxIterCG = 0;
	config->initDamp = 0;
	config->deltaDecayInitial = 0;
	config->deltaDecayRatio = 0;
	config->deltaDecayMax = 0;
	config->deltaIter = Config::LAST_ITER;
	config->dampThreshLower = 0;
	config->dampThreshUpper = 0;
	config->stdInitFF = 0;
	config->stdInitRec = 0;

	config->numBatchGrad = 1;
	config->numBatchG = 0;
	config->numBatchError = 1;

	//pretrain
	config->k = 1;
	config->persistent = true;

	//train

	config->dropouts = new dtypeh[7] {0, 0, 0.5, 0, 0, 0.5, 0};

	config->gradClipThresh = 0;
	config->gradClipMax = 0;

	config->lossScaleEnabled = true;
	config->lossScaleFac = 50.0;
	config->lossScaleUpFac = 2.0;
	config->lossScaleDnFac = 0.5;
	config->lossScalePeriod = 2000;

	config->encoder_mod = 1;
	config->decoder_mod = 1;

	config->learningRate = rndExp(1e-6, 1e-4);

	float lrDecayInv = rndExp(1e-7, 1e-5);
	config->lrDecay = 1.0 - lrDecayInv;

	double r = uniRnd();
	if (r < 0.2) {
		config->adamP1 = 0.8;
	} else if (r < 0.4) {
		config->adamP1 = 0.85;
	} else if (r < 0.6) {
		config->adamP1 = 0.9;
	} else if (r < 0.8) {
		config->adamP1 = 0.95;
	} else {
		config->adamP1 = 0.99;
	}

	r = uniRnd();
	if (r < 0.2) {
		config->adamP2 = 0.8;
	} else if (r < 0.4) {
		config->adamP2 = 0.9;
	} else if (r < 0.6) {
		config->adamP2 = 0.99;
	} else if (r < 0.8) {
		config->adamP2 = 0.995;
	} else {
		config->adamP2 = 0.999;
	}

	config->pDropMatch = 1;

	config->testPeriod = 2000;
	config->testMatchingPeriod = 10000;
	config->demoPeriod = 500;
	config->savePeriod = 10000;
	config->tempSavePeriod = 200;
	config->printPeriod = 20;
	config->testMatchingMaxBatches = 1000;

	return config;
}

void Config::updateFromState(State* state) {
	if (state == NULL) {
		cout << "state is null" << endl;
		return;
	}

	weightsFilename = state->weightsFilename;
	curandStatesFilename = state->curandStatesFilename;

	initDamp = state->damping;
	deltaDecayInitial = state->deltaDecay;

	if (state->l2 >= 0) {
		l2 = state->l2;
	}

	numBatchGrad = state->numBatchGrad;
	numBatchG = state->numBatchG;
	numBatchError = state->numBatchError;

	if (state->maxIterCG > 0) {
		maxIterCG = state->maxIterCG;
	}

	if (state->learningRate > 0) {
		learningRate = state->learningRate;
	}

	lossScaleFac = state->lossScaleFac;
	iterNoOverflow = state->iterNoOverflow;
}

int Config::rndUniform(int i, int j) {
	int interval = j - i;
	double r = uniRnd();
	int del = floor(r * interval);
	return i + del;
}

dtypeh Config::rndUniform(dtypeh d, dtypeh e) {
	dtypeh interval = e - d;
	double r = uniRnd();
	dtypeh del = (dtypeh) (r * interval);
	return d + del;
}

dtypeh Config::rndExp(dtypeh d, dtypeh e) {
	dtypeh l1 = (dtypeh) log(d);
	dtypeh l2 = (dtypeh) log(e);
	dtypeh intermed = rndUniform(l1, l2);
	return (dtypeh) exp(intermed);
}

bool Config::rndBool(double prob) {
	double r = uniRnd();
	return r < prob;
}

} /* namespace netlib */
