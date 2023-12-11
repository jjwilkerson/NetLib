/**
 * @file
 * @brief Defines the State class, which encapsulates, saves, and loads program state.
 *
 */

#include "State.h"

#include <json/json.h>
#include <fstream>
#include <iostream>
#include "../config/Config.h"

namespace netlib {

State::State() {
}

State::~State() {
}

void State::save(const char* filename) {
	Json::Value root;
	root["epoch"] = epoch;
	root["iter"] = iter;
	root["sentenceIx"] = (Json::UInt64) sentenceIx;
	root["docPos"] = (Json::UInt64) docPos;
	root["ixFilename"] = ixFilename;
	root["weightsFilename"] = weightsFilename;
	root["curandStatesFilename"] = curandStatesFilename;
	root["replacerStateFilename"] = replacerStateFilename;
	root["questionFactoryStateFilename"] = questionFactoryStateFilename;
	root["initDeltaFilename"] = initDeltaFilename;
	root["damping"] = damping;
	root["l2"] = l2;
	root["deltaDecay"] = deltaDecay;
	root["numBatchGrad"] = numBatchGrad;
	root["numBatchG"] = numBatchG;
	root["numBatchError"] = numBatchError;
	root["maxIterCG"] = maxIterCG;
	root["clock"] = (Json::Int) clock;
	root["learningRate"] = learningRate;
	root["lossScaleFac"] = lossScaleFac;
	root["iterNoOverflow"] = iterNoOverflow;

	cout << "saving state:" << endl;
	cout << root << endl;

	ofstream sFile(filename, ios::trunc);
	sFile << root << endl;
	sFile.close();
}

void State::load(const char* filename) {
	Json::Value root;

	ifstream sFile(filename);
	sFile >> root;
	sFile.close();

	cout << "state loaded:" << endl;
	cout << root << endl;

	if (root.isMember("epoch")) {
		epoch = root["epoch"].asInt();
	} else {
		epoch = 0;
	}

	iter = root["iter"].asInt();
	sentenceIx = root["sentenceIx"].asUInt64();
	docPos = root["docPos"].asUInt64();

	if (root.isMember("ixFilename")) {
		ixFilename = root["ixFilename"].asString();
	} else {
		ixFilename = "";
	}

	weightsFilename = root["weightsFilename"].asString();
	curandStatesFilename = root["curandStatesFilename"].asString();
	replacerStateFilename = root["replacerStateFilename"].asString();
	questionFactoryStateFilename = root["questionFactoryStateFilename"].asString();
	initDeltaFilename = root["initDeltaFilename"].asString();
	damping = root["damping"].asFloat();
	deltaDecay = root["deltaDecay"].asFloat();

	if (root.isMember("l2")) {
		l2 = root["l2"].asFloat();
	} else {
		l2 = -1;
	}

	numBatchGrad = root["numBatchGrad"].asInt();
	numBatchG = root["numBatchG"].asInt();
	numBatchError = root["numBatchError"].asInt();
	clock = root["clock"].asUInt();

	if (root.isMember("maxIterCG")) {
		maxIterCG = root["maxIterCG"].asInt();
	} else {
		maxIterCG = 0;
	}

	if (root.isMember("learningRate")) {
		learningRate = root["learningRate"].asFloat();
	} else {
		learningRate = 0.0f;
	}

	if (root.isMember("lossScaleFac")) {
		lossScaleFac = root["lossScaleFac"].asFloat();
	} else {
		lossScaleFac = 1.0f;
	}

	if (root.isMember("iterNoOverflow")) {
		iterNoOverflow = root["iterNoOverflow"].asInt();
	} else {
		iterNoOverflow = 0;
	}
}

State* State::randomize() {
	State* state = new State();

	state->clock = clock;
	state->epoch = epoch;
	state->iter = iter;
	state->sentenceIx = sentenceIx;
	state->ixFilename = ixFilename;
	state->weightsFilename = weightsFilename;
	state->curandStatesFilename = curandStatesFilename;
	state->initDeltaFilename = initDeltaFilename;

	state->damping = damping;

	state->deltaDecay = Config::rndUniform(0.8, (dtype2) 0.99);

	if (Config::rndBool(0.25)) {
		state->l2 = 0.0;
	} else {
		state->l2 = l2;
	}

	state->numBatchGrad = Config::rndUniform(numBatchGrad, 280);
	state->numBatchG = Config::rndUniform(numBatchG, 8);
	state->numBatchError = Config::rndUniform(numBatchError, 80);

	state->maxIterCG = Config::rndUniform(10, 75);

	return state;
}

} /* namespace netlib */
