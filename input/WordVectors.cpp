/**
 * @file
 * @brief Defines the WordVectors class, which wraps the Word2Vec class and adds functionality useful for sentence encoding,
 * such as special vectors.
 *
 */

#include "WordVectors.h"

#include "../Network.h"
#include <ctime>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include "../util/FileUtil.h"

const float svSimThreshold = 0.95;
const bool svZero = false;

const string GO = "GO";
const string UNK = "UNK";
const string EOS = "EOS";
const string AND = "and";
const string TO = "to";
const string A = "a";
const string OF = "of";
const string COMMA = ",";
const string QUESTION_MARK = "?";
const string QUOTE = "'";
const string DQUOTE = "\"";
const string COLON = ":";
const string HYPHEN = "-";
const string DASH = "--";
const string SEMICOLON = ";";
const string ELLIPSIS = "...";
const string LPAREN = "(";
const string RPAREN = ")";
const string EXCLAM = "!";
const string PERIOD = ".";
const string SLASH = "/";
const string BLANK = ":X:";

const string svLabels[] = {UNK, EOS, AND, TO, A, OF, COMMA, QUESTION_MARK, QUOTE, DQUOTE, COLON, HYPHEN, DASH,
		SEMICOLON, ELLIPSIS, RPAREN, LPAREN, EXCLAM, PERIOD, SLASH, BLANK};

typedef boost::mt19937 base_generator_type;

namespace netlib {

WordVectors::WordVectors(cublasHandle_t& handle, bool dictGpu1, int batchSize) : w2v(handle, dictGpu1, batchSize) {
	modelFilename = datasetsDir + "/GoogleNews-vectors-negative300.bin";
	wordsFilename = datasetsDir + "/news.2007-2017.en.proc3_sym9.words";
	specialVectorsFilename = datasetsDir + "/special_vectors.bin";
	specialVectorsNewFilename = datasetsDir + "special_vectors_new.bin";
	statsFilename = datasetsDir + "/news.2007-2017.en.proc3_sym9_max59.stats";
	loadStats();
	initSpecialVectors();
	w2v.init(modelFilename, wordsFilename, false, specialVectors.size(), wvMean, wvStd);
}

WordVectors::~WordVectors() {
	map<string,float*>::iterator it;

	for (it = specialVectors.begin(); it != specialVectors.end(); it++) {
		float* sv = it->second;
		delete [] sv;
	}

	if (wvMean != NULL) {
		free(wvMean);
	}
	if (wvStd != NULL) {
		free(wvStd);
	}
}

dtype2* WordVectors::lookup(string label, int* ix) {
	dtype2* vec = w2v.lookup(label, ix);
	if (vec != NULL) {
		return vec;
	}

	return lookup(UNK, ix);
}

void WordVectors::freeVector(dtype2* vec) {
	w2v.freeVector(vec);
}

int WordVectors::indexOf(string label) {
	int ix = w2v.indexOf(label);
	if (ix == -1) {
		return w2v.indexOf(UNK);
	} else {
		return ix;
	}
}

pair<string, dtypeh> WordVectors::nearest(dtype2* vec) {
	return w2v.nearest(vec);
}

#ifndef DEBUG
string* WordVectors::nearestBatch(dtype2* output, int batchSize) {
	return w2v.nearestBatch(output, batchSize);
}
#endif

//#ifndef DEBUG
//string* WordVectors::nearestBatchLS(dtype2* output, int batchSize, unsigned int* h_inputLengths, int s) {
//	return w2v.nearestBatchLS(output, batchSize, h_inputLengths, s);
//}
//#endif

void WordVectors::freeTemp() {
	w2v.freeTemp();
}

int* WordVectors::nearestBatchIx(dtype2* output) {
	return w2v.nearestBatchIx(output);
}

#ifndef DEBUG
void WordVectors::nearestBatchMask(dtype2* output, int* ixTarget, unsigned int *h_inputLengths,
		int s, dtype2* mask, unsigned int *d_inputLengths) {
	return w2v.nearestBatchMask(output, ixTarget, h_inputLengths, s, mask, d_inputLengths);
}
#endif

dtype2* WordVectors::getDictGpu() {
	return w2v.getDictGpu();
}

int WordVectors::getNumWords() {
	return w2v.getNumWords();
}

void WordVectors::initSpecialVectors() {
	ifstream svFile(specialVectorsFilename, ios::binary);
	if (svFile) {
		loadSpecialVectors(svFile);

		bool newSv = false;
		map<string,float*>::iterator pos;
		for (string label : svLabels) {
			pos = specialVectors.find(label);
			if (pos == specialVectors.end()) {
				cout << "adding " << label << endl;
				specialVectors.insert(pair<string,float*>(label, randVec()));
				newSv = true;
			}
		}

		if (newSv) {
			saveSpecialVectorsNew();
		}
	} else {
		for (string label : svLabels) {
			specialVectors.insert(pair<string,float*>(label, randVec()));
		}
//		specialVectors.insert(pair<string,float*>(UNK, randVec()));
//		specialVectors.insert(pair<string,float*>(COMMA, randVec()));
//		specialVectors.insert(pair<string,float*>(EOS, randVec()));
//		specialVectors.insert(pair<string,float*>(QUESTION_MARK, randVec()));
//		specialVectors.insert(pair<string,float*>(AND, randVec()));
//		specialVectors.insert(pair<string,float*>(TO, randVec()));
//		specialVectors.insert(pair<string,float*>(A, randVec()));
//		specialVectors.insert(pair<string,float*>(OF, randVec()));
//		specialVectors.insert(pair<string,float*>(HYPHEN, randVec()));
//		specialVectors.insert(pair<string,float*>(COLON, randVec()));
//		specialVectors.insert(pair<string,float*>(QUOTE, randVec()));
//		specialVectors.insert(pair<string,float*>(DQUOTE, randVec()));
//		specialVectors.insert(pair<string,float*>(DASH, randVec()));

		saveSpecialVectors();
	}

	map<string,float*>::iterator it;
	for (it = specialVectors.begin(); it != specialVectors.end(); it++) {
		float* vector = it->second;
#ifndef DEBUG
		if (vecUnitNorm) w2v.normalize(vector);
		w2v.whiten(vector, wvMean, wvStd);
#endif
	}

	UNKNOWN_VECTOR = specialVectors.find(UNK)->second;
	EOS_VECTOR = specialVectors.find(EOS)->second;
	COMMA_VECTOR = specialVectors.find(COMMA)->second;

	cout << specialVectors.size() << " special vectors" << endl;
}

void WordVectors::saveSpecialVectors() {
	ofstream svFile(specialVectorsFilename, ios::binary | ios::trunc);

	map<string,float*>::iterator it;

	for (it = specialVectors.begin(); it != specialVectors.end(); it++) {
		string label = it->first;
		float* vector = it->second;
		saveVec(label, vector, svFile);
	}

	svFile.close();
}

void WordVectors::saveSpecialVectorsNew() {
	ofstream svFile(specialVectorsNewFilename, ios::binary | ios::trunc);

	map<string,float*>::iterator it;

	for (it = specialVectors.begin(); it != specialVectors.end(); it++) {
		string label = it->first;
		float* vector = it->second;
		saveVec(label, vector, svFile);
	}

	svFile.close();
}

void WordVectors::loadSpecialVectors(ifstream &svFile) {
	cout << endl << "loadSpecialVectors" << endl;
	if (svZero) cout << endl << "setting all to zero" << endl;

	while (svFile) {
		string label;
		float* vector = new float[WV_LENGTH];

		label = FileUtil::readString(svFile);
		if (!svFile) {
			break;
		}

		for (int i = 0; i < WV_LENGTH; i++) {
			float f = FileUtil::readFloat(svFile);
			if (svZero) f = 0.0f;
			vector[i] = f;
		}

		if (label == GO) {
			continue;
		}

		cout << label << endl;
		specialVectors.insert(pair<string,float*>(label, vector));
	}

	svFile.close();
}

void WordVectors::loadStats() {
	ifstream statsFile(statsFilename, ios::binary);
	if (statsFile) {
		cout << "loading word stats" << endl;

		wvMean = (dtypeh*) malloc(WV_LENGTH*sizeof(dtypeh));
		wvStd = (dtypeh*) malloc(WV_LENGTH*sizeof(dtypeh));

		for (int j = 0; j < WV_LENGTH; j++)
		{
			float fValue = FileUtil::readFloat(statsFile);
			wvMean[j] = float2h(fValue);
		}
		for (int j = 0; j < WV_LENGTH; j++)
		{
			float fValue = FileUtil::readFloat(statsFile);
			wvStd[j] = float2h(fValue);
		}

		statsFile.close();

//		cout << "mean" << endl;
//		Network::printAll(wvMean, WV_LENGTH);

//		cout << "std" << endl;
//		Network::printAll(wvStd, WV_LENGTH);
	}
}

float* WordVectors::randVec() {
	static base_generator_type generator(static_cast<unsigned int>(time(0)));
//	static boost::uniform_real<> uni_dist(-1,1);
//	static boost::variate_generator<base_generator_type&, boost::uniform_real<> > rnd(generator, uni_dist);
	static boost::normal_distribution<> normal_dist(0, 1);
	static boost::variate_generator<base_generator_type&, boost::normal_distribution<> > rnd(generator, normal_dist);

	bool haveStats = wvMean != NULL && wvStd != NULL;

	float *vec = new float[WV_LENGTH];
	for (int i = 0; i < WV_LENGTH; i++) {
		vec[i] = (float) rnd();
		if (haveStats) {
			vec[i] *= wvStd[i];
			vec[i] += wvMean[i];
		}
	}

//	w2v.normalize(vec);
	return vec;
}

void WordVectors::saveVec(string label, float* vector, ofstream &svFile) {
	FileUtil::writeString(label, svFile);
	for (int i = 0; i < WV_LENGTH; i++) {
		FileUtil::writeFloat(vector[i], svFile);
	}
}

dtypeh WordVectors::cosineSim(dtype2* vec1, dtype2* vec2) {
	return w2v.cosineSim(vec1, vec2);
}

void WordVectors::initDictGpu(cublasHandle_t& handle) {
	w2v.initDictGpu(handle, &specialVectors);
}

void WordVectors::deviceNormalize(dtype2* vec) {
	w2v.deviceNormalize(vec);
}

} /* namespace netlib */
