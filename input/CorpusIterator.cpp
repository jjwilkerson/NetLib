/**
 * @file
 * @brief Defines CorpusIterator class, a base class for objects that iterate over an input corpus.
 *
 * CorpusIterator class is a base class for objects that iterator over an input corpus. They provide batches
 * of samples for training.
 *
 */

#include "CorpusIterator.h"

#include "../util/FileUtil.h"

typedef boost::mt19937 base_generator_type;

void printStats(const char* label, dtype2* a, unsigned int size);
void printAll(const char* label, dtype2* a, unsigned int size);

namespace netlib {

CorpusIterator::CorpusIterator(string corpusFilename,
		int batchSize, int maxSentenceLength, bool training, string ixFilename,
		ix_type ix)
		: corpusFilename(corpusFilename), batchSize(batchSize),
		maxSentenceLength(maxSentenceLength), training(training), ixFilename(ixFilename) {

	sentenceIxs = NULL;

	if (ixFilename == "") {
		initCorpus();
		this->ix = 0;
		currentBatch = 0;
		if (training) {
			shuffle();
			saveIxs("sentence_ixs_new_epoch0");
		}
	} else {
		initCorpus();
		this->ix = ix;
		currentBatch = floor(ix / batchSize);
	}
}

CorpusIterator::~CorpusIterator() {
	if (corpusFile.is_open()) {
		corpusFile.close();
	}

	delete [] sentenceIxs;
}

void CorpusIterator::initCorpus() {
	if (ixFilename == "") {
		ixFilename = corpusFilename + ".ix";
	}
	loadIxs(ixFilename);

	totalBatches = floor(corpusSize / batchSize);
	corpusFile.open(corpusFilename.c_str(), ios::binary);
}

bool CorpusIterator::hasNext() {
	return currentBatch < totalBatches - 1;
}

void CorpusIterator::shuffle() {
	static base_generator_type generator(static_cast<unsigned int>(time(0)));
	static boost::uniform_int<> uni_dist;
	static boost::variate_generator<base_generator_type&, boost::uniform_int<> > rnd(generator, uni_dist);

	random_shuffle(&sentenceIxs[0], &sentenceIxs[corpusSize], rnd);
}

void CorpusIterator::reset() {
	currentBatch = 0;
	ix = 0;
}

void CorpusIterator::resetTo(ix_type newIx) {
	ix = newIx;
	currentBatch = floor(ix / batchSize);
}

ix_type CorpusIterator::size() {
	return corpusSize;
}

void CorpusIterator::get(ix_type selectedIx, string& sentence) {
	ix_type fs_ix = sentenceIxs[selectedIx];
	corpusFile.seekg(fs_ix, ios::beg);
	getline(corpusFile, sentence);
	boost::trim(sentence);
}

void CorpusIterator::next(int **batchIx, unsigned int* lengths, vector<string>* tokensRet) {
	ix_type remaining = corpusSize - ix - 1;
	assert((unsigned int) batchSize <= remaining);

	string sentences[batchSize]; //move to class member?
	for (unsigned i = 0; i < batchSize; i++) {
		get(ix, sentences[i]);
		ix++;
	}

	currentBatch++;

	inputFor(sentences, batchSize, batchIx, lengths, false, tokensRet);
}

void CorpusIterator::loadIxs(string filename) {
	ifstream ixFile(filename.c_str(), ios::binary);
	if (!ixFile) {
		cerr << "Couldn't open index file " << filename << endl;
		exit(1);
	}

	corpusSize = FileUtil::readInt64(ixFile);
	cout << "length " << corpusSize << endl;

	if (sentenceIxs == NULL) {
		sentenceIxs = new ix_type[corpusSize];
	}

	for (ix_type i = 0; i < corpusSize; i++) {
		sentenceIxs[i] = FileUtil::readInt64(ixFile);
	}

	ixFile.close();

	ixFilename = filename;
}

void CorpusIterator::saveIxs(string filename) {
	ofstream ixFile(filename.c_str(), ios::binary | ios::trunc);
	FileUtil::writeInt64(corpusSize, ixFile);
	for (ix_type i = 0; i < corpusSize; i++) {
		FileUtil::writeInt64(sentenceIxs[i], ixFile);
	}
	ixFile.close();

	ixFilename = filename;
}

bool CorpusIterator::endsWith(const string& str, const string& ending) {
	if (str.length() >= ending.length()) {
		return (0 == str.compare(str.length() - ending.length(), ending.length(), ending));
	} else {
		return false;
	}
}

string CorpusIterator::getIxFilename() {
	return ixFilename;
}

void CorpusIterator::setIxFilename(string filename) {
	ixFilename = filename;
}

unsigned CorpusIterator::getBatchSize() {
	return batchSize;
}

ix_type CorpusIterator::currentIx() {
	return ix;
}

} /* namespace netlib */

void printStats(const char* label, dtype2* a, unsigned int size) {
	cout << endl << label << endl;

	int arraySize = size * sizeof(dtype2);

	dtypeh min = 99999999;
	dtypeh max = -99999999;
	dtypeh sum = 0;
	dtypeh sumAbs = 0;
	int numNonZero = 0;
	for (int i = 0 ; i < size; i++) {
		dtype2 val = a[i];
		if (val < min) {
			min = val;
		}
		if (val > max) {
			max = val;
		}
		sum += val;
		sumAbs += abs(val);
		if (val != 0) {
			numNonZero++;
		}
	}
	cout << "  min: " << min << endl;
	cout << "  max: " << max << endl;
	cout << "  mean: " << (sum/size) << endl;
	cout << "  mean(abs): " << (sumAbs/size) << endl;
	cout << "  numNonZero: " << numNonZero << endl;
	cout << endl;
}

void printAll(const char* label, dtype2* a, unsigned int size) {
	cout << endl << label << ": ";
	for (int i = 0; i < size; i++) {
		dtype2 val = a[i];
		cout << val << " ";
	}
}
