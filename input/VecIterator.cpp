/**
 * @file
 * @brief Defines VecIterator, a CorpusIterator that iterates over a set of vectors.
 *
 */

#include "VecIterator.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "../util/FileUtil.h"

typedef boost::mt19937 base_generator_type;

using namespace std;

namespace netlib {

VecIterator::VecIterator(string corpusFilename, int vecSize, int batchSize, bool training,
		string ixFilename, ix_type ix) : corpusFilename(corpusFilename), vecSize(vecSize),
				batchSize(batchSize), training(training), ixFilename(ixFilename),
				ix(ix) {

	vecIxs = NULL;

	if (ixFilename == "") {
		initCorpus();
		this->ix = 0;
		currentBatch = 0;
		if (training) {
			shuffle();
			saveIxs("sv_ixs_new_epoch0");
		}
	} else {
		initCorpus();
		this->ix = ix;
		currentBatch = floor(ix / batchSize);
	}

	int arraySize = vecSize * batchSize * sizeof(dtype2);
	checkCudaErrors(cudaMallocHost((void **)&h_input, arraySize));
}

VecIterator::~VecIterator() {
	if (corpusFile.is_open()) {
		corpusFile.close();
	}

	delete [] vecIxs;

	checkCudaErrors(cudaFreeHost(h_input));
}

void VecIterator::initCorpus() {
	if (ixFilename == "") {
		ixFilename = corpusFilename + ".ix";
	}
	loadIxs(ixFilename);

	totalBatches = floor(corpusSize / batchSize);
	corpusFile.open(corpusFilename.c_str(), ios::binary);
}

bool VecIterator::hasNext() {
	return currentBatch < totalBatches - 1;
}

void VecIterator::next(dtype2 *input) {
	ix_type remaining = corpusSize - ix;
	assert((unsigned int) batchSize <= remaining);

	inputFor(input, batchSize);
	currentBatch++;
}

void VecIterator::inputFor(dtype2 *input, int count) {
	for (int i = 0; i < count; i++) {
		get(ix, h_input+IDX2(i,0,batchSize));
		ix++;
	}

	int arraySize = vecSize * batchSize * sizeof(dtype2);
	checkCudaErrors(cudaMemcpy(input, h_input, arraySize, cudaMemcpyHostToDevice));
}

void VecIterator::get(ix_type selectedIx, dtype2* d) {
	ix_type fs_ix = vecIxs[selectedIx];
	corpusFile.seekg(fs_ix, ios::beg);

#ifdef DEVICE_HALF
	for (int i = 0; i < vecSize-1; i+=2) {
		int val = FileUtil::readInt(corpusFile);
		dtype2* d2p = (dtype2*) &val;
		d[i*batchSize] = d2p[0];
		d[(i+1)*batchSize] = d2p[1];
	}
#else
	for (int i = 0; i < vecSize; i++) {
		float f = FileUtil::readFloat(corpusFile);
		d[i*batchSize] = float2d(f);
	}
#endif
}

void VecIterator::reset() {
	currentBatch = 0;
	ix = 0;
}

void VecIterator::resetTo(ix_type newIx) {
	ix = newIx;
	currentBatch = floor(ix / batchSize);
}

void VecIterator::shuffle() {
	static base_generator_type generator(static_cast<unsigned int>(time(0)));
	static boost::uniform_int<> uni_dist;
	static boost::variate_generator<base_generator_type&, boost::uniform_int<> > rnd(generator, uni_dist);

	random_shuffle(&vecIxs[0], &vecIxs[corpusSize], rnd);
}

void VecIterator::loadIxs(string filename) {
	ifstream ixFile(filename.c_str(), ios::binary);
	if (!ixFile) {
		cerr << "Couldn't open index file " << filename << endl;
		exit(1);
	}

	corpusSize = FileUtil::readInt64(ixFile);
	cout << "length " << corpusSize << endl;

	if (vecIxs == NULL) {
		vecIxs = new ix_type[corpusSize];
	}

	for (ix_type i = 0; i < corpusSize; i++) {
		vecIxs[i] = FileUtil::readInt64(ixFile);
	}

	ixFile.close();

	ixFilename = filename;
}

void VecIterator::saveIxs(string filename) {
	ofstream ixFile(filename.c_str(), ios::binary | ios::trunc);
	FileUtil::writeInt64(corpusSize, ixFile);
	for (ix_type i = 0; i < corpusSize; i++) {
		FileUtil::writeInt64(vecIxs[i], ixFile);
	}
	ixFile.close();

	ixFilename = filename;
}

ix_type VecIterator::currentIx() {
	return ix;
}

ix_type VecIterator::size() {
	return corpusSize;
}

string VecIterator::getIxFilename() {
	return ixFilename;
}

void VecIterator::setIxFilename(string filename) {
	ixFilename = filename;
}

unsigned VecIterator::getBatchSize() {
	return batchSize;
}

} /* namespace netlib */
