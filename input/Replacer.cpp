/**
 * @file
 * @brief Defines Replacer class, which replaces a random content (not function) word in a sentence with a unique token.
 *
 *  Replacer class replaces a random content (not function) word in a sentence with a unique token. It uses a list of
 *  function words. It can be used to train a network to fill in the missing content word.
 *
 */

#include "../NetLib.h"
#include "Replacer.h"

#include <iostream>
#include <cassert>
#include <string>
#include <boost/algorithm/string.hpp>
#include "../util/FileUtil.h"

using namespace std;

const string BLANK = ":X:";
const int maxTries = 30;

namespace netlib {

Replacer::Replacer(unsigned int seed, string randStateFilename, unsigned int replacePeriod)
		: replacePeriod(replacePeriod) {
	functionWordsFilename = datasetsDir + "/function_words.txt";
	loadFunctionWords();
	if (randStateFilename == "") {
		initRand(seed);
	} else {
		loadState(randStateFilename);
	}
}

Replacer::~Replacer() {
}

void Replacer::loadFunctionWords() {
	ifstream file(functionWordsFilename);

	string word;
	int num = 0;
	while (!file.eof()) {
		getline(file, word, '\n');
		boost::trim(word);
		if (word.length() == 0) {
			continue;
		}

		functionWords.insert(word);
		num++;
	}
	cout << "loaded " << num << " function words" << endl;

}

void Replacer::initRand(unsigned int seed) {
	cout << "Initializing RNG with seed " << seed << endl;
	replacer_generator = new base_generator_type(seed);
	replacer_uni_dist = new boost::uniform_real<>(0, 1);
	replacer_uniRnd = new boost::variate_generator<base_generator_type&, boost::uniform_real<> >(*replacer_generator, *replacer_uni_dist);
}

void Replacer::loadState(string randStateFilename) {
	cout << "loading RNG state from file " << randStateFilename << endl;
	ifstream stateFile(randStateFilename.c_str());
	loadState(stateFile);
	stateFile.close();
}

void Replacer::loadState(ifstream& stateFile) {
	if (replacer_generator != NULL) {
		delete replacer_generator;
	}
	replacer_generator = new base_generator_type(0);
	counter = FileUtil::readInt64(stateFile);
	stateFile >> *replacer_generator;

	replacer_uni_dist = new boost::uniform_real<>(0, 1);
	replacer_uniRnd = new boost::variate_generator<base_generator_type&, boost::uniform_real<> >(*replacer_generator, *replacer_uni_dist);
}

void Replacer::saveState(const char* randStateFilename) {
	ofstream stateFile(randStateFilename, ios::trunc);
	saveState(stateFile);
	stateFile.close();
}

void Replacer::saveState(ofstream& stateFile) {
	FileUtil::writeInt64(counter, stateFile);
	stateFile << *replacer_generator;
}


void Replacer::replace(vector<string> &tokens) {
	counter++;
	if (counter % replacePeriod != 0) {
		return;
	}

	int num = tokens.size();

	bool replaced = false;
	int tries = 0;
	while (!replaced && tries < maxTries) {
		double r = (*replacer_uniRnd)();
		int ix = (int) (r * num);
		assert(r < num);

		string word = tokens[ix];
		if (isFunctionWord(word)) {
			tries++;
			continue;
		}

//		cout << "replacing " << tokens[ix] << " at " << ix << " with " << BLANK << endl;
		tokens[ix] = BLANK;
		replaced = true;
	}
}

bool Replacer::isFunctionWord(string word)
{
	string lowerWord = boost::algorithm::to_lower_copy(word);
	set<string>::iterator it = functionWords.find(lowerWord);
	return (it != functionWords.end());
}

} /* namespace netlib */
