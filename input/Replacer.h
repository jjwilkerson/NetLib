/**
 * @file
 * @brief Declares Replacer class, which replaces a random content (not function) word in a sentence with a unique token.
 *
 *  Replacer class replaces a random content (not function) word in a sentence with a unique token. It uses a list of
 *  function words. It can be used to train a network to fill in the missing content word.
 *
 */

#ifndef REPLACER_H_
#define REPLACER_H_

#include <cstdint>
#include <string>
#include <vector>
#include <set>
#include <fstream>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>

typedef boost::mt19937 base_generator_type;

using namespace std;

namespace netlib {

/**
 * @brief Replaces a random content (not function) word in a sentence with a unique token.
 *
 *  Replaces a random content (not function) word in a sentence with a unique token. It uses a list of
 *  function words from a text file. It can be used to train a network to fill in the missing content word.
 *
 */
class Replacer {
public:
	Replacer(unsigned int seed, string randStateFilename, unsigned int replacePeriod);
	virtual ~Replacer();
	void replace(vector<string>& tokens);
	void saveState(const char* randStateFilename);
	void saveState(ofstream& stateFile);
	void loadState(ifstream& stateFile);
private:
	void loadFunctionWords();
	bool isFunctionWord(string word);
	void initRand(unsigned int seed);
	void loadState(string randStateFilename);
	string functionWordsFilename;
	set<string> functionWords;
	unsigned int replacePeriod = 200;
	uint64_t counter = 0;
	base_generator_type* replacer_generator = NULL;
	boost::uniform_real<>* replacer_uni_dist;
	boost::variate_generator<base_generator_type&, boost::uniform_real<> >* replacer_uniRnd;
};

} /* namespace netlib */

#endif /* REPLACER_H_ */
