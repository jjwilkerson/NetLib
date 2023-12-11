/**
 * @file
 * @brief Declares Corrector class, which corrects common spelling errors in an input corpus.
 *
 * Corrector class corrects common spelling errors in an input corpus. It loads corrections from a text file.
 *
 */

#ifndef CORRECTOR_H_
#define CORRECTOR_H_

#include <string>
#include <vector>
#include <map>

using namespace std;

namespace netlib {

/**
 * @brief Corrects common spelling errors in an input corpus.
 *
 * Corrects common spelling errors in an input corpus. Loads corrections from a text file.
 *
 */
class Corrector {
public:
	Corrector();
	virtual ~Corrector();
	void correct(vector<string>& tokens);

private:
	void loadCorrections();
	map<string, string> corrections;
	std::string correctionsFilename;
};

} /* namespace netlib */

#endif /* CORRECTOR_H_ */
