/**
 * @file
 * @brief Defines Corrector class, which corrects common spelling errors in an input corpus.
 *
 * Corrector class corrects common spelling errors in an input corpus. It loads corrections from a text file.
 *
 */

#include "../NetLib.h"
#include "Corrector.h"

#include <iostream>
#include <fstream>
#include <string>
#include <boost/regex.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/regex.hpp>

namespace netlib {

Corrector::Corrector() {
	correctionsFilename = datasetsDir + "/corrections.txt";
	loadCorrections();
}

Corrector::~Corrector() {
}

void Corrector::loadCorrections() {
	ifstream file(correctionsFilename);

	string line;
	vector<string> words;
	int num = 0;
	while (!file.eof()) {
		getline(file, line, '\n');
		boost::trim(line);
		if (line.length() == 0) {
			continue;
		}

		boost::split_regex(words, line, boost::regex("\\s+"));

		string w1 = words[0];
		string w2 = words[1];

		corrections.insert(pair<string,string>(w1, w2));
		num++;
	}
	cout << "loaded " << num << " corrections" << endl;
}

void Corrector::correct(vector<string> &tokens) {
	for (int i = 0; i < tokens.size(); i++) {
		string token = tokens[i];

		map<string,string>::iterator it = corrections.find(token);
		if (it != corrections.end()) {
//			cout << "correcting " << token << " to " << it->second << endl;
			tokens[i] = it->second;
		}
	}
}

} /* namespace netlib */
