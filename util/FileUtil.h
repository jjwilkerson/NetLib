/**
 * @file
 * @brief Declares the FileUtil class, which has methods to read from and write to files.
 *
 */

#ifndef FILEUTIL_H_
#define FILEUTIL_H_

#include <fstream>
#include <iostream>
#include <string>
#include <utility>

typedef unsigned long long int uint64;

using namespace std;

namespace netlib {

/**
 * @brief Has methods to read from and write to files.
 *
 */
class FileUtil {
public:
	FileUtil();
	virtual ~FileUtil();
	static float readFloat(ifstream& file);
	static void writeFloat(float f, ofstream& file);
	static long readLong(ifstream& file);
	static void writeLong(long l, ofstream& file);
	static int readInt(ifstream& file);
	static void writeInt(int i, ofstream& file);
	static uint64 readInt64(ifstream &file);
	static void writeInt64(uint64 val, ofstream &file);
	static string readString(ifstream& file);
	static void writeString(string s, ofstream& file);
};

} /* namespace netlib */

#endif /* FILEUTIL_H_ */
