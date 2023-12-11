/**
 * @file
 * @brief Defines the FileUtil class, which has methods to read from and write to files.
 *
 */

#include "FileUtil.h"

namespace netlib {

FileUtil::FileUtil() {
}

FileUtil::~FileUtil() {
}

float FileUtil::readFloat(ifstream& file) {
	float f;
	file.read(reinterpret_cast<char *>(&f), sizeof(float));
	return f;
}

void FileUtil::writeFloat(float f, ofstream& file) {
	file.write(reinterpret_cast<char *>(&f), sizeof(float));
}

long FileUtil::readLong(ifstream& file) {
	long l;
	file.read(reinterpret_cast<char *>(&l), sizeof(long));
	return l;
}

void FileUtil::writeLong(long l, ofstream& file) {
	file.write(reinterpret_cast<char *>(&l), sizeof(long));
}

int FileUtil::readInt(ifstream& file) {
	int i;
	file.read(reinterpret_cast<char *>(&i), sizeof(int));
	return i;
}

void FileUtil::writeInt(int i, ofstream& file) {
	file.write(reinterpret_cast<char *>(&i), sizeof(int));
}

uint64 FileUtil::readInt64(ifstream &file) {
	uint64 result = 0;
	unsigned char buf[8];
	file.read(reinterpret_cast<char*>(buf), 8);
	for (int i = 0; i < 8; i++) {
		int shift = (7-i) * 8;
		uint64 b = ((uint64) buf[i]) & 0xff;
		result += b << shift;
	}
	return result;
}

void FileUtil::writeInt64(uint64 val, ofstream &file) {
	unsigned char buf[8];
	for (int i = 7; i >= 0; i--) {
		buf[i] = val & 0xff;
		val = val >> 8;
	}
	file.write(reinterpret_cast<char*>(buf), 8);
}

string FileUtil::readString(ifstream& file) {
	string s;
	getline(file, s, '\0');
	return s;
}

void FileUtil::writeString(string s, ofstream& file) {
	file.write(s.c_str(), s.length() + 1); //+1 causes null term to be written
}

} /* namespace netlib */
