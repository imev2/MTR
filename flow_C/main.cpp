#include<iostream>
#include<fstream>
#include<string>
#include <sstream>
#include <vector>

int main(int argc, char** argv) {
	std::printf("hello! %s",argv[1]);

	int idex = atoi("10");
	std::string folder = "C:\\repos\\MTR\\Scripts\\data\\ST1\\umap\\val";
	std::ifstream  f_out;
	std::string s = folder + "\\meta.txt";
	f_out.open(folder + "\\meta.txt");
	int dim;
	f_out >> dim;
	std::sstrea line;
	std::getline(f_out, line);
	std::getline(f_out, line);
	
	f_out.close();

}