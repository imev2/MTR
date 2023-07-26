#include"Data.h"

int main(int argc, char** argv) {
	/*
	arguments:
	p   file_quimera   num_partition  file_split
	c   file_sample  file_split
	s   file_sample  file_split
	*/
	//std::cout << argv[1] << " " << argv[2];
	if (argv[1][0] == 'p') {
		Data data(argv[2], atoi(argv[3]));
		data.save(argv[4]);
	}
	if (argv[1][0] == 'c') {
		Data data2;
		data2.load(argv[3]);
		data2.apply_cells(argv[2]);
	}
	if (argv[1][0] == 's') {
		Data data2;
		data2.load(argv[3]);
		data2.apply_space(argv[2]);
	}
	
	return 0;
}