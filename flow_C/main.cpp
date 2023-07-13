#include"Data.h"

int main(int argc, char** argv) {
	const char* file_split = "C:\\repos\\MTR\\Scripts\\data\\ST1\\umap\\split.dat";
	const char* file = "C:\\repos\\MTR\\Scripts\\data\\ST1\\umap\\val_copy\\AD018C.dat";
	const char* file_space = "C:\\repos\\MTR\\Scripts\\data\\ST1\\umap\\space.txt";
	int num_partition = 5;

	
	Data data(file_space,num_partition);
	data.save(file_split);
	
	Data data2;
	data2.load(file_split);
	data2.apply_cells(file);
	return 0;
}