#pragma once
#include<iostream>
#include<fstream>
#include<string>
#include <sstream>
#include <vector>
class Data
{
private:
	double** painel;

	double* get_density(const char* file);
public:
	int num_partition;
	int dim;

	Data();
	~Data();
	Data(const char* file_space, int num_partition);
	void save(const char* file);
	void load(const char* file_space);
	void apply_cells(const char* file);
	


};

