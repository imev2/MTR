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
	void binary_search(float value, double** table, int pos_i, int pos_f, int dim);
	void get_density(const char* file,double** table);
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

