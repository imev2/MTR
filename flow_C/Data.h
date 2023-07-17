#pragma once
#include<iostream>
#include<fstream>
#include"Table.h"
class Data
{
private:
	double** split;
	
	//void get_density(const char* file,float** table);
public:
	int num_partition;
	int dim;
	int num_channel;

	Data();
	~Data();
	Data(const char* file_space, int num_partition);
	void save(const char* file);
	void load(const char* file_space);
	void apply_cells(const char* file);
	void apply_space(const char* file);
};

