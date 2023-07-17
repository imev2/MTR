#pragma once
#include<iostream>
#include<fstream>
#include <math.h>

class Table
{
protected:
	int binary_search(double value, int pos_i, int pos_f, int dim);
	void readFile(const char* file, int& n_lin, int& n_col, int& y, double*& dados);
	double** table;
public:
	int dim;
	int num_partition;
	int num_channel;
	double** split;

	virtual void get_density(const char* file, double*& dados, bool save_data) = 0;
	virtual void log_transform() = 0;
	virtual void standart() = 0;
	//	virtual void cell_file(const char* file);
};

class Table1D : private Table {
public:
	Table1D(int num_partition, int num_channel, double** split);
	~Table1D();
	double get(int channel, int dim1);
	void set(double value, int channel, int dim1);
	void get_density(const char* file, double*& dados, bool save_data);
	void log_transform();
	void standart();
};

class Table2D : private Table {
public:
	Table2D(int num_partition, int num_channel, double** split);
	~Table2D();
	double get(int channel, int dim1, int dim2);
	void set(double value, int channel, int dim1, int dim2);
	void get_density(const char* file, double*& dados, bool save_data);
	void log_transform();
	void standart();
};

class Table3D : private Table {
public:
	Table3D(int num_partition, int num_channel, double** split);
	~Table3D();
	double get(int channel, int dim1, int dim2, int dim3);
	void set(double value, int channel, int dim1, int dim2, int dim3);
	void get_density(const char* file, double*& dados, bool save_data);
	void log_transform();
	void standart();
};

