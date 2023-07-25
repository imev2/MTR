#include "Data.h"

Data::Data()
{
	num_channel = 0;
	num_partition = 0;
	dim = 0;
	split = nullptr;
}

Data::~Data()
{
	if (split != nullptr) {
		for (int i = 0; i < dim; i++) {
			delete[] split[i];
		}
		delete[] split;
	}
}

Data::Data(const char* file_space, int num_partition)
{
	//std::sync_with_stdio(false);
	double* aux;
	double max, min;
	std::ifstream file;
	int n_lin, n_col,c,i,l,d;
	double** dim_value;
	this->num_partition = num_partition;
	file.open(file_space, std::ios::in);
	file >> n_lin >> n_col >> dim;
	//painel alocation
	split = new double* [dim];
	for (i = 0; i < dim; i++) {
		split[i] = new double[num_partition - 1];
	}

	//dimmention alocation
	num_channel = n_col - dim;
	dim_value = new double* [dim];
	for (i = 0; i < dim; i++) {
		dim_value[i] = new double[n_lin];
	}
	//load cells
	aux = new double[n_col];
	for (l = 0; l < n_lin; l++) {
		if (l % 1000 == 0) std::cout << l / 1000 << " mil\n";
		for (c = 0; c < n_col; c++) {
			file >> aux[c];
		}
		
		for (int d = 0; d < dim; d++) {
			dim_value[d][l] = aux[num_channel + d];
		}
	}
	file.close();
	//calculate painel
	for (int d = 0; d < dim; d++) {
		max = -10000;
		min = 10000;
		for (int l = 0; l < n_lin; l++) {
			if (dim_value[d][l] < min) {
				min = dim_value[d][l];
				continue;
			}
			if (dim_value[d][l] > max) {
				max = dim_value[d][l];
			}
		}
		double range = (max - min) / num_partition;

		for (int i = 1; i < num_partition; i++) {
			split[d][i - 1] = range * i + min;
		}
	}
	std::cout << "";
	//unalocate
	for (int i = 0; i < dim; i++) {
		delete[] dim_value[i];
	}
	delete[] dim_value;
	delete[] aux;
}

void Data::save(const char* file_space)
{
	std::ofstream file;
	file.open(file_space, std::ios::out | std::ios::binary);
	file.write(reinterpret_cast<char*>(&dim), sizeof(int));
	file.write(reinterpret_cast<char*>(&num_partition), sizeof(int));
	file.write(reinterpret_cast<char*>(&num_channel), sizeof(int));
	for (int d = 0; d < dim; d++) {
		file.write(reinterpret_cast<char*>(split[d]), sizeof(double) * (num_partition - 1));
	}
	file.close();
}

void Data::load(const char* file_space)
{
	std::ifstream file;
	file.open(file_space, std::ios::in | std::ios::binary);
	file.read(reinterpret_cast<char*>(&dim), sizeof(int));
	file.read(reinterpret_cast<char*>(&num_partition), sizeof(int));
	file.read(reinterpret_cast<char*>(&num_channel), sizeof(int));
	split = new double* [dim];
	for (int i = 0; i < dim; i++) {
		split[i] = new double[num_partition - 1];
	}
	for (int d = 0; d < dim; d++) {
		file.read(reinterpret_cast<char*>(split[d]), sizeof(double) * (num_partition - 1));
	}
	file.close();
}

void Data::apply_cells(const char* file)
{
	double* dados;
	//alocate
	Table* tab;
	if (dim == 1) {
		Table1D* table = new Table1D(num_partition, num_channel, split);
		tab = (Table*)table;

	}
	else {
		if (dim == 2) {
			Table2D* table = new Table2D(num_partition, num_channel, split);
			tab = (Table*)table;
		}
		else {
			Table3D* table = new Table3D(num_partition, num_channel, split);
			tab = (Table*)table;
		}
	}
	tab->get_density(file, dados, true);
	//generate density
	//load file in memory





	//unalocate
}

void Data::apply_space(const char* file)
{
	double* dados;
	//alocate
	Table* tab;
	if (dim == 1) {
		Table1D* table = new Table1D(num_partition, num_channel, split);
		tab = (Table*)table;

	}
	else {
		if (dim == 2) {
			Table2D* table = new Table2D(num_partition, num_channel, split);
			tab = (Table*)table;
		}
		else {
			Table3D* table = new Table3D(num_partition, num_channel, split);
			tab = (Table*)table;
		}
	}
	tab->get_density(file, dados, false);
}



