#include "Data.h"

Data::Data()
{
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
	float* values;
	float max, min;
	std::ifstream file;
	int n_lin, n_col;
	int n_mark;
	float** dim_value;
	this->num_partition = num_partition;
	file.open(file_space, std::ios::in);
	file >> n_lin >> n_col >> dim;
	//painel alocation
	split = new float*[dim];
	for (int i = 0; i < dim; i++) {
		split[i] = new float[num_partition-1];
	}

	//dimmention alocation
	n_mark = n_col - dim;
	dim_value = new float* [dim];
	for (int i = 0; i < dim; i++) {
		dim_value[i] = new float[n_lin];
	}
	//load cells
	values = new float[n_lin * n_mark];
	for (int l = 0; l < n_lin; l++) {
		if (l % 1000==0) std::cout << l / 1000 << " mil\n";
		for (int c = 0; c < n_mark; c++) {
			file>>values[c * n_lin + l];
		}
		for (int d = 0; d < dim; d++) {
			file >> dim_value[d][l];
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
		float range = (max - min)/num_partition;
		
		for (int i = 1; i < num_partition; i++) {
			split[d][i-1] = range * i + min;
		}
	}
	std::cout << "";
	//unalocate
	for (int i = 0; i < dim; i++) {
		delete[] dim_value[i];
	}
	delete[] dim_value;

	delete[] values;
}

void Data::save(const char* file_space)
{
	std::ofstream file;
	file.open(file_space, std::ios::out | std::ios::binary);
	file.write(reinterpret_cast<char*>(&dim), sizeof(int));
	file.write(reinterpret_cast<char*>(&num_partition), sizeof(int));
	for (int d = 0; d < dim; d++) {
		file.write(reinterpret_cast<char*>(split[d]), sizeof(float)*(num_partition-1));
	}
	file.close();
}

void Data::load(const char* file_space)
{
	std::ifstream file;
	file.open(file_space, std::ios::in | std::ios::binary);
	file.read(reinterpret_cast<char*>(&dim), sizeof(int));
	file.read(reinterpret_cast<char*>(& num_partition), sizeof(int));
	split = new float* [dim];
	for (int i = 0; i < dim; i++) {
		split[i] = new float[num_partition - 1];
	}
	for (int d = 0; d < dim; d++) {
		file.read(reinterpret_cast<char*>(split[d]), sizeof(float) * (num_partition - 1));
	}
	file.close();
}

void Data::apply_cells(const char* file)
{	
	float* dados;
	//alocate
	Table* tab;
	if (dim == 1) {
		Table1D* table = new Table1D(num_partition, split);
		tab = (Table*)table;

	}
	else {
		if (dim == 2) {
			Table2D* table = new Table2D(num_partition, split);
			tab = (Table*)table;
		}
		else {
			Table3D* table = new Table3D(num_partition, split);
			tab = (Table*)table;
		}
	}	
	tab->get_density(file, dados);
	//generate density
	//load file in memory
	




	//unalocate
	delete[] dados;
}


/*

void Data::log_transform(float** table)
{
	for (int d = 0; d < dim; d++)
		for (int i = 0; i < num_partition; i++)
			table[d][i] = log10(table[d][i] + 1);
}

void Data::standart(float** table)
{
	float mean, soma;

	for (int d = 0; d < dim; d++) {
		//mean
		mean = 0.0f;
		soma = 0.0f;
		for (int i = 0; i < num_partition; i++)
			soma += table[d][i];
		mean = soma / num_partition;
		//variancy
		soma = 0.0f;
		for (int i = 0; i < num_partition; i++)
			soma += powf(table[d][i]-mean,2);
		soma = soma / num_partition;
		//sd
		soma = sqrtf(soma);
		//standart
		for (int i = 0; i < num_partition; i++)
			table[d][i] = (table[d][i]-mean)/soma ;
	}
		
}
*/

