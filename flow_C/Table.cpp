#include "Table.h"
void Table1D::get_density(const char* file)
{
	float* dados;
	get_density(file, dados);
	//unalloc
	delete[] dados;
}

void Table1D::get_density(const char* file, float*& dados)
{
	int n_lin, n_col, y;
	readFile(file, n_lin, n_col, y, dados);



}

Table1D::Table1D(int num_partition, float** split)
{
	dim = 1;
	this->num_partition = num_partition;
	this->split = split;
	table = new float[num_partition];
	for (int i = 0; i < num_partition; i++)
		table[i] = 0.0f;
}

Table1D::~Table1D()
{
	delete[] table;
}


void Table2D::get_density(const char* file)
{
	float* dados;
	get_density(file, dados);


	//unalloc
	delete[] dados;
}

void Table2D::get_density(const char* file, float*& dados)
{
	int pos1,pos2;
	int n_lin, n_col, y, col1,col2;
	readFile(file, n_lin, n_col, y, dados);
	col1 = n_col - 2;
	col2 = n_col - 1;
	for (int i = 0; i < n_lin; i++) {
		pos1 = binary_search(dados[i*n_col+col1],split)
	}






		delete[] dados;
}

Table2D::Table2D(int num_partition, float** split)
{
	int j;
	dim = 2;
	this->num_partition = num_partition;
	this->split = split;
	table = new float[num_partition * dim];
	for (int i = 0; i < num_partition; i++)
		for (j = 0; j < num_partition; j++)
			table[i * num_partition + j] = 0.0f;
}

Table2D::~Table2D()
{
	delete[] table;
}

void Table3D::get_density(const char* file)
{
	float* dados;
	get_density(file, dados);
	//unalloc
	delete[] dados;
}

void Table3D::get_density(const char* file, float*& dados)
{
	int n_lin, n_col, y;
	readFile(file, n_lin, n_col, y, dados);
}

Table3D::Table3D(int num_partition, float** split)
{
	int j;
	int i = 0;
	int tam;
	tam = num_partition * dim;
	dim = 3;
	this->num_partition = num_partition;
	this->split = split;
	table = new float* [num_partition];
	for (; i < num_partition; i++) {
		table[i] = new float[tam];
	}
	for (int d = 0; d < num_partition; d++) {
		for (i = 0; i < num_partition; i++) {
			for (j = 0; j < num_partition; j++)
				table[d][i * num_partition + j] = 0.0f;
		}
	}
}

Table3D::~Table3D()
{
	for (int i = 0; i < num_partition; i++)
		delete[] table[i];
	delete[] table;
}


int Table::binary_search(float value, int pos_i, int pos_f, int dim)
{
	int p;
	if (pos_f - pos_i == 1) {
		return pos_f;
	}
	int i = (pos_f - pos_i) / 2 + pos_i;
	if (value > split[dim][i]) {
		p = binary_search(value,i, pos_f, dim);
	}
	else {
		p = binary_search(value, pos_i, i, dim);
	}
	return p;
}
void Table::readFile(const char* file, int& n_lin, int& n_col, int& y, float*& dados)
{
	int aux[3];
	int col;
	float* d_aux;
	std::ifstream f;
	f.open(file, std::ios::in | std::ios::binary);
	f.read(reinterpret_cast<char*>(aux), sizeof(int) * 3);
	//"i i i",pheno ,nlin,ncol
	y = aux[0];
	n_lin = aux[1];
	n_col = aux[2];
	dados = new float[n_lin * n_col];
	d_aux = new float[aux[2]];

	col = n_col - dim;
	size_t s = sizeof(float) * n_col;
	for (int l = 0; l < n_lin; l++) {
		f.read(reinterpret_cast<char*>(d_aux), s);
		memcpy((void*)&(dados[l * n_col]), (const void*)(d_aux), s);
	}
	f.close();
	delete[] d_aux;
}