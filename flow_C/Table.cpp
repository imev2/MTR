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

void Table1D::log_transform()
{
	int l;
	int n_che = num_channel + 1;
	for (int c = 0; c < n_che; c++)
	for (int l = 0; l < num_partition; l++) {
			table[c][l] = log10(table[c][l] + 1);
	}
}

Table1D::Table1D(int num_partition,int num_channel, float** split)
{
	int i;
	dim = 1;
	this->num_partition = num_partition;
	this->split = split;
	this->num_channel = num_channel;
	int n_che = num_channel + 1;
	table = new float*[n_che];
	for (int c = 0; c < n_che; c++) {
		table[c] = new float[num_partition];
		for (i = 0; i < num_partition; i++)
			table[c][i] = 0.0f;
	}
	
}

Table1D::~Table1D()
{
	int tam = num_channel + 1;
	for (int i = 0; i < tam; i++)
		delete[] table[i];
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
	int* dim1 = new int[n_lin];
	int* dim2 = new int[n_lin];
	for (int i = 0; i < n_lin; i++) {
		dim1[i] = binary_search(dados[i * n_col + col1], -1, num_partition, 0);
		dim2[i] = binary_search(dados[i * n_col + col2], -1, num_partition, 1);
		table[dim1[i] * num_partition + dim2[i]] += 1.0f;
	}

	log_transform();


	delete[] dim2;
	delete[] dim1;
	delete[] dados;
}

void Table2D::log_transform()
{
	int i,l, lin;
	int n_che = num_channel + 1;
	for (int c = 0; c < n_che; c++)
	for (int l = 0; l < num_partition; l++) {
		lin = l * num_partition;
		for (i = 0; i < num_partition; i++)
			table[c][lin + i] = log10(table[c][lin + i] + 1);
	}
	
}


Table2D::Table2D(int num_partition, int num_channel, float** split)
{
	int j,i;
	dim = 2;
	this->num_partition = num_partition;
	this->split = split;
	this->num_channel = num_channel;
	table = new float* [num_channel + 1];
	for (int c = 0; c < num_channel + 1; c++) {
		table[c] = new float[num_partition * dim];
		for (i = 0; i < num_partition; i++)
			for (j = 0; j < num_partition; j++)
				table[c][i * num_partition + j] = 0.0f;
	}
		
}

Table2D::~Table2D()
{
	int tam = num_channel + 1;
	for (int i = 0; i < tam; i++)
		delete[] table[i];
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

void Table3D::log_transform()
{
}

Table3D::Table3D(int num_partition, int num_channel, float** split)
{
	int d1,d2,d3,pos1,pos2;
	int i = 0;
	int tam1, tam2;
	dim = 3;
	this->num_channel = num_channel;
	int n_chen = num_channel + 1;
	this->num_partition = num_partition;
	this->split = split;
	tam2 = num_partition * num_partition;
	table = new float* [n_chen * num_partition];
	for (int c = 0; c < n_chen; c++) {
		tam1 = c * num_partition;
		for (d1 = 0; d1 < num_partition; d1++)
			pos1 = tam1 + d1;
			table[pos1] = new float[tam2];
			for (d2 = 0; d2 < num_partition; d2++) {
				tam2 = d2 * num_partition;
				for (d3 = 0; d3 < num_partition; d3++)
					table[pos1][tam2+d3] = 0.0f;
			}			
	}
}

Table3D::~Table3D()
{
	int tam1 = (num_channel + 1)*num_partition;
	for (int i = 0; i < tam1; i++)
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