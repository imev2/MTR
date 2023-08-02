#include "Table.h"


void Table1D::get_density(const char* file, double*& dados, bool save_data)
{
	int n_lin, n_col, c, y;
	int d1;
	readFile(file, n_lin, n_col, y, dados);
	int che = num_channel + 1;
	int col = n_col - 1;
	int col2 = col + 1;
	int* dim1 = new int[n_lin];
	double aux;
	size_t au;
	//get position for each line
	for (d1 = 0; d1 < n_lin; d1++) {
		dim1[d1] = binary_search(dados[d1 * n_col + col], -1, num_partition - 1, 0);
		//storege cell value for density and each chanel
		table[0][dim1[d1]] += 1.0f;
		for (c = 0; c < num_channel; c++)
			table[c + 1][dim1[d1]] += dados[d1 * n_col + c];
	}

	for (d1 = 0; d1 < num_partition; d1++) {
		for (c = 0; c < col2; c++) {
			if (table[0][d1] > 0) {
				table[c + 1][d1] = table[c + 1][d1] / table[0][d1];
			}
			else {
				table[c + 1][d1] = 0.0;
			}
		}

	}

	//log transform
	log_transform();
	standart();
	if (save_data) {
		std::ofstream f;
		f.open(file, std::ios::out | std::ios::binary);
		//"i i i",pheno ,nlin,ncol

		f.write(reinterpret_cast<const char*>(&y), sizeof(int));
		f.write(reinterpret_cast<const char*>(&n_lin), sizeof(int));
		col = n_col + che;
		f.write(reinterpret_cast<const char*>(&col), sizeof(int));
		for (d1 = 0; d1 < n_lin; d1++) {
			col2 = n_col * d1;
			f.write(reinterpret_cast<const char*>(&dados[col2]), sizeof(double) * col2);
			for (c = 0; c < che; c++) {
				aux = get(c, dim1[d1]);
				f.write(reinterpret_cast<const char*>(&aux), sizeof(double) * col2);
			}
		}
		f.close();
	}
	else {
		std::ofstream f;
		f.open(file, std::ios::out | std::ios::binary);
		//"i i i",pheno ,nlin,ncol

		f.write(reinterpret_cast<const char*>(&y), sizeof(int));
		f.write(reinterpret_cast<const char*>(&che), sizeof(int));
		f.write(reinterpret_cast<const char*>(&num_partition), sizeof(int));
		au = sizeof(double) * num_partition;
		for (c = 0; c < che; c++)
			f.write(reinterpret_cast<const char*>(table[c]), au);
		f.close();
	}




	delete[] dim1;
	delete[] dados;
}

void Table1D::log_transform()
{
	double mini;
	mini = 100000;
	for (int l = 0; l < num_partition; l++) {
		if (table[0][l] < mini)
			mini = table[0][l];
	}

	for (int l = 0; l < num_partition; l++) {
		table[0][l] = log10(table[0][l] - mini + 1);
	}



}

void Table1D::standart()
{
	double mean, soma;
	mean = 0.0f;
	soma = 0.0f;
	//mean
	for (int l = 0; l < num_partition; l++) {
		soma += table[0][l];
	}
	mean = soma / num_partition;
	//variancy
	soma = 0.0f;
	for (int l = 0; l < num_partition; l++) {
		soma += pow(table[0][l] - mean, 2);
	}
	soma = soma / num_partition;
	//sd
	soma = sqrt(soma);
	//standart
	for (int l = 0; l < num_partition; l++) {
		table[0][l] = (table[0][l] - mean) / soma;
	}



}

Table1D::Table1D(int num_partition, int num_channel, double** split)
{
	int i;
	dim = 1;
	this->num_partition = num_partition;
	this->split = split;
	this->num_channel = num_channel;
	int n_che = num_channel + 1;
	table = new double* [n_che];
	for (int c = 0; c < n_che; c++) {
		table[c] = new double[num_partition];
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

double Table1D::get(int channel, int dim1)
{
	return table[channel][dim1];
}

void Table1D::set(double value, int channel, int dim1)
{
	table[channel][dim1] = value;
}


void Table2D::get_density(const char* file, double*& dados, bool save_data)
{
	int n_lin, n_col, c, y, aux;
	int d1, d2;
	readFile(file, n_lin, n_col, y, dados);
	int che = num_channel + 1;
	int col1 = n_col - 2;
	int col2 = n_col - 1;
	int* dim1 = new int[n_lin];
	int* dim2 = new int[n_lin];
	double dx;



	//get position for each line
	for (d1 = 0; d1 < n_lin; d1++) {
		dim1[d1] = binary_search(dados[d1 * n_col + col1], -1, num_partition - 1, 0);
		dim2[d1] = binary_search(dados[d1 * n_col + col2], -1, num_partition - 1, 1);
		//storege cell value for density and each chanel
		table[0][dim1[d1] * num_partition + dim2[d1]] += 1.0f;
		d2 = dim1[d1] * num_partition;
		for (c = 0; c < num_channel; c++) {
			double aux2 = dados[d1 * n_col + c];
			table[c + 1][d2 + dim2[d1]] += aux2;
		}

	}

	for (d1 = 0; d1 < num_partition; d1++) {
		for (d2 = 0; d2 < num_partition; d2++) {
			aux = d1 * num_partition;
			if (table[0][aux + d2] > 0) {
				for (c = 0; c < num_channel; c++)
					table[c + 1][aux + d2] = table[c + 1][aux + d2] / table[0][aux + d2];
			}
			else {
				for (c = 0; c < num_channel; c++)
					table[c + 1][aux + d2] = 0.0;
			}
		}
	}
	/*
	c = 0;
	for (d1 = 0; d1 < num_partition; d1++) {
		for (d2 = 0; d2 < num_partition; d2++) {
			std::cout << table[c][d1 * num_partition + d2] << "  ";
		}
		std::cout << std::endl;
	}
	std::cout << "\n\n";
	*/
	//log transform
	log_transform();
	standart();
	/*
	c = 1;
	for (d1 = 0; d1 < num_partition; d1++) {
		for (d2 = 0; d2 < num_partition; d2++) {
			std::cout << table[c][d1 * num_partition + d2] << "  ";
		}
		std::cout << std::endl;
	}
	*/

	if (save_data) {
		std::ofstream f;
		f.open(file, std::ios::out | std::ios::binary);
		//"i i i",pheno ,nlin,ncol

		f.write(reinterpret_cast<char*>(&y), sizeof(int));
		f.write(reinterpret_cast<char*>(&n_lin), sizeof(int));
		col1 = num_channel + che;
		f.write(reinterpret_cast<char*>(&col1), sizeof(int));
		for (d1 = 0; d1 < n_lin; d1++) {
			col2 = n_col * d1;
			f.write(reinterpret_cast<char*>(&dados[col2]), sizeof(double) * num_channel);
			for (c = 0; c < che; c++) {
				double dx = get(c, dim1[d1], dim2[d1]);
				f.write(reinterpret_cast<char*>(&dx), sizeof(double));
			}
		}
		f.close();
	}
	else {
		std::ofstream f;
		f.open(file, std::ios::out | std::ios::binary);
		//"i i i",pheno ,nlin,ncol

		f.write(reinterpret_cast<char*>(&y), sizeof(int));
		f.write(reinterpret_cast<char*>(&n_lin), sizeof(int));
		col1 = che;
		f.write(reinterpret_cast<char*>(&col1), sizeof(int));
		for (c = 0; c < che; c++)
			for (d1 = 0; d1 < num_partition; d1++) {
				aux = d1 * num_partition;
				for (d2 = 0; d2 < num_partition; d2++) {
					dx = table[c][aux + d2];
					f.write(reinterpret_cast<char*>(&dx), sizeof(double));
				}
			}
		f.close();
	}


	delete[] dim1;
	delete[] dim2;
	delete[] dados;
}

void Table2D::log_transform()
{
	int i;
	int	l;
	int lin;
	int n_che = num_channel + 1;
	double mini;

	mini = 100000.0f;
	for (l = 0; l < num_partition; l++) {
		lin = l * num_partition;
		for (i = 0; i < num_partition; i++)
			if (table[0][lin + i] < mini)
				mini = table[0][lin + i];
	}
	for (l = 0; l < num_partition; l++) {
		lin = l * num_partition;
		for (i = 0; i < num_partition; i++)
			table[0][lin + i] = log10(table[0][lin + i] - mini + 1);
	}
}


void Table2D::standart()
{
	int i, l, lin;
	int n_che = num_channel + 1;
	double mean, soma;

	mean = 0.0f;
	soma = 0.0f;
	//mean
	for (l = 0; l < num_partition; l++) {
		lin = l * num_partition;
		for (i = 0; i < num_partition; i++)
			soma += table[0][lin + i];
	}
	mean = soma / (num_partition * num_partition);
	//variancy
	soma = 0.0f;
	for (int l = 0; l < num_partition; l++) {
		lin = l * num_partition;
		for (i = 0; i < num_partition; i++)
			soma += pow(table[0][lin + i] - mean, 2);
	}
	soma = soma / (num_partition * num_partition);
	//sd
	soma = sqrt(soma);
	//standart
	for (int l = 0; l < num_partition; l++) {
		lin = l * num_partition;
		for (i = 0; i < num_partition; i++)
			table[0][lin + i] = (table[0][lin + i] - mean) / soma;
	}
}




Table2D::Table2D(int num_partition, int num_channel, double** split)
{
	int j, i;
	dim = 2;
	this->num_partition = num_partition;
	this->split = split;
	this->num_channel = num_channel;
	table = new double* [num_channel + 1];
	for (int c = 0; c < num_channel + 1; c++) {
		table[c] = new double[num_partition * num_partition];
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

double Table2D::get(int channel, int dim1, int dim2)
{
	return table[channel][dim1 * num_partition + dim2];
}

void Table2D::set(double value, int channel, int dim1, int dim2)
{
	table[channel][dim1 * num_partition + dim2] = value;
}



void Table3D::get_density(const char* file, double*& dados, bool save_data)
{
	int n_lin, n_col, c, y,au,au2,au3;
	int d1, d2, d3;
	readFile(file, n_lin, n_col, y, dados);
	int che = num_channel + 1;
	int col1 = n_col - 3;
	int col2 = n_col - 2;
	int col3 = n_col - 1;
	int* dim1 = new int[n_lin];
	int* dim2 = new int[n_lin];
	int* dim3 = new int[n_lin];
	double dx, aux;
	//get position for each line
	for (d1 = 0; d1 < n_lin; d1++) {
		dim1[d1] = binary_search(dados[d1 * n_col + col1], -1, num_partition - 1, 0);
		dim2[d1] = binary_search(dados[d1 * n_col + col2], -1, num_partition - 1, 1);
		dim3[d1] = binary_search(dados[d1 * n_col + col3], -1, num_partition - 1, 2);
		//storege cell value for density and each chanel
		table[dim1[d1]][dim2[d1] * num_partition + dim3[d1]] += 1.0f;

		d2 = dim2[d1] * num_partition + dim3[d1];
		for (c = 0; c < col2; c++)
			table[(c + 1) * num_partition + dim1[d1]][d2] += dados[d1 * n_col + c];
	}

	for (d1 = 0; d1 < num_partition; d1++) {
		for (d2 = 0; d2 < num_partition; d2++) {
			for (int d3 = 0; d3 < num_partition; d3++) {
				for (c = 0; c < num_channel; c++)
					if (table[d1][d2 * num_partition + d3] > 0.0) {
						table[(c + 1) * num_partition + d1][d2 * num_partition + d3] = table[c * num_partition + d1][d2 * num_partition + d3] / table[d1][d2 * num_partition + d3];
					}
					else {
						table[(c + 1) * num_partition + d1][d2 * num_partition + d3] = 0.0;
					}
			}
		}
	}

	//log transform
	log_transform();
	standart();

	if (save_data) {
		std::ofstream f;
		f.open(file, std::ios::out | std::ios::binary);
		//"i i i",pheno ,nlin,ncol

		f.write(reinterpret_cast<const char*>(&y), sizeof(int));
		f.write(reinterpret_cast<const char*>(&n_lin), sizeof(int));
		col1 = n_col + che;
		f.write(reinterpret_cast<const char*>(&col1), sizeof(int));
		for (d1 = 0; d1 < n_lin; d1++) {
			col2 = n_col * d1;
			f.write(reinterpret_cast<const char*>(&dados[col2]), sizeof(double) * col2);
			for (c = 0; c < che; c++) {
				aux = get(c, dim1[d1], dim2[d1], dim3[d1]);
				f.write(reinterpret_cast<const char*>(&aux), sizeof(double) * col2);
			}
		}
		f.close();
	}
	else {
		std::ofstream f;
		f.open(file, std::ios::out | std::ios::binary);
		//"i i i",pheno ,nlin,ncol

		f.write(reinterpret_cast<char*>(&y), sizeof(int));
		f.write(reinterpret_cast<char*>(&n_lin), sizeof(int));
		col1 = che;
		f.write(reinterpret_cast<char*>(&col1), sizeof(int));
		for (c = 0; c < che; c++) {
			au = c * num_partition;
			for (d1 = 0; d1 < num_partition; d1++) {
				au2 = au + d1;
				for (d2 = 0; d2 < num_partition; d2++) {
					au3 = d2 * num_partition;
					for (d3 = 0; d3 < num_partition; d3++) {
						dx = table[au2][au3 + d3];
						f.write(reinterpret_cast<char*>(&dx), sizeof(double));
					}
				}
			}
		}
		f.close();
	}

	delete[] dim1;
	delete[] dim2;
	delete[] dim3;
	delete[] dados;
}

void Table3D::log_transform()
{
	int d1, d2, d3;
	int i = 0;
	int tam2;
	int n_chen = num_channel + 1;
	int n_chen2 = num_partition * num_partition;
	double mini;

	mini = 100000;
	for (d1 = 0; d1 < num_partition; d1++) {
		for (d2 = 0; d2 < num_partition; d2++) {
			tam2 = d2 * num_partition;
			for (d3 = 0; d3 < num_partition; d3++)
				if (mini < table[d1][tam2 + d3])
					table[d1][tam2 + d3] = mini;
		}
	}
	for (d1 = 0; d1 < num_partition; d1++) {
		for (d2 = 0; d2 < num_partition; d2++) {
			tam2 = d2 * num_partition;
			for (d3 = 0; d3 < num_partition; d3++)
				table[d1][tam2 + d3] = log10(table[d1][tam2 + d3] - mini + 1);
		}
	}
}


void Table3D::standart()
{
	int d1, d2, d3, pos1;
	int i = 0;
	int tam1, tam2;
	int n_chen = num_channel + 1;
	int n_chen2 = num_partition * num_partition;
	double mean, soma;
	for (int c = 0; c < n_chen; c++) {
		tam1 = c * num_partition;
		mean = 0.0f;
		soma = 0.0f;
		//mean
		for (d1 = 0; d1 < num_partition; d1++) {
			pos1 = tam1 + d1;
			for (d2 = 0; d2 < num_partition; d2++) {
				tam2 = d2 * num_partition;
				for (d3 = 0; d3 < num_partition; d3++)
					soma += table[pos1][tam2 + d3];
			}
		}
		mean = soma / (num_partition * num_partition * num_partition);
		//variancy
		soma = 0.0f;
		for (d1 = 0; d1 < num_partition; d1++) {
			pos1 = tam1 + d1;
			for (d2 = 0; d2 < num_partition; d2++) {
				tam2 = d2 * num_partition;
				for (d3 = 0; d3 < num_partition; d3++)
					soma += pow(table[pos1][tam2 + d3] - mean, 2);
			}
		}
		soma = soma / (num_partition * num_partition * num_partition);
		//sd
		soma = sqrt(soma);
		//standart
		for (d1 = 0; d1 < num_partition; d1++) {
			pos1 = tam1 + d1;
			for (d2 = 0; d2 < num_partition; d2++) {
				tam2 = d2 * num_partition;
				for (d3 = 0; d3 < num_partition; d3++)
					table[pos1][tam2 + d3] = (table[pos1][tam2 + d3] - mean) / soma;
			}
		}

	}

}

Table3D::Table3D(int num_partition, int num_channel, double** split)
{
	int d1, d2, d3, pos1;
	int i = 0;
	int tam1, tam2;
	dim = 3;
	this->num_channel = num_channel;
	int n_chen = num_channel + 1;
	this->num_partition = num_partition;
	this->split = split;
	int n_chen2 = num_partition * num_partition;
	table = new double* [n_chen * num_partition];
	for (int c = 0; c < n_chen; c++) {
		tam1 = c * num_partition;
		for (d1 = 0; d1 < num_partition; d1++)
			pos1 = tam1 + d1;
		table[pos1] = new double[n_chen2];
		for (d2 = 0; d2 < num_partition; d2++) {
			tam2 = d2 * num_partition;
			for (d3 = 0; d3 < num_partition; d3++)
				table[pos1][tam2 + d3] = 0.0f;
		}
	}
}

Table3D::~Table3D()
{
	int tam1 = (num_channel + 1) * num_partition;
	for (int i = 0; i < tam1; i++)
		delete[] table[i];
	delete[] table;
}

double Table3D::get(int channel, int dim1, int dim2, int dim3)
{
	return table[channel * num_partition + dim1][dim2 * num_partition + dim3];
}

void Table3D::set(double value, int channel, int dim1, int dim2, int dim3)
{
	table[channel * num_partition + dim1][dim2 * num_partition + dim3] = value;
}


int Table::binary_search(double value, int pos_i, int pos_f, int dim)
{
	int p;
	if (pos_f - pos_i == 1) {
		return pos_f;
	}
	int i = (pos_f - pos_i) / 2 + pos_i;
	if (value > split[dim][i]) {
		p = binary_search(value, i, pos_f, dim);
	}
	else {
		p = binary_search(value, pos_i, i, dim);
	}
	return p;
}
void Table::readFile(const char* file, int& n_lin, int& n_col, int& y, double*& dados)
{
	int aux[3];
	int col;
	double* d_aux;
	std::ifstream f;
	f.open(file, std::ios::in | std::ios::binary);
	f.read(reinterpret_cast<char*>(aux), sizeof(int) * 3);
	//"i i i",pheno ,nlin,ncol
	y = aux[0];
	n_lin = aux[1];
	n_col = aux[2];
	dados = new double[n_lin * n_col];
	d_aux = new double[aux[2]];

	col = n_col - dim;
	size_t s = sizeof(double) * n_col;
	for (int l = 0; l < n_lin; l++) {
		f.read(reinterpret_cast<char*>(d_aux), s);
		memcpy((void*)&(dados[l * n_col]), (const void*)(d_aux), s);
	}
	f.close();
	delete[] d_aux;
}