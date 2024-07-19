#pragma once
#pragma once
#pragma once

#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <vector>
#include <time.h>
#include <algorithm>
#include <cmath>
#include <Windows.h>
using namespace std;

//�������ģ��
template<typename T> class Matrix
{

public:

	/*************************************************
	Function: 	    	// Matrix()
	Description: 		// �޲ι��캯����ָ��ָ���
	Input:        	// ��
	Output: 		// ��
	Others: 		// ��
	*************************************************/
	Matrix() : width(0), height(0), pdata(nullptr), internal_data(false) {}

	/*************************************************
	Function: 	    	// Matrix()
	Description: 		// �вι��캯�������������׵�ַ�����߳�ʼֵ���൱�ڴ�����ǳ�������󣬽�ԭ��������������������
	Input:        	// ��
	Output: 		// ��
	Others: 		// �൱��ǳ����
	*************************************************/
	Matrix(void* data, int width, int height) : pdata((T*)data), width(width), height(height), internal_data(false) {}

	/*************************************************
	Function: 	    	// Matrix()
	Description: 		// �вι��캯����������ߣ������������飬�����ڴ�ռ�
	Input:        	// ��
	Output: 		// ��
	Others: 		// �����ڴ�ռ�
	*************************************************/
	Matrix(int width, int height) : width(width), height(height), internal_data(true) { pdata = new(std::nothrow) T[width * height](); } //  

	/*************************************************
	Function: 	    	// Matrix_copy
	Description: 		// ���ƾ������
	Input:        	// ��
	Output: 		// ��
	Others: 		// ��
	*************************************************/
	void Matrix_copy(const Matrix& m)
	{
		if (width >= m.width && height >= m.height && pdata)
		{
			memcpy(pdata, m.pdata, m.width * m.height * sizeof(T));
		}
	}

	/*************************************************
	Function: 	    	// Matrix()  ||
	Description: 		// ��ֵ����������������Ķ���Ϊ����ȥ��m����
	Input:        	// ��
	Output: 		// ��
	Others: 		// ������ȥ�ľ������m����������ǰ������
	*************************************************/
	Matrix(const Matrix& m)
	{
		width = m.width;
		height = m.height;
		pdata = new(std::nothrow) T[width * height]();
		memcpy(pdata, m.pdata, width * height * sizeof(T));
		internal_data = true;
	}

	/*************************************************
	Function: 	    	// Matric_Eye
	Description: 		// ���ɵ�λ����
	Input:        	// intputmatrix���������
	Output: 		// intputmatrix�����������ת��ɵ�λ����
	Others: 		// ��λ����Խ���Ϊ1������Ԫ��Ϊ0
	*************************************************/
	Matrix(int width, int height, bool eyeMtrix) : width(width), height(height), internal_data(true)
	{
		pdata = new(std::nothrow) T[width * height]();
		for (int i = 0; i < height; i++)
		{
			T* p = pdata + i * width;
			for (int j = 0; j < width; j++)
			{
				p[j] = (i == j) ? 1 : 0;
			}
		}
	}

	int width;
	int height;
	T* pdata;
	bool internal_data;

	/*************************************************
	Function: 	    	// ~Matrix()
	Description: 		// �����������Ƿ��������ռ�
	Input:        	// ��
	Output: 		// ��
	Others: 		// ��
	*************************************************/
	~Matrix()
	{
		if (internal_data)
		{
			if (pdata != NULL)
			{
				delete[] pdata;
				pdata = NULL;
			}
		}
	}

	/*************************************************
	Function: 	    	// SetPixel
	Description: 		// �����������Ƿ��������ռ�
	Input:        	// ��
	Output: 		// ��
	Others: 		// ��
	*************************************************/
	void SetPixel(int x, int y, T pixelValue) { pdata[y * width + x] = pixelValue; }

	/*************************************************
	Function: 	    	// GetPixel
	Description: 		// ���������X��Y���꣬��ȡ�������µ���ֵ
	Input:        	// ��
	Output: 		// ��
	Others: 		// ��
	*************************************************/
	T GetPixel(int x, int y) { return pdata[y * width + x]; }

	/*************************************************
	Function: 	    	// Matrix_length
	Description: 		// ��ȡ�˶�������鳤��
	Input:        	// ��
	Output: 		// ��
	Others: 		// ��
	*************************************************/
	int Matrix_length()
	{
		return  width * height;
	}

	/*************************************************
	Function: 	    	// Matric_multi
	Description: 		// �������
	Input:        	// leftmatrix����˾���rightmatrix���ҳ˾���
	Output: 		// outmatrix�˷��������
	Others: 		// ��
	*************************************************/
	void Matric_multi(Matrix<T>& leftmatrix, Matrix<T>& rightmatrix)
	{
		//���Ҿ������ת��
		Matrix<T> rightmatrix_T(rightmatrix.height, rightmatrix.width);
		rightmatrix_T.Matric_Trans(rightmatrix);//ת��

#pragma omp parallel for
		for (int i = 0; i < leftmatrix.height; i++)//������������൱��������������
		{
			T* mat1row = leftmatrix.pdata + i * leftmatrix.width;//��һ�����������
			T* outmatrow = pdata + i * width;//������������
			for (int j = 0; j < rightmatrix.width; j++)//�Ҳ�ڶ�������������൱��������������
			{
				T* mat2row = rightmatrix_T.pdata + j * leftmatrix.width;//�ڶ���������ת�ú�ľ��󣩵�����				
				T sumvalue = 0;
				for (int k = 0; k < leftmatrix.width; k++)
				{
					T mat1pos = *(mat1row + k);
					T mat2pos = *(mat2row + k);
					sumvalue = sumvalue + mat1pos * mat2pos;
				}

				*(outmatrow + j) = sumvalue;//�������i�У�j�еĵ���ֵ				
			}
		}

		//rightmatrix_T.~rightmatrix_T();
	}

	/*************************************************
	Function: 	    	// Matric_Trans
	Description: 		// ����ת��
	Input:        	// intputmatrix���������
	Output: 		// ���ת�þ���
	Others: 		// ת�þ������������Ҫ����������С��ƥ��
	*************************************************/
	void Matric_Trans(Matrix<T>& intputmatrix)
	{
		if ((width * height == intputmatrix.height * intputmatrix.width))
		{
			/*for (int i = 0; i < width; i++)
			{
				double* inputmatcol = intputmatrix.pdata + i;
				double* outputmatrow = pdata + i * intputmatrix.height;

				for (int j = 0; j < intputmatrix.height; j++)
				{
					*(outputmatrow + j) = *(inputmatcol + (j * intputmatrix.width));
				}
			}*/

			width = intputmatrix.height;
			height = intputmatrix.width;

			for (int i = 0, step1 = 0; i < height; i++, step1 += width)//�����ÿһ��
			{
				for (int j = 0, step2 = 0; j < width; j++, step2 += height)//�������ÿһ�еĽ��б���
				{
					pdata[step1 + j] = intputmatrix.pdata[step2 + i];
				}
			}

		}
		else
		{
			return;
		}
	}

	/*************************************************
	Function: 	    	// Matric_Eye
	Description: 		// ���ɵ�λ����
	Input:        	// intputmatrix���������
	Output: 		// intputmatrix�����������ת��ɵ�λ����
	Others: 		// ��λ����Խ���Ϊ1������Ԫ��Ϊ0
	*************************************************/
	void Matric_Eye()
	{
		for (size_t i = 0; i < height; i++)
		{
			T* inputrow = pdata + i * width;

			for (size_t j = 0; j < width; j++)
			{
				if (i == j)
				{
					*(inputrow + j) = 1;
				}
				else
				{
					*(inputrow + j) = 0;
				}
			}
		}
	}

	/*************************************************
	Function: 	    	// Matric_powervector
	Description: 		// �ݷ���������������ֵ����������
	Input:        	// orgmat����Ҫ��ľ���
	Output: 		// vectout��������������������eigmaxvalue��������ɷ�����ֵ
	Others: 		// �ݷ�����ֻ�����������ֵ������������������һ�������¿��������������ֵ����������
	*************************************************/
	void Matric_powervector(Matrix<T>& matEigvect, vector<T>& vecEigvalue);

	void House_func(Matrix<T>& xmat, Matrix<T>& vmat, double& beta);

	void HouseHolder_func(Matrix<T>& vmat, Matrix<T>& Pmat, double beta);

	void HouseHolder_Tridiagonal(Matrix<T>& Amat, Matrix<T>& Pmat, double beta);
};

//����ѭ����ȡ����������ֵ������������
template<typename T> void Matrix<T>::Matric_powervector(Matrix<T>& matEigvect, vector<T>& vecEigvalue)
{
	//��ʼ���ò���
	double tol = 1e-10;//��������
	int iternum = 100;//������������
	double sumtep = 1e-16;//�����µ�΢Сֵ
	//���������ֵ��Ŀ���Լ�����������Ŀ��
	int eignum = matEigvect.width;

	//���������ĳ���
	int eigveclenth = this->width;
	// 
	//������ʼ��һ��2-������������������Ϊ�м��������
	Matrix<T> matitervect(1, eigveclenth);
	for (int i = 0; i < eigveclenth; i++)
	{
		matitervect.pdata[i] = i + 1;
		sumtep = sumtep + (i + 1) * (i + 1);
	}

	for (int i = 0; i < eigveclenth; i++)
	{
		matitervect.pdata[i] = matitervect.pdata[i] / sqrt(sumtep);
	}

	//�ݷ���������
	Matrix<T> matitervectnol(1, eigveclenth);//��һ�����������

	//�������֮��Ĳ�ֵ����
	vector<double> tempvalue;

	//��ʼ�������������
	Matrix<T> matEiginout(1, eigveclenth);//��λ����

	//����������ת��
	Matrix<T> matEiginout_T(eigveclenth, 1);

	//�������ֵ����ʱת��
	Matrix<T> ladatemp(eigveclenth, 1);

	//�������ڻ�
	Matrix<T> labda(1, 1);

	//�������ֵ�����������
	for (size_t i = 0; i < iternum; i++)//��������
	{
		//���й�һ������2-������һ����λ����
		sumtep = 1e-16;
		for (int i = 0; i < matitervect.height; i++)
		{
			sumtep = sumtep + matitervect.pdata[i] * matitervect.pdata[i];
		}
		for (int j = 0; j < matitervect.height; j++)//��һ������
		{
			matitervectnol.pdata[j] = matitervect.pdata[j] / sqrt(sumtep);//��һ��
		}

		matEiginout.Matric_multi(*this, matitervectnol); //����������ˡ�
		//�������֮��Ĳ�ֵ����
		tempvalue.clear();
		for (int j = 0; j < matitervect.height; j++)
		{
			tempvalue.push_back(abs(matEiginout.pdata[j] - matitervect.pdata[j]));
		}
		double maxtempvalue = *max_element(tempvalue.begin(), tempvalue.end());

		//С��һ���ľ��Ȼ��ߴﵽ������ʱ����ʱ��������ѭ��������,�����������
		if ((maxtempvalue < tol) || (i == iternum - 1))
		{
			//��һ����������
			sumtep = 1e-16;
			for (int j = 0; j < matEiginout.height; j++)
			{
				sumtep += (matEiginout.pdata[j] * matEiginout.pdata[j]);
			}

			for (int j = 0; j < matEiginout.height; j++)
			{
				matEiginout.pdata[j] = matEiginout.pdata[j] / sqrtl(sumtep);
			}
			break;
		}
		else
		{
			//����������е���
			matitervect.Matrix_copy(matEiginout); // 
		}
	}

	//��������ֵ
	matEiginout_T.Matric_Trans(matEiginout);//ת��	
	ladatemp.Matric_multi(matEiginout_T, *this);//�������	
	labda.Matric_multi(ladatemp, matEiginout);//��������ֵ
	T eigmaxvalue = *(labda.pdata);
	vecEigvalue.push_back(eigmaxvalue);//������ֵ����ӿ�

	//��������������ӿ�
	for (int i = 0; i < eignum; i++)
	{
		for (int j = 0; j < eigveclenth; j++)
		{
			matEigvect.pdata[j * eignum + i] = matEiginout.pdata[j];
		}
	}

	int aaa = 1;
	//�������ֵ�������������
	if (eignum > 1)
	{
		Matrix<T> matiterTT(eigveclenth, eigveclenth);//���

		Matrix<T> matiterinput(this->width, this->height);//��ʼ����������
		matiterinput.Matrix_copy(*this);

		matitervect.Matrix_copy(matEiginout);//�м�ĵ���

		for (int k = 1; k < eignum; k++)
		{
			matitervect.Matrix_copy(matEiginout);
			matEiginout_T.Matric_Trans(matitervect);//��ȡת��
			matiterTT.Matric_multi(matitervect, matEiginout_T);//��ȡת�õ����(���)

			//�õ��µ�����������ڼ�����һ������ֵ����������
			for (int m = 0; m < matiterinput.height; m++)
			{
				for (int n = 0; n < matiterinput.width; n++)
				{
					matiterinput.pdata[m * matiterinput.width + n] = matiterinput.pdata[m * matiterinput.width + n] -
						eigmaxvalue * matiterTT.pdata[m * matiterinput.width + n];
				}
			}

			//�ݷ��������
			for (int i = 0; i < iternum; i++)//��������
			{
				sumtep = 1e-16;
				for (int j = 0; j < eigveclenth; j++)
				{
					sumtep = sumtep + matitervect.pdata[j] * matitervect.pdata[j];
				}

				for (int j = 0; j < eigveclenth; j++)//��һ������
				{
					matitervectnol.pdata[j] = matitervect.pdata[j] / sqrtl(sumtep);//��һ��
				}


				matEiginout.Matric_multi(matiterinput, matitervectnol);//����������ˡ�

				tempvalue.clear();
				for (int j = 0; j < eigveclenth; j++)
				{
					tempvalue.push_back(abs(matEiginout.pdata[j] - matitervect.pdata[j]));
				}
				double maxtempvalue1 = *max_element(tempvalue.begin(), tempvalue.end());

				if (maxtempvalue1 < tol || i == iternum - 1)//С��һ���ľ��Ȼ��ߴﵽ������������������ѭ��
				{
					sumtep = 1e-16;
					for (size_t m = 0; m < eigveclenth; m++)
					{
						sumtep = sumtep + matEiginout.pdata[m] * matEiginout.pdata[m];
					}

					for (size_t m = 0; m < eigveclenth; m++)
					{
						matEiginout.pdata[m] = matEiginout.pdata[m] / sqrt(sumtep);//��һ������������һ��
					}
					break;
				}
				else
				{
					matitervect.Matrix_copy(matEiginout);//��������
				}
			}

			//��������ֵ
			matEiginout_T.Matric_Trans(matEiginout);//ת��	
			ladatemp.Matric_multi(matEiginout_T, *this);//�������	
			labda.Matric_multi(ladatemp, matEiginout);//��������ֵ
			eigmaxvalue = *(labda.pdata);
			vecEigvalue.push_back(eigmaxvalue);//������ֵ����ӿ�

			//��������������ӿ�
			for (int j = 0; j < eigveclenth; j++)
			{
				matEigvect.pdata[j * eignum + k] = matEiginout.pdata[j];
			}
			//++aaa;
			//cout << "����ֵ: " << aaa << endl;
			//cout << eigmaxvalue << endl;
		}
	}
	else
	{
		//����
	}
}

//xmatΪ�������������vmat�����������������betaΪ���������hoouseholder����
template<typename T> void Matrix<T>::House_func(Matrix<T>& xmat, Matrix<T>& vmat, double& beta)
{
	int rvlength = xmat.height;

	//��vmat����ֵ
	vmat.pdata[0] = 1;
	for (int i = 1; i < rvlength; i++)
	{
		vmat.pdata[i] = xmat.pdata[i];
	}

	//�����ڻ�
	T sigmav = 0;
	for (int i = 1; i < rvlength; i++)
	{
		sigmav += xmat.pdata[i] * xmat.pdata[i];
	}

	if ((abs(sigmav - 0) < 1e-6) && (xmat.pdata[0] >= 0))
	{
		beta = 0;
	}
	else if ((abs(sigmav - 0) < 1e-6) && (xmat.pdata[0] < 0))
	{
		beta = 2;
	}
	else
	{
		double miuvalue = sqrt(xmat.pdata[0] * xmat.pdata[0] + sigmav);
		if (xmat.pdata[0] <= 0)
		{
			vmat.pdata[0] = xmat.pdata[0] - miuvalue;
		}
		else
		{
			vmat.pdata[0] = -sigmav / (xmat.pdata[0] + miuvalue);
		}
		beta = 2 * vmat.pdata[0] * vmat.pdata[0] / (vmat.pdata[0] * vmat.pdata[0] + sigmav);

		for (int i = 0; i < rvlength; i++)
		{
			vmat.pdata[i] = vmat.pdata[i] / vmat.pdata[0];
		}
	}
}

//vmatΪ�������������betaΪ�������������householder�������Pmat��Ϊ�Գ���������
template<typename T> void Matrix<T>::HouseHolder_func(Matrix<T>& vmat, Matrix<T>& Pmat, double beta)
{
	int vlength = vmat.height;
	Matrix<T> vmat_T(vlength, 1);
	Matrix<T> vmat_TT(vlength, vlength);

	//ת��
	vmat_T.Matric_Trans(vmat);
	//���
	vmat_TT.Matric_multi(vmat, vmat_T);

	//householder����,�õ��Գ���������Pmat
	for (int i = 0, step1 = 0; i < Pmat.height; i++, step1 += Pmat.width)
	{
		for (int j = 0; j < Pmat.width; j++)
		{
			if (i == j)
			{
				Pmat.pdata[step1 + j] = 1.0 - beta * vmat_TT.pdata[step1 + j];
			}
			else
			{
				Pmat.pdata[step1 + j] = 0 - beta * vmat_TT.pdata[step1 + j];
			}
		}
	}
}

//Householder ���Խǻ��������Գƾ���Amat����ȡT=Q_t*Amat*Q,T�����ԽǾ���
template<typename T> void Matrix<T>::HouseHolder_Tridiagonal(Matrix<T>& Amat, Matrix<T>& Pmat, double beta)
{
	int n = Amat.width;
	Matrix<T> xmat(1, n);

	Matrix<T> vmat(1, n);
	Matrix<T> v_Tmat(n, 1);

	Matrix<T> ppmat(1, n);

	Matrix<T> wmat(1, n);
	Matrix<T> w_Tmat(n, 1);

	Matrix<T> vwmat(n, n);
	Matrix<T> wvmat(n, n);
	for (int k = 0; k < n - 2; k++)
	{
		//����xmat
		for (int j = k + 1, step1 = 0; j < n; j++, step1 += 1)
		{
			xmat.pdata[step1] = Amat.pdata[j * n + k];

		}
		xmat.height = n - k;
		vmat.height = n - k;

		House_func(xmat, vmat, beta);//�����е�house

		ppmat.height = n - k;
		wmat.heigth = n - k;

		//��ppmat
		for (int j = k + 1, step1 = (k + 1) * Amat.width, step3 = 0; j < Amat.height; j++.step1 += Amat.width, step3 += 1)
		{
			T tempsum = 0;
			for (int q = k + 1, step2 = 0; q < Amat.width; q++, step2 += 1)
			{
				tempsum += Amat.pdata[step1 + q] * vmat.pdata[step2];
			}
			ppmat.pdata[step3] = beta * tempsum;
		}

		//��wmat
		for (int j = 0; j < wmat.height; j++)
		{
			T tempsum = 0;
			for (int q = 0; q < ppmat.height; q++)
			{
				tempsum += ppmat.pdata[q] * vmat.pdata[q];
			}
			wmat.pdata[j] = ppmat.pdata[j] - beta * tempsum * 0.5 * vmat.pdata[j];
		}

		//��A(k+1��k)��A(k��k+1)
		T sqrtsum = 0;
		for (int j = k + 1, step1 = 0; j < n; j++, step1 += 1)//��2����
		{
			sqrtsum += Amat.pdata[j * n + k] * Amat.pdata[j * n + k];
		}
		Amat.pdata[(k + 1) * n + k] = sqrtl(sqrtsum);
		Amat.pdata[k * n + k + 1] = sqrtl(sqrtsum);

		//����Amat
		vwmat.width = vmat.height;
		vwmat.height = vmat.height;

		wvmat.width = vmat.height;
		wvmat.height = vmat.height;

		v_Tmat.width = vmat.height;
		v_Tmat.Matric_Trans(vmat);

		w_Tmat.width = wmat.height;
		w_Tmat.Matric_Trans(wmat);

		vwmat.Matric_multi(vmat, w_Tmat);
		wvmat.Matric_multi(wmat, v_Tmat);

		//forѭ������ȡ�µ�A��k+1:n,k+1:n���е�Ԫ�أ�������Խǻ�
		for (int j = k + 1, step1 = (k + 1) * Amat.width, step2 = 0; j < n; j++, step1 += Amat.width, step2 += vwmat.width)
		{
			for (int q = k + 1, step3 = 0; q < n; q++, step3 += 1)
			{
				Amat.pdata[step1 + q] = Amat.pdata[step1 + q] - vwmat.pdata[step2 + step3] - wvmat.pdata[step2 + step3];
			}
		}
	}

}

/*QR�ֽⷨ�͸�˹��Ԫ��������ֵ����������
//QR�ֽⷨ��ȡȫ��������ֵ��Amat�ĶԽ��ߵ�ֵ��Ϊ����ֵ
	void Matric_QR_eigenvalue(Matrix<T>& Amat, int itertornum, double precisvalue)
	{
		Matrix<double> QMat(Amat.height, Amat.height);
		Matrix<double> RMat(Amat.width, Amat.height);

		for (size_t i = 0; i < itertornum; i++)
		{
			vector<double> digvalue1;
			vector<double> digvalue2;
			for (size_t j = 0; j < Amat.height; j++)
			{
				for (size_t k = 0; k < Amat.width; k++)
				{
					if (j == k)//�������������
					{
						digvalue1.push_back(Amat.pdata[j * Amat.width + k]);
					}
				}
			}
			//QR�ֽ�
			Amat.QR_decomposition(Amat, QMat, RMat);
			//QR�������ɵ���A����
			Amat.Matric_multi(RMat, QMat, Amat);

			//����жϱ�׼�����������ﵽ����������ֵ�仯��С����������������ѭ��
			for (size_t j = 0; j < Amat.height; j++)
			{
				for (size_t k = 0; k < Amat.width; k++)
				{
					if (j == k)//�������������
					{
						digvalue2.push_back(Amat.pdata[j * Amat.width + k]);
					}
				}
			}
			//���������Сֵ��С��ĳ��ֵ���������ֵ
			for (size_t j = 0; j < digvalue2.size(); j++)
			{
				digvalue2[j] = abs(digvalue2[j] - digvalue1[j]);
			}
			double subvalue = *max_element(digvalue2.begin(), digvalue2.end());
			cout << "��ֵ�� " << subvalue << endl;
			if (subvalue < precisvalue || i == itertornum - 1)
			{
				//��A������С�ھ��ȵ�ֵ��Ϊ0
				for (size_t j = 0; j < Amat.height * Amat.width; j++)
				{
					if (abs(Amat.pdata[j]) < precisvalue)
					{
						Amat.pdata[j] = 0;
					}
				}
				break;
			}
		}
	}

	//QR����ֽ�
	void QR_decomposition(Matrix<T>& inputmat, Matrix<T>& Qmat, Matrix<T>& Rmat)
	{
		//��ʼ�о���
		Matrix<T> umat(1, inputmat.height);

		//��ʼ�о���
		Matrix<T> umat_T(inputmat.height, 1);

		//��ʼת����˾���
		Matrix<T> hmat(inputmat.height, inputmat.height);

		//������ʼA����
		Matrix<T> Amat(inputmat);

		//����������H����
		Matrix<T> Hmat(inputmat.width, inputmat.height);
		//��λ����
		Matrix<T> Emat(inputmat.width, inputmat.height);
		Emat.Matric_Eye(Emat);

		//��������
		Matrix outmat(inputmat.width, inputmat.height);

		for (size_t i = 0; i < Amat.width - 1; i++)
		{
			//��Ϊ��λ�������
			Hmat.Matric_Eye(Hmat);

			//����
			vector<double> tempmat;
			tempmat.reserve(Amat.height);

			int dim = 0;////ά��
			double vsqrt = 0;//����
			for (size_t j = i; j < Amat.height; j++)//
			{
				T valuecur = Amat.pdata[j * Amat.width + i];
				tempmat.push_back(valuecur);//�����е�ÿһ�е�����

				vsqrt += (valuecur * valuecur);
			}

			vsqrt = sqrtl(vsqrt);//�������ݵķ���
			//if (Amat.pdata[i * Amat.width + i] < 1e-12)//С��0��ȡ��
			//{
			//	vsqrt = -vsqrt;
			//}
			//��ǰ����ά��Ϊtempmat.size����

			//��ȡ��ǰ�е�U����
			memset(umat.pdata, 0, sizeof(T) * tempmat.size());
			umat.height = tempmat.size();
			umat.width = 1;

			double  tempvalue = 0;
			for (size_t j = 0; j < tempmat.size(); j++)
			{
				if (j == 0)//�������������ʱ
				{
					umat.pdata[j] = tempmat[j] - vsqrt;//��ӷ���
				}
				else
				{
					umat.pdata[j] = tempmat[j];
				}
				tempvalue += (umat.pdata[j] * umat.pdata[j]);
			}
			if (tempvalue > 1e-12)
			{
				tempvalue = 2 / tempvalue;//��ȡU�����ķ���2/��
			}
			else
			{
				tempvalue = 1;
				//cout << "�˾���ĳ�з���Ϊ�ƽ�0��ֱ����Ϊ0" << endl;
			}

			//�����ȡH����
			//u*ut��ת��
			umat_T.width = tempmat.size();
			umat_T.height = 1;

			hmat.height = tempmat.size();
			hmat.width = tempmat.size();

			Matric_Trans(umat, umat_T);//����ת��
			Matric_multi(umat, umat_T, hmat);//��������

			for (size_t j = 0; j < hmat.width * hmat.height; j++)
			{
				hmat.pdata[j] = hmat.pdata[j] * tempvalue; //m
			}

			//��ȡh����
			for (size_t j = 0; j < hmat.height; j++)
			{
				T* hmatrow = hmat.pdata + j * hmat.width;
				for (size_t k = 0; k < hmat.width; k++)
				{
					if (j == k)
					{
						*(hmatrow + k) = 1 - *(hmatrow + k);
					}
					else
					{
						*(hmatrow + k) = 0 - *(hmatrow + k);
					}
				}
			}

			//��ȡH����
			int deta = Hmat.height - tempmat.size();
			for (int q = deta; q < Hmat.height; q++)
			{
				for (int j = deta; j < Hmat.width; j++)
				{
					Hmat.pdata[q * Hmat.width + j] = hmat.pdata[(q - deta) * hmat.width + (j - deta)];
				}
			}

			//����Q����Qmat*Hmat
			Qmat.height = Hmat.height;
			Qmat.width = Hmat.width;
			Omat_Calulte(i, Hmat, Qmat);//iΪ����

			//���»�ȡ����A����ΪR���󣩾���H*A
			Matric_multi(Hmat, Amat, outmat);
			//������A����,���е���
			Amat.Matrix_copy(outmat);
		}
		Rmat.Matrix_copy(Amat);//���յ�A�������R����

		int aa = 4;
	}

	//��������Q����
	void Omat_Calulte(int i, Matrix<T>& Hmat, Matrix<T>& Qmat)
	{
		if (i == 0)//��һ�β��丳ֵ
		{
			for (size_t j = 0; j < Hmat.width * Hmat.height; j++)
			{
				Qmat.pdata[j] = Hmat.pdata[j];
			}
		}
		else
		{
			//��ǰ�������Q����
			Matrix outmat(Hmat.width, Qmat.height);
			Matric_multi(Qmat, Hmat, outmat);
			//��Qmat���и���
			for (size_t j = 0; j < outmat.width * outmat.height; j++)
			{
				Qmat.pdata[j] = outmat.pdata[j];
			}
		}
	}
-----------------------------------------------------------------------------------
	//��������ֵeigvalue���������Է����飬��ÿ������ֵ��Ӧ����������������matric_vec������
	void Matric_vector(Matrix<T>& matric_vec, Matrix<T>& orgmat, vector<T>& eigvalue)
	{
		Matrix<T> temp(orgmat.width, orgmat.height);
		//temp.Matrix_copy(orgmat);//��������ԭʼ������ʱ������
		//ÿ�����������ж��ٵ�ά��
		int le = orgmat.width;
		T* exchangerow = new T[le]();

		for (size_t i = 0; i < eigvalue.size(); i++)//����ֵ����
		{
			T evalecur = eigvalue[i];//��ǰ����ֵ
			temp.Matrix_copy(orgmat);//ÿ�μ��㵱ǰ����ֵ��Ӧ����������ʱ����������ԭʼ������ʱ������
			//����ĶԽ���ֵ��ȥ����ֵ
			for (size_t j = 0; j < temp.height; j++)
			{
				T* temprow = temp.pdata + j * temp.width;
				for (size_t k = 0; k < temp.width; k++)
				{
					if (j == k)
					{
						*(temprow + k) -= evalecur;//�Խ���ֵ��ȥ����ֵ
					}
				}
			}
			T middvalue = 0;//�Խ����ϵ�ֵ
			//��temp��Ϊ�������ǽ��ݾ���
			for (size_t j = 0; j < temp.height - 1; j++)//ÿһ�еı���
			{
				//��J�е�j�е�ֵ
				for (size_t k = 0; k < temp.width; k++)
				{
					if (k == j)
					{
						middvalue = temp.pdata[j * temp.width + k];//�ҵ��Խ����ϵ�ֵ
					}
				}

				if (abs(middvalue) > 1e-10)//����Խ��ߵĵ�ǰ����ֵֵ������
				{
					for (size_t k = j; k < temp.width; k++)
					{
						temp.pdata[j * temp.width + k] = temp.pdata[j * temp.width + k] / middvalue;//��j�еĵ�j�е���ֵ��һ����������ͬ����
					}

					//��j��֮����е���ֵ����0
					for (size_t k = j + 1; k < temp.height; k++)
					{
						middvalue = temp.pdata[k * temp.width + j];
						for (size_t q = j; q < temp.width; q++)
						{
							temp.pdata[k * temp.width + q] -=
								(middvalue * temp.pdata[j * temp.width + q]);//
						}
					}
				}
				else//��֮��������������Ϊ����У�Ȼ������н���,�ٽ�������
				{
					int tempmiddvalue = 0;
					for (size_t m = j; m < temp.height; m++)
					{
						tempmiddvalue = temp.pdata[m * temp.width + j];
						if (abs(tempmiddvalue) > 1e-10)//����ҵ��˾���ֵ����0����
						{
							middvalue = tempmiddvalue;
							//ִ���н���
							for (size_t q = 0; q < temp.width; q++)
							{
								exchangerow[q] = temp.pdata[m * temp.width + q];
								temp.pdata[m * temp.width + q] = temp.pdata[j * temp.width + q];
								temp.pdata[j * temp.width + q] = exchangerow[q];
							}

							//�������ٽ�������
							int indexrow = j * temp.width;
							for (size_t k = j; k < temp.width; k++)
							{
								temp.pdata[indexrow + k] = temp.pdata[indexrow + k] / middvalue;//��j�еĵ�j�е���ֵ��һ����������ͬ����
							}

							//��j��֮����е���ֵ����0
							for (size_t k = j + 1; k < temp.height; k++)
							{
								middvalue = temp.pdata[k * temp.width + j];
								for (size_t q = j; q < temp.width; q++)
								{
									temp.pdata[k * temp.width + q] = temp.pdata[k * temp.width + q]
										- middvalue * temp.pdata[j * temp.width + q];//
								}
							}
							//�������������ǰѭ��
							break;
						}
					}
				}

			}//����������ǽ��ݾ���Ļ���

			matric_vec.pdata[(matric_vec.height - 1) * matric_vec.width + i] = 1;//�������ǰ�е����·���Ϊ1���������������·�ֵΪ1
			middvalue = 1;
			//����������,�����·���ʼ��������������ÿ��ֵ��
			T sum = 0;
			for (int j = temp.height - 2; j >= 0; --j)//�ӵ����ڶ��п�ʼ���ϱ�������
			{
				T tempsum = 0;
				for (int k = j + 1; k < temp.width; k++)//���д��е���һ�����ݿ�ʼ��������
				{
					tempsum += temp.pdata[j * temp.width + k] * matric_vec.pdata[k * matric_vec.width + i];
				}
				tempsum = 0 - (tempsum / (temp.pdata[j * temp.width + j] + 1e-16));
				middvalue += tempsum * tempsum;
				matric_vec.pdata[j * matric_vec.width + i] = tempsum;
			}

			middvalue = sqrtl(middvalue + 1e-16);
			if (middvalue > 1e-12)
			{
				//���������Ĺ�һ��
				for (int j = 0; j < matric_vec.height; j++)
				{
					matric_vec.pdata[j * matric_vec.width + i] = matric_vec.pdata[j * matric_vec.width + i] / (middvalue + 1e-16);
				}
			}
			else
			{
				//ԭ�ⲻ��
				for (int j = 0; j < matric_vec.height; j++)
				{
					matric_vec.pdata[j * matric_vec.width + i] = matric_vec.pdata[j * matric_vec.width + i] / 1.0;
				}
			}
		}
	}



*/