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

//定义矩阵模板
template<typename T> class Matrix
{

public:

	/*************************************************
	Function: 	    	// Matrix()
	Description: 		// 无参构造函数，指针指向空
	Input:        	// 无
	Output: 		// 无
	Others: 		// 无
	*************************************************/
	Matrix() : width(0), height(0), pdata(nullptr), internal_data(false) {}

	/*************************************************
	Function: 	    	// Matrix()
	Description: 		// 有参构造函数，赋初数组首地址，宽，高初始值，相当于创建了浅拷贝对象，将原来的数组给到矩阵对象中
	Input:        	// 无
	Output: 		// 无
	Others: 		// 相当于浅拷贝
	*************************************************/
	Matrix(void* data, int width, int height) : pdata((T*)data), width(width), height(height), internal_data(false) {}

	/*************************************************
	Function: 	    	// Matrix()
	Description: 		// 有参构造函数，给出宽高，创建矩阵数组，申请内存空间
	Input:        	// 无
	Output: 		// 无
	Others: 		// 申请内存空间
	*************************************************/
	Matrix(int width, int height) : width(width), height(height), internal_data(true) { pdata = new(std::nothrow) T[width * height](); } //  

	/*************************************************
	Function: 	    	// Matrix_copy
	Description: 		// 复制矩阵，深拷贝
	Input:        	// 无
	Output: 		// 无
	Others: 		// 无
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
	Description: 		// 赋值操作，深拷贝。拷贝的对象为传进去的m对象
	Input:        	// 无
	Output: 		// 无
	Others: 		// 将传进去的矩阵对象m，拷贝到当前对象中
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
	Description: 		// 生成单位矩阵
	Input:        	// intputmatrix：输入矩阵；
	Output: 		// intputmatrix：将输入矩阵转变成单位矩阵
	Others: 		// 单位矩阵对角线为1，其他元素为0
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
	Description: 		// 析构函数，是否矩阵数组空间
	Input:        	// 无
	Output: 		// 无
	Others: 		// 无
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
	Description: 		// 析构函数，是否矩阵数组空间
	Input:        	// 无
	Output: 		// 无
	Others: 		// 无
	*************************************************/
	void SetPixel(int x, int y, T pixelValue) { pdata[y * width + x] = pixelValue; }

	/*************************************************
	Function: 	    	// GetPixel
	Description: 		// 给定数组的X和Y坐标，获取此坐标下的数值
	Input:        	// 无
	Output: 		// 无
	Others: 		// 无
	*************************************************/
	T GetPixel(int x, int y) { return pdata[y * width + x]; }

	/*************************************************
	Function: 	    	// Matrix_length
	Description: 		// 获取此对象的数组长度
	Input:        	// 无
	Output: 		// 无
	Others: 		// 无
	*************************************************/
	int Matrix_length()
	{
		return  width * height;
	}

	/*************************************************
	Function: 	    	// Matric_multi
	Description: 		// 矩阵相乘
	Input:        	// leftmatrix：左乘矩阵；rightmatrix：右乘矩阵；
	Output: 		// outmatrix乘法输出矩阵
	Others: 		// 无
	*************************************************/
	void Matric_multi(Matrix<T>& leftmatrix, Matrix<T>& rightmatrix)
	{
		//对右矩阵进行转置
		Matrix<T> rightmatrix_T(rightmatrix.height, rightmatrix.width);
		rightmatrix_T.Matric_Trans(rightmatrix);//转置

#pragma omp parallel for
		for (int i = 0; i < leftmatrix.height; i++)//左侧矩阵的行数相当于输出矩阵的行数
		{
			T* mat1row = leftmatrix.pdata + i * leftmatrix.width;//第一个矩阵的行首
			T* outmatrow = pdata + i * width;//输出矩阵的行首
			for (int j = 0; j < rightmatrix.width; j++)//右侧第二个矩阵的列数相当于输出矩阵的列数
			{
				T* mat2row = rightmatrix_T.pdata + j * leftmatrix.width;//第二个矩阵（用转置后的矩阵）的列首				
				T sumvalue = 0;
				for (int k = 0; k < leftmatrix.width; k++)
				{
					T mat1pos = *(mat1row + k);
					T mat2pos = *(mat2row + k);
					sumvalue = sumvalue + mat1pos * mat2pos;
				}

				*(outmatrow + j) = sumvalue;//输出矩阵i行，j列的的数值				
			}
		}

		//rightmatrix_T.~rightmatrix_T();
	}

	/*************************************************
	Function: 	    	// Matric_Trans
	Description: 		// 矩阵转置
	Input:        	// intputmatrix：输入矩阵；
	Output: 		// 输出转置矩阵
	Others: 		// 转置矩阵输出矩阵需要和输入矩阵大小相匹配
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

			for (int i = 0, step1 = 0; i < height; i++, step1 += width)//输出的每一行
			{
				for (int j = 0, step2 = 0; j < width; j++, step2 += height)//对输出行每一行的进行遍历
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
	Description: 		// 生成单位矩阵
	Input:        	// intputmatrix：输入矩阵；
	Output: 		// intputmatrix：将输入矩阵转变成单位矩阵
	Others: 		// 单位矩阵对角线为1，其他元素为0
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
	Description: 		// 幂法迭代求解最大特征值和特征向量
	Input:        	// orgmat：需要求的矩阵；
	Output: 		// vectout：矩阵的最大特征向量；eigmaxvalue矩阵的最大成分特征值
	Others: 		// 幂法迭代只能求最大特征值和特征向量，但是在一定条件下可以求解所有特征值和特征向量
	*************************************************/
	void Matric_powervector(Matrix<T>& matEigvect, vector<T>& vecEigvalue);

	void House_func(Matrix<T>& xmat, Matrix<T>& vmat, double& beta);

	void HouseHolder_func(Matrix<T>& vmat, Matrix<T>& Pmat, double beta);

	void HouseHolder_Tridiagonal(Matrix<T>& Amat, Matrix<T>& Pmat, double beta);
};

//迭代循环求取其他的特征值和特征向量，
template<typename T> void Matrix<T>::Matric_powervector(Matrix<T>& matEigvect, vector<T>& vecEigvalue)
{
	//初始设置参数
	double tol = 1e-10;//精度设置
	int iternum = 100;//迭代次数设置
	double sumtep = 1e-16;//根号下的微小值
	//所需的特征值数目（以及特征向量数目）
	int eignum = matEigvect.width;

	//特征向量的长度
	int eigveclenth = this->width;
	// 
	//构建初始归一化2-范数迭代向量，并作为中间迭代参数
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

	//幂法迭代计算
	Matrix<T> matitervectnol(1, eigveclenth);//归一化的向量结果

	//计算迭代之间的差值精度
	vector<double> tempvalue;

	//初始输出的特征向量
	Matrix<T> matEiginout(1, eigveclenth);//首位迭代

	//特征向量的转置
	Matrix<T> matEiginout_T(eigveclenth, 1);

	//求解特征值的临时转换
	Matrix<T> ladatemp(eigveclenth, 1);

	//向量的内积
	Matrix<T> labda(1, 1);

	//最大特征值特征向量求解
	for (size_t i = 0; i < iternum; i++)//迭代次数
	{
		//进行归一化操作2-范数归一化单位向量
		sumtep = 1e-16;
		for (int i = 0; i < matitervect.height; i++)
		{
			sumtep = sumtep + matitervect.pdata[i] * matitervect.pdata[i];
		}
		for (int j = 0; j < matitervect.height; j++)//归一化处理
		{
			matitervectnol.pdata[j] = matitervect.pdata[j] / sqrt(sumtep);//归一化
		}

		matEiginout.Matric_multi(*this, matitervectnol); //迭代矩阵相乘。
		//计算迭代之间的差值精度
		tempvalue.clear();
		for (int j = 0; j < matitervect.height; j++)
		{
			tempvalue.push_back(abs(matEiginout.pdata[j] - matitervect.pdata[j]));
		}
		double maxtempvalue = *max_element(tempvalue.begin(), tempvalue.end());

		//小于一定的精度或者达到迭代此时，此时跳出迭代循环输出结果,否则继续迭代
		if ((maxtempvalue < tol) || (i == iternum - 1))
		{
			//归一化特征向量
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
			//拷贝结果进行迭代
			matitervect.Matrix_copy(matEiginout); // 
		}
	}

	//计算特征值
	matEiginout_T.Matric_Trans(matEiginout);//转置	
	ladatemp.Matric_multi(matEiginout_T, *this);//计算外积	
	labda.Matric_multi(ladatemp, matEiginout);//计算特征值
	T eigmaxvalue = *(labda.pdata);
	vecEigvalue.push_back(eigmaxvalue);//将特征值纳入接口

	//将特征向量纳入接口
	for (int i = 0; i < eignum; i++)
	{
		for (int j = 0; j < eigveclenth; j++)
		{
			matEigvect.pdata[j * eignum + i] = matEiginout.pdata[j];
		}
	}

	int aaa = 1;
	//多个特征值特征向量的求解
	if (eignum > 1)
	{
		Matrix<T> matiterTT(eigveclenth, eigveclenth);//外积

		Matrix<T> matiterinput(this->width, this->height);//初始化迭代矩阵
		matiterinput.Matrix_copy(*this);

		matitervect.Matrix_copy(matEiginout);//中间的迭代

		for (int k = 1; k < eignum; k++)
		{
			matitervect.Matrix_copy(matEiginout);
			matEiginout_T.Matric_Trans(matitervect);//获取转置
			matiterTT.Matric_multi(matitervect, matEiginout_T);//获取转置的相乘(外积)

			//得到新的输入矩阵用于计算下一个特征值和特征向量
			for (int m = 0; m < matiterinput.height; m++)
			{
				for (int n = 0; n < matiterinput.width; n++)
				{
					matiterinput.pdata[m * matiterinput.width + n] = matiterinput.pdata[m * matiterinput.width + n] -
						eigmaxvalue * matiterTT.pdata[m * matiterinput.width + n];
				}
			}

			//幂法迭代求解
			for (int i = 0; i < iternum; i++)//迭代次数
			{
				sumtep = 1e-16;
				for (int j = 0; j < eigveclenth; j++)
				{
					sumtep = sumtep + matitervect.pdata[j] * matitervect.pdata[j];
				}

				for (int j = 0; j < eigveclenth; j++)//归一化处理
				{
					matitervectnol.pdata[j] = matitervect.pdata[j] / sqrtl(sumtep);//归一化
				}


				matEiginout.Matric_multi(matiterinput, matitervectnol);//迭代矩阵相乘。

				tempvalue.clear();
				for (int j = 0; j < eigveclenth; j++)
				{
					tempvalue.push_back(abs(matEiginout.pdata[j] - matitervect.pdata[j]));
				}
				double maxtempvalue1 = *max_element(tempvalue.begin(), tempvalue.end());

				if (maxtempvalue1 < tol || i == iternum - 1)//小于一定的精度或者达到迭代次数，跳出迭代循环
				{
					sumtep = 1e-16;
					for (size_t m = 0; m < eigveclenth; m++)
					{
						sumtep = sumtep + matEiginout.pdata[m] * matEiginout.pdata[m];
					}

					for (size_t m = 0; m < eigveclenth; m++)
					{
						matEiginout.pdata[m] = matEiginout.pdata[m] / sqrt(sumtep);//上一个特征向量归一化
					}
					break;
				}
				else
				{
					matitervect.Matrix_copy(matEiginout);//继续迭代
				}
			}

			//计算特征值
			matEiginout_T.Matric_Trans(matEiginout);//转置	
			ladatemp.Matric_multi(matEiginout_T, *this);//计算外积	
			labda.Matric_multi(ladatemp, matEiginout);//计算特征值
			eigmaxvalue = *(labda.pdata);
			vecEigvalue.push_back(eigmaxvalue);//将特征值纳入接口

			//将特征向量纳入接口
			for (int j = 0; j < eigveclenth; j++)
			{
				matEigvect.pdata[j * eignum + k] = matEiginout.pdata[j];
			}
			//++aaa;
			//cout << "特征值: " << aaa << endl;
			//cout << eigmaxvalue << endl;
		}
	}
	else
	{
		//结束
	}
}

//xmat为输入的列向量，vmat（输出的列向量）和beta为输出，计算hoouseholder向量
template<typename T> void Matrix<T>::House_func(Matrix<T>& xmat, Matrix<T>& vmat, double& beta)
{
	int rvlength = xmat.height;

	//给vmat赋初值
	vmat.pdata[0] = 1;
	for (int i = 1; i < rvlength; i++)
	{
		vmat.pdata[i] = xmat.pdata[i];
	}

	//计算内积
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

//vmat为输入的列向量，beta为输入参数，计算householder反射矩阵Pmat（为对称正交矩阵）
template<typename T> void Matrix<T>::HouseHolder_func(Matrix<T>& vmat, Matrix<T>& Pmat, double beta)
{
	int vlength = vmat.height;
	Matrix<T> vmat_T(vlength, 1);
	Matrix<T> vmat_TT(vlength, vlength);

	//转置
	vmat_T.Matric_Trans(vmat);
	//外积
	vmat_TT.Matric_multi(vmat, vmat_T);

	//householder反射,得到对称正交矩阵Pmat
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

//Householder 三对角化，给定对称矩阵Amat，获取T=Q_t*Amat*Q,T是三对角矩阵
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
		//构建xmat
		for (int j = k + 1, step1 = 0; j < n; j++, step1 += 1)
		{
			xmat.pdata[step1] = Amat.pdata[j * n + k];

		}
		xmat.height = n - k;
		vmat.height = n - k;

		House_func(xmat, vmat, beta);//计算列的house

		ppmat.height = n - k;
		wmat.heigth = n - k;

		//求ppmat
		for (int j = k + 1, step1 = (k + 1) * Amat.width, step3 = 0; j < Amat.height; j++.step1 += Amat.width, step3 += 1)
		{
			T tempsum = 0;
			for (int q = k + 1, step2 = 0; q < Amat.width; q++, step2 += 1)
			{
				tempsum += Amat.pdata[step1 + q] * vmat.pdata[step2];
			}
			ppmat.pdata[step3] = beta * tempsum;
		}

		//求wmat
		for (int j = 0; j < wmat.height; j++)
		{
			T tempsum = 0;
			for (int q = 0; q < ppmat.height; q++)
			{
				tempsum += ppmat.pdata[q] * vmat.pdata[q];
			}
			wmat.pdata[j] = ppmat.pdata[j] - beta * tempsum * 0.5 * vmat.pdata[j];
		}

		//求A(k+1，k)和A(k，k+1)
		T sqrtsum = 0;
		for (int j = k + 1, step1 = 0; j < n; j++, step1 += 1)//求2范数
		{
			sqrtsum += Amat.pdata[j * n + k] * Amat.pdata[j * n + k];
		}
		Amat.pdata[(k + 1) * n + k] = sqrtl(sqrtsum);
		Amat.pdata[k * n + k + 1] = sqrtl(sqrtsum);

		//跟新Amat
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

		//for循环，获取新的A（k+1:n,k+1:n）中的元素，完成三对角化
		for (int j = k + 1, step1 = (k + 1) * Amat.width, step2 = 0; j < n; j++, step1 += Amat.width, step2 += vwmat.width)
		{
			for (int q = k + 1, step3 = 0; q < n; q++, step3 += 1)
			{
				Amat.pdata[step1 + q] = Amat.pdata[step1 + q] - vwmat.pdata[step2 + step3] - wvmat.pdata[step2 + step3];
			}
		}
	}

}

/*QR分解法和高斯消元根据特征值求特征向量
//QR分解法求取全部的特征值，Amat的对角线的值即为特征值
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
					if (j == k)//行数和列数相等
					{
						digvalue1.push_back(Amat.pdata[j * Amat.width + k]);
					}
				}
			}
			//QR分解
			Amat.QR_decomposition(Amat, QMat, RMat);
			//QR迭代生成迭代A矩阵
			Amat.Matric_multi(RMat, QMat, Amat);

			//添加判断标准：迭代次数达到最大或者特征值变化极小两个条件即可跳出循环
			for (size_t j = 0; j < Amat.height; j++)
			{
				for (size_t k = 0; k < Amat.width; k++)
				{
					if (j == k)//行数和列数相等
					{
						digvalue2.push_back(Amat.pdata[j * Amat.width + k]);
					}
				}
			}
			//二者相差最小值，小于某个值，输出特征值
			for (size_t j = 0; j < digvalue2.size(); j++)
			{
				digvalue2[j] = abs(digvalue2[j] - digvalue1[j]);
			}
			double subvalue = *max_element(digvalue2.begin(), digvalue2.end());
			cout << "差值： " << subvalue << endl;
			if (subvalue < precisvalue || i == itertornum - 1)
			{
				//将A矩阵中小于精度的值置为0
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

	//QR矩阵分解
	void QR_decomposition(Matrix<T>& inputmat, Matrix<T>& Qmat, Matrix<T>& Rmat)
	{
		//初始列矩阵
		Matrix<T> umat(1, inputmat.height);

		//初始行矩阵
		Matrix<T> umat_T(inputmat.height, 1);

		//初始转置相乘矩阵
		Matrix<T> hmat(inputmat.height, inputmat.height);

		//迭代初始A矩阵
		Matrix<T> Amat(inputmat);

		//迭代的正交H矩阵
		Matrix<T> Hmat(inputmat.width, inputmat.height);
		//单位矩阵
		Matrix<T> Emat(inputmat.width, inputmat.height);
		Emat.Matric_Eye(Emat);

		//迭代矩阵
		Matrix outmat(inputmat.width, inputmat.height);

		for (size_t i = 0; i < Amat.width - 1; i++)
		{
			//置为单位矩阵矩阵
			Hmat.Matric_Eye(Hmat);

			//置零
			vector<double> tempmat;
			tempmat.reserve(Amat.height);

			int dim = 0;////维度
			double vsqrt = 0;//范数
			for (size_t j = i; j < Amat.height; j++)//
			{
				T valuecur = Amat.pdata[j * Amat.width + i];
				tempmat.push_back(valuecur);//矩阵中的每一列的数据

				vsqrt += (valuecur * valuecur);
			}

			vsqrt = sqrtl(vsqrt);//当列数据的范数
			//if (Amat.pdata[i * Amat.width + i] < 1e-12)//小于0，取反
			//{
			//	vsqrt = -vsqrt;
			//}
			//当前数据维度为tempmat.size（）

			//获取当前列的U向量
			memset(umat.pdata, 0, sizeof(T) * tempmat.size());
			umat.height = tempmat.size();
			umat.width = 1;

			double  tempvalue = 0;
			for (size_t j = 0; j < tempmat.size(); j++)
			{
				if (j == 0)//列数与行数相等时
				{
					umat.pdata[j] = tempmat[j] - vsqrt;//添加范数
				}
				else
				{
					umat.pdata[j] = tempmat[j];
				}
				tempvalue += (umat.pdata[j] * umat.pdata[j]);
			}
			if (tempvalue > 1e-12)
			{
				tempvalue = 2 / tempvalue;//获取U向量的范数2/ρ
			}
			else
			{
				tempvalue = 1;
				//cout << "此矩阵某列范数为逼近0，直接置为0" << endl;
			}

			//计算获取H矩阵
			//u*ut的转置
			umat_T.width = tempmat.size();
			umat_T.height = 1;

			hmat.height = tempmat.size();
			hmat.width = tempmat.size();

			Matric_Trans(umat, umat_T);//矩阵转置
			Matric_multi(umat, umat_T, hmat);//矩阵的相乘

			for (size_t j = 0; j < hmat.width * hmat.height; j++)
			{
				hmat.pdata[j] = hmat.pdata[j] * tempvalue; //m
			}

			//获取h矩阵
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

			//获取H矩阵
			int deta = Hmat.height - tempmat.size();
			for (int q = deta; q < Hmat.height; q++)
			{
				for (int j = deta; j < Hmat.width; j++)
				{
					Hmat.pdata[q * Hmat.width + j] = hmat.pdata[(q - deta) * hmat.width + (j - deta)];
				}
			}

			//计算Q矩阵Qmat*Hmat
			Qmat.height = Hmat.height;
			Qmat.width = Hmat.width;
			Omat_Calulte(i, Hmat, Qmat);//i为次数

			//重新获取迭代A（即为R矩阵）矩阵H*A
			Matric_multi(Hmat, Amat, outmat);
			//拷贝给A矩阵,进行迭代
			Amat.Matrix_copy(outmat);
		}
		Rmat.Matrix_copy(Amat);//最终的A矩阵给到R矩阵

		int aa = 4;
	}

	//迭代生成Q矩阵
	void Omat_Calulte(int i, Matrix<T>& Hmat, Matrix<T>& Qmat)
	{
		if (i == 0)//第一次不变赋值
		{
			for (size_t j = 0; j < Hmat.width * Hmat.height; j++)
			{
				Qmat.pdata[j] = Hmat.pdata[j];
			}
		}
		else
		{
			//当前计算出的Q矩阵
			Matrix outmat(Hmat.width, Qmat.height);
			Matric_multi(Qmat, Hmat, outmat);
			//将Qmat进行覆盖
			for (size_t j = 0; j < outmat.width * outmat.height; j++)
			{
				Qmat.pdata[j] = outmat.pdata[j];
			}
		}
	}
-----------------------------------------------------------------------------------
	//根据特征值eigvalue求解齐次线性方程组，求每个特征值对应的特征向量，存在matric_vec矩阵中
	void Matric_vector(Matrix<T>& matric_vec, Matrix<T>& orgmat, vector<T>& eigvalue)
	{
		Matrix<T> temp(orgmat.width, orgmat.height);
		//temp.Matrix_copy(orgmat);//完整复制原始矩阵到临时矩阵中
		//每个特征向量有多少的维度
		int le = orgmat.width;
		T* exchangerow = new T[le]();

		for (size_t i = 0; i < eigvalue.size(); i++)//特征值遍历
		{
			T evalecur = eigvalue[i];//当前特征值
			temp.Matrix_copy(orgmat);//每次计算当前特征值对应的特征向量时，完整复制原始矩阵到临时矩阵中
			//矩阵的对角线值减去特征值
			for (size_t j = 0; j < temp.height; j++)
			{
				T* temprow = temp.pdata + j * temp.width;
				for (size_t k = 0; k < temp.width; k++)
				{
					if (j == k)
					{
						*(temprow + k) -= evalecur;//对角线值减去特征值
					}
				}
			}
			T middvalue = 0;//对角线上的值
			//将temp化为右上三角阶梯矩阵
			for (size_t j = 0; j < temp.height - 1; j++)//每一行的遍历
			{
				//第J行第j列的值
				for (size_t k = 0; k < temp.width; k++)
				{
					if (k == j)
					{
						middvalue = temp.pdata[j * temp.width + k];//找到对角线上的值
					}
				}

				if (abs(middvalue) > 1e-10)//如果对角线的当前绝对值值大于零
				{
					for (size_t k = j; k < temp.width; k++)
					{
						temp.pdata[j * temp.width + k] = temp.pdata[j * temp.width + k] / middvalue;//第j行的第j列的数值归一化，后续相同除数
					}

					//第j行之后的行的数值置于0
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
				else//反之，先向下搜索不为零的行，然后进行行交换,再进行运算
				{
					int tempmiddvalue = 0;
					for (size_t m = j; m < temp.height; m++)
					{
						tempmiddvalue = temp.pdata[m * temp.width + j];
						if (abs(tempmiddvalue) > 1e-10)//如果找到了绝对值大于0的行
						{
							middvalue = tempmiddvalue;
							//执行行交换
							for (size_t q = 0; q < temp.width; q++)
							{
								exchangerow[q] = temp.pdata[m * temp.width + q];
								temp.pdata[m * temp.width + q] = temp.pdata[j * temp.width + q];
								temp.pdata[j * temp.width + q] = exchangerow[q];
							}

							//交换后再进行运算
							int indexrow = j * temp.width;
							for (size_t k = j; k < temp.width; k++)
							{
								temp.pdata[indexrow + k] = temp.pdata[indexrow + k] / middvalue;//第j行的第j列的数值归一化，后续相同除数
							}

							//第j行之后的行的数值置于0
							for (size_t k = j + 1; k < temp.height; k++)
							{
								middvalue = temp.pdata[k * temp.width + j];
								for (size_t q = j; q < temp.width; q++)
								{
									temp.pdata[k * temp.width + q] = temp.pdata[k * temp.width + q]
										- middvalue * temp.pdata[j * temp.width + q];//
								}
							}
							//运算完后跳出当前循环
							break;
						}
					}
				}

			}//完成右上三角阶梯矩阵的化简

			matric_vec.pdata[(matric_vec.height - 1) * matric_vec.width + i] = 1;//输出矩阵当前列的最下方置为1，特征向量的最下方值为1
			middvalue = 1;
			//求特征向量,从最下方开始计算特征向量的每个值。
			T sum = 0;
			for (int j = temp.height - 2; j >= 0; --j)//从倒数第二行开始向上遍历计算
			{
				T tempsum = 0;
				for (int k = j + 1; k < temp.width; k++)//此行此列的下一个数据开始遍历计算
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
				//特征向量的归一化
				for (int j = 0; j < matric_vec.height; j++)
				{
					matric_vec.pdata[j * matric_vec.width + i] = matric_vec.pdata[j * matric_vec.width + i] / (middvalue + 1e-16);
				}
			}
			else
			{
				//原封不动
				for (int j = 0; j < matric_vec.height; j++)
				{
					matric_vec.pdata[j * matric_vec.width + i] = matric_vec.pdata[j * matric_vec.width + i] / 1.0;
				}
			}
		}
	}



*/