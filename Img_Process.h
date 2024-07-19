#pragma once
#include<algorithm>
#include<math.h>
#include<numeric>
#include<string>
#include<vector>
#include<map>
#include <io.h>  
#include"Matrix_calculate.h"

using namespace std;

class Img_Process
{
public:


	void GetAllFormatFiles(string path, vector<string>& files, string format);

    void Proc_main(unsigned short* pArrimg, int imgwidth, int imgheight, unsigned char* pmap255);

	/*************************************************
    Function: 	    	// Normalize_func
    Description: 		// 图像的归一化操作，图像灰度范围限制在01之间
    Input:        	// matInputimg：输入图像；
    Output: 		// matOutputimg：输出图像
    Others: 		// 归一化过程后输出的是浮点型归一化图像
    *************************************************/
    void Normalize_func(double* pInputimg, double* pOutputimg, int width, int height); //归一化   

    /*************************************************
    Function: 	    	// Imadjust_func
    Description: 		// 图像对比度增强，类似matlab中imadjust函数
    Input:        	// matInputimg：输入图像，uplimit和downlimit分别为对比度增强的上下限
    Output: 		// matOutputimg：输出图像
    Others: 		// 归一化过程后输出的是浮点型归一化图像
    *************************************************/
    void Imadjust_func(double* pInputimg, double downlimit, double uplimit, double* pOutputimg, int width, int height);//对比度调节

    /*************************************************
    Function: 	    	// AdaptIogImage
    Description: 		// 图像自适应对数变换
    Input:        	// matDownsample:输入图像，matSublogimag和matL3为中间变量的子空间
    Output: 		// matLogimg: 自适应对数变换数组的输出图像
    Others: 		// 自适应对数变换输入输出图像皆为浮点型，偏置参数bias设置为0.85.
    *************************************************/
    void AdaptIogImage(double* pDownsample, double* pSublogimag, double* pL3, double* pLogimg, int width, int height);

    /*************************************************
    Function: 	    	// Mapto255
    Description: 		// 图像自适应对数变换
    Input:        	// matDownsample:输入图像，matSublogimag和matL3为中间变量的子空间
    Output: 		// matLogimg: 自适应对数变换数组的输出图像
    Others: 		// 自适应对数变换输入输出图像皆为浮点型，偏置参数bias设置为0.85.
    *************************************************/
    void Mapto255(double* pInputimg, unsigned char* pOutputimg, int width, int height);

	/*************************************************
	Function: 	    	// DownSample
	Description: 		// 图像自适应对数变换
	Input:        	// matDownsample:输入图像，matSublogimag和matL3为中间变量的子空间
	Output: 		// matLogimg: 自适应对数变换数组的输出图像
	Others: 		// 自适应对数变换输入输出图像皆为浮点型，偏置参数bias设置为0.85.
	*************************************************/
    void DownSample(double* pGauresult, double* pdownsample, int ratio, int width, int height,
        int sample_wid, int sample_heigh);

    void Hist_Equaliation(double* pInputimg, double* pOutputimg, int levelnumber, int width, int height);

	void Gaussiantemplatefunc(double* matGastemplate, int wide, double tho);

	void Templatefilter(double* matInputimg, int width, int height, double* matTemplate,
		int matTemplatewidth, int matTemplateheight, double* matOutputimg);

	void Conectchose(Matrix<unsigned short>& matInputimg, Matrix<unsigned short>& matLabelimg, vector<int>& labval, vector<int>& labind, bool TF);
    void Isinverimg(unsigned short* pArrimg, unsigned char* pmap255,int imgwidth, int imgheight);
	
};

