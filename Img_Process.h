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
    Description: 		// ͼ��Ĺ�һ��������ͼ��Ҷȷ�Χ������01֮��
    Input:        	// matInputimg������ͼ��
    Output: 		// matOutputimg�����ͼ��
    Others: 		// ��һ�����̺�������Ǹ����͹�һ��ͼ��
    *************************************************/
    void Normalize_func(double* pInputimg, double* pOutputimg, int width, int height); //��һ��   

    /*************************************************
    Function: 	    	// Imadjust_func
    Description: 		// ͼ��Աȶ���ǿ������matlab��imadjust����
    Input:        	// matInputimg������ͼ��uplimit��downlimit�ֱ�Ϊ�Աȶ���ǿ��������
    Output: 		// matOutputimg�����ͼ��
    Others: 		// ��һ�����̺�������Ǹ����͹�һ��ͼ��
    *************************************************/
    void Imadjust_func(double* pInputimg, double downlimit, double uplimit, double* pOutputimg, int width, int height);//�Աȶȵ���

    /*************************************************
    Function: 	    	// AdaptIogImage
    Description: 		// ͼ������Ӧ�����任
    Input:        	// matDownsample:����ͼ��matSublogimag��matL3Ϊ�м�������ӿռ�
    Output: 		// matLogimg: ����Ӧ�����任��������ͼ��
    Others: 		// ����Ӧ�����任�������ͼ���Ϊ�����ͣ�ƫ�ò���bias����Ϊ0.85.
    *************************************************/
    void AdaptIogImage(double* pDownsample, double* pSublogimag, double* pL3, double* pLogimg, int width, int height);

    /*************************************************
    Function: 	    	// Mapto255
    Description: 		// ͼ������Ӧ�����任
    Input:        	// matDownsample:����ͼ��matSublogimag��matL3Ϊ�м�������ӿռ�
    Output: 		// matLogimg: ����Ӧ�����任��������ͼ��
    Others: 		// ����Ӧ�����任�������ͼ���Ϊ�����ͣ�ƫ�ò���bias����Ϊ0.85.
    *************************************************/
    void Mapto255(double* pInputimg, unsigned char* pOutputimg, int width, int height);

	/*************************************************
	Function: 	    	// DownSample
	Description: 		// ͼ������Ӧ�����任
	Input:        	// matDownsample:����ͼ��matSublogimag��matL3Ϊ�м�������ӿռ�
	Output: 		// matLogimg: ����Ӧ�����任��������ͼ��
	Others: 		// ����Ӧ�����任�������ͼ���Ϊ�����ͣ�ƫ�ò���bias����Ϊ0.85.
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

