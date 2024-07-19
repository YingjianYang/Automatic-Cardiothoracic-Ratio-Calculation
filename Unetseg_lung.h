#pragma once
#include <torch/script.h> // One-stop header. 
#include <torch/torch.h>
//#include <iostream> 
//#include <fstream>
//#include <string>
#include <opencv2\opencv.hpp> 
#include <opencv2\imgproc\types_c.h> 

#include "Img_Process.h"
#include"CTRUnet_Detection.h"
#include"Matrix_calculate.h"

using namespace std;
using namespace cv;//yangyingjian,�ùؼ���namespace�����������ռ�cv

class Unetseg_lung
{

public:

	Unetseg_lung();//���캯��

	~Unetseg_lung();//��������

	// 
	/*************************************************
	Function: 	    	// Imgformat_Process��
	Description: 		// �ָ����ĸ�ʽת��
	Input:        	// pintimgchar������ģ�������ַ��bitdephΪ���������λ����imgwidth��imgheightΪ��������Ŀ��
	Output: 		// imgPNG8UC1_Size512Ϊ��������mat������
	Others: 		//
	*************************************************/
	void Model_main(unsigned char* pImagechar, int params[], unsigned char* presult_img,  int& downsampwidth, int& downsampheight);

	//����ģ��
	/*************************************************
	Function: 	    	// Unetseg_lung_proimg��
	Description: 		// ����ģ�ͽ�����ģ�ķָ�
	Input:        	// pintimgchar������ģ�������ַ��bitdephΪ���������λ����imgwidth��imgheightΪ��������Ŀ��
	Output: 		// imgPNG8UC1_Size512Ϊ��������mat������
	Others: 		//
	*************************************************/
	void Unetseg_lung_proimg(Mat& imgPNG8UC1_Size512, Mat& result_img);

    //��ʽת��
	/*************************************************
	Function: 	    	// Imgformat_Process��
	Description: 		// �ָ����ĸ�ʽת��
	Input:        	// pintimgchar������ģ�������ַ��bitdephΪ���������λ����imgwidth��imgheightΪ��������Ŀ��
	Output: 		// imgPNG8UC1_Size512Ϊ��������mat������
	Others: 		//
	*************************************************/
	void Imgformat_Process(unsigned char* pintimgchar, int params[], Mat& imgPNG8UC1_Size512);

	//opencv������ʾ
	void img_show(unsigned char* pintimgchar, int params[], unsigned short* pdiaph_Line_imgorg, unsigned short* pleftrightlung_imgmaskorg);



private:

	torch::DeviceType m_device;
	torch::jit::script::Module m_module;

};



