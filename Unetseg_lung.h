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
using namespace cv;//yangyingjian,用关键字namespace来定义命名空间cv

class Unetseg_lung
{

public:

	Unetseg_lung();//构造函数

	~Unetseg_lung();//析构函数

	// 
	/*************************************************
	Function: 	    	// Imgformat_Process。
	Description: 		// 分割结果的格式转换
	Input:        	// pintimgchar输入掩模的数组地址；bitdeph为输入数组的位数；imgwidth和imgheight为输入数组的宽高
	Output: 		// imgPNG8UC1_Size512为输出数组的mat型数据
	Others: 		//
	*************************************************/
	void Model_main(unsigned char* pImagechar, int params[], unsigned char* presult_img,  int& downsampwidth, int& downsampheight);

	//调用模型
	/*************************************************
	Function: 	    	// Unetseg_lung_proimg。
	Description: 		// 调用模型进行掩模的分割
	Input:        	// pintimgchar输入掩模的数组地址；bitdeph为输入数组的位数；imgwidth和imgheight为输入数组的宽高
	Output: 		// imgPNG8UC1_Size512为输出数组的mat型数据
	Others: 		//
	*************************************************/
	void Unetseg_lung_proimg(Mat& imgPNG8UC1_Size512, Mat& result_img);

    //格式转换
	/*************************************************
	Function: 	    	// Imgformat_Process。
	Description: 		// 分割结果的格式转换
	Input:        	// pintimgchar输入掩模的数组地址；bitdeph为输入数组的位数；imgwidth和imgheight为输入数组的宽高
	Output: 		// imgPNG8UC1_Size512为输出数组的mat型数据
	Others: 		//
	*************************************************/
	void Imgformat_Process(unsigned char* pintimgchar, int params[], Mat& imgPNG8UC1_Size512);

	//opencv测试显示
	void img_show(unsigned char* pintimgchar, int params[], unsigned short* pdiaph_Line_imgorg, unsigned short* pleftrightlung_imgmaskorg);



private:

	torch::DeviceType m_device;
	torch::jit::script::Module m_module;

};



