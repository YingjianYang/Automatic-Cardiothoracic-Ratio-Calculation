// 此cpp用于代码的整合的主函数，包含
#pragma once

#include"Unetseg_lung.h"
#include"Dicomfile_readwrite.h"
#include <chrono>
//#include<opencv2/opencv.hpp>

using namespace cv;//yangyingjian,用关键字namespace来定义命名空间cv
using namespace std;

/*
	数组参数定义
	pImagechar:		输入胸片正位原始图像
	params[0]:	输入图像的宽
	params[1]:	输入图像的高
	params[2]:	输入图像的位数

	params[3]:	输出左侧胸腔的Y坐标
	params[4]:	输出左侧胸腔的X坐标
	params[5]:	输出右侧胸腔的Y坐标
	params[6]:	输出右侧胸腔的X坐标

	params[7]:	输出左侧心脏的Y坐标
	params[8]:	输出左侧心脏的X坐标
	params[9]:	输出右侧心脏的Y坐标
	params[10]:	输出右侧心脏的X坐标

	params[11]: 输出左侧横膈肌Y坐标
	params[12]: 输出左侧横膈肌X坐标
	params[13]: 输出右侧横膈肌Y坐标
	params[14]: 输出右侧横膈肌X坐标

	params[15]: 输出左侧拐点y坐标
	params[16]: 输出左侧拐点x坐标
	params[17]: 输出右侧拐点Y坐标
	params[18]: 输出右侧拐点x坐标

	params[19]: 输出左侧肺野顶点Y坐标
	params[20]: 输出左侧肺野顶点x坐标
	params[21]: 输出右侧肺野顶点Y坐标
	params[22]: 输出右侧肺野顶点x坐标

	params[19]: 输出左侧肺野顶点Y坐标
	params[20]: 输出左侧肺野顶点x坐标
	params[21]: 输出右侧肺野顶点Y坐标
	params[22]: 输出右侧肺野顶点x坐标

	params[23]: 输出左侧肺野底点Y坐标
	params[24]: 输出左侧肺野底点x坐标
	params[25]: 输出右侧肺野底点Y坐标
	params[26]: 输出右侧肺野底点x坐标

	params[27]: 输出左侧图像肺面积
	params[28]: 输出右侧图像左肺面积
	params[29]: 判断左右肺肺野的位置，值为1时，左侧的即为右肺，右侧的即为左肺；值为2时左侧的即为左肺，右侧的即为右肺

	输出横膈肌边界接口  pdiaph_Line_imgorg数组接口: //横膈肌边界线，左侧图像的膈肌128，右侧图像的膈肌255
	输出左右肺识别掩模接口 pleftrightlung_imgmaskorg数组接口: //左右肺识别输出，输出掩模中右肺1，左肺2

	return:		1:执行成功;	0:执行失败;

*/

int main()
{
	int params[30];

	int sig = 0;
	int imgwidth = 3072;//宽度
	int imgheight = 3072;//高度
	int bitwidth = 16;//位数
	int widthheight = imgheight * imgwidth; //图像数组长度
	Dicomfile_readwrite Dcm_readwriteobj;
	Unetseg_lung Unetseg_lungobj;//从构造函数中加载深度学习模型，并可以输出日志是否调用GPU
	CTRUnet_Detection CTRUnet_Detectionobj;
	unsigned short* pArrimg = new(std::nothrow) unsigned short[widthheight]();//  创建动态数组存储输入数据  //   

	//读取文件夹中所有图像
	string filePath_original = "G:\\Yangyingjian\\MatlabToC\\LUNG_ventilation_data\\original";
	//string write512path = "G:\\Train_images\\Chest_images_train";
	string format = ".dcm";
	vector<string> files_original;
	Dcm_readwriteobj.GetAllFormatFiles(filePath_original, files_original, format);//读取filePath_mask路径下所有dcm格式对应的原始DR图像
	int size_original = files_original.size();//所有dcm格式对应的原始DR图像对应的数目
	string readpathlist;

	//加载深度学习模型
	//输出膈肌和左右肺数组接口设置 
	unsigned short* pdiaph_Line_imgorg = new unsigned short[imgwidth*imgheight]();//横膈肌边界线，左侧图像的膈肌128，右侧图像的膈肌255
	unsigned short* pleftrightlung_imgmaskorg = new unsigned short[imgwidth*imgheight]();//左右肺识别，输出掩模中右肺1，左肺2

	//遍历每张图像
	for (size_t i = 0; i < size_original; i++)//files_original.size()
	{
		readpathlist = files_original[i];
		cout << "文件名称" << files_original[i].c_str() << endl;
		string strpath;
		strpath.append(filePath_original).append("\\").append(readpathlist);//原始图像的读取路径
		//  wstring wp = s2ws(strpath);// string转宽字符串wstring   
		cout << "整个路径名称" << strpath.c_str() << endl;
		Dcm_readwriteobj.ReadSingleFormFiles(strpath, pArrimg, imgwidth, imgheight);//读取dcm图像打开此语句

		//赋值获取宽高位数
		params[0] = imgwidth;
		params[1] = imgheight;
		params[2] = 16;
		int signum = 1;//  功能返回值，值为1运行成功，值为0运行失败
		unsigned char* pImagechar = (unsigned char*)&pArrimg[0];//  读取dcm，作为输入数组的地址; 

		//----------------肺功能开启运算，计算模型获取下采样后的肺野掩模图--------
		int downsampwidth = 3072;
		int downsampheight = 3072;
		unsigned char* presult_img = new unsigned char[downsampwidth*downsampheight]();//肺功能函数的输入地址和下采样的宽高downsampwidth downsampheight
		//获取时间开始节点
		auto start = std::chrono::high_resolution_clock::now();
		Unetseg_lungobj.Model_main(pImagechar, params, presult_img, downsampwidth, downsampheight);//
		// 获取结束时间点
		auto end = std::chrono::high_resolution_clock::now();
		// 计算持续时间并转换为毫秒
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		// 输出运行时间
		std::cout << "Code segment took " << duration.count() << " milliseconds." << std::endl;


		cv::Mat imgmat(512, 512, CV_8UC1, presult_img);
		cv::namedWindow("multiwindow", cv::WINDOW_FREERATIO);
		cv::Mat garay;
		//cv::normalize(imgmat12, garay12, 0, 1,cv::NORM_MINMAX);//归一化   
		cv::threshold(imgmat, garay, 0, 255, cv::THRESH_BINARY);//二值化
		cv::imshow("multiwindow", garay);
		cv::waitKey(0);

		//心胸比以及肺功能各个特征点的检测和计算，输出params数组
		start = std::chrono::high_resolution_clock::now();
		signum = 1;
		signum = CTRUnet_Detectionobj.CTR_main(presult_img, downsampwidth, downsampheight , params);//预留512*512肺野掩膜输出窗口，右肺（面积大）为1，左肺（面积小）为2，最下方代为映射原始图码放入函数内即可
		if (signum != 1)
		{
			//可以加日志
			return 0;
		}
		end = std::chrono::high_resolution_clock::now();
		// 计算持续时间并转换为毫秒
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		// 输出运行时间
		std::cout << "Code segment took " << duration.count() << " milliseconds." << std::endl;

		//------------------------------------------图像的显示-----------------------------------------///
		float ri = abs(float(params[10] - params[8] + 1)) / abs(float(params[6] - params[4] + 1));
		cout << "心胸比的数值为：" << ri << endl;

		//膈肌点的检测，输出pdiaph_Line_img的膈肌边界图，左侧图像的膈肌像素值为128，右侧图像的膈肌像素值为255
		signum = 1;
		signum = CTRUnet_Detectionobj.Diaphragm_detect(presult_img, downsampwidth, downsampheight, params, pdiaph_Line_imgorg);
		if (signum != 1)
		{
			//可以加日志
			return 0;
		}

		//左右肺的识别检测和面积计算，输出左右肺掩模pleftrightlung_imgmask中右肺1，左肺2；
		signum = 1;
		signum = CTRUnet_Detectionobj.Lung_Areacalculate(presult_img, downsampwidth, downsampheight, pleftrightlung_imgmaskorg, params);// 左右肺识别和面积计算
		if (signum != 1)
		{
			//可以加日志
			return 0;
		}

		//效果测试显示
		//Unetseg_lungobj.img_show(pImagechar, params, pdiaph_Line_imgorg, pleftrightlung_imgmaskorg); //                    

		delete[] presult_img;
		presult_img = nullptr;
	}
	getchar();
	return 1;
}