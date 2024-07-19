#include "Unetseg_lung.h"

//构造函数
Unetseg_lung::Unetseg_lung()
{

	//定义模型和加载模型
	if (torch::cuda::is_available() && torch::cuda::cudnn_is_available())//可以用cuda进行运算
	{
		//此处最好能加载日志，可观测使用的是CPU还是GPU模型
		this->m_device = torch::kCUDA;//用GPU进行模型的运算
	}
	else
	{
		//此处最好能加载日志，可观测使用的是CPU还是GPU模型
		this->m_device = torch::kCPU;//只可用cpu进行模型计算
	}

	//std::cout << "cudu support:" << (torch::cuda::is_available() ? "ture" : "false") << std::endl;
	//std::cout << "cudnn support:" << (torch::cuda::cudnn_is_available() ? "ture" : "false") << std::endl;
	//加载模型
	this->m_module = torch::jit::load("G:\\Yangyingjian\\tianqi\\pth\\segnet\\260.pt", this->m_device);
}

//析构函数
Unetseg_lung::~Unetseg_lung()
{
}


void Unetseg_lung::Model_main(unsigned char* pImagechar, int params[],  unsigned char* presult_img,
	int& downsampwidth, int& downsampheight)
{

	//-----------------------肺功能模块用例开始，输入16位数组地址pintimgchar-----------------------
	CTRUnet_Detection CTRUnet_Detectionobj;
	cv::Mat imgPNG8UC1_Size512(512, 512, CV_8UC1);//转换成512*512
	Imgformat_Process(pImagechar, params, imgPNG8UC1_Size512);//格式位数转换和预处理		

	cv::Mat result_img(512, 512, CV_8UC1);//生成512*512的掩膜0和1，可以输出掩模图
	Unetseg_lung_proimg(imgPNG8UC1_Size512, result_img);//开始分割，调用模型肺野分割处理	
	//-----------------------完成肺野分割，为下一步功能实现做基础,输出接口result_img

	//Unet深度学习分割后的接口转换，生成肺功能各个函数的输入接口。
	CTRUnet_Detectionobj.Lungmask_clean(result_img);//对分割好的肺野区域进行连通域清理
	downsampwidth = result_img.cols;
	downsampheight = result_img.rows;

    //肺功能函数的输入地址
	for (int i = 0; i < downsampwidth*downsampheight; i++)
	{
		presult_img[i] = ((unsigned char*)result_img.data)[i];
	}
}

void Unetseg_lung::Imgformat_Process(unsigned char* pintimgchar, int params[], Mat& imgPNG8UC1_Size512)
{
	//数据格式转换和预处理
	int imgwidth=params[0];
	int imgheight = params[1];
	int bitdeph = params[2];

	int widthheight = imgwidth * imgheight;
	Img_Process Img_Processobj;//
	unsigned short* pArrimg = new unsigned short[imgwidth * imgheight]();//16位输入图像

	//位数转换
	if (bitdeph==16 || bitdeph == 12)//如果是16或者12位，格式转换赋值
	{
		for (size_t i = 0; i < imgwidth * imgheight; i++)
		{
			pArrimg[i] = ((unsigned short*)pintimgchar)[i];
		}
	}
	else // 如果是8位
	{
		for (size_t i = 0; i < imgwidth * imgheight; i++)
		{
			pArrimg[i] = float(pintimgchar[i])/255.0*65535.0;
		}
	}
	
	unsigned char* pmap255 = new(std::nothrow) unsigned char[widthheight]();//8位输出图像地址

	//如果输入的是处理后图像，则是否图像反白再进行分割
	//Img_Processobj.Isinverimg(pArrimg, pmap255, imgwidth, imgheight);

	//如果输入原始图像，则原始图像进行预处理再进行分割
	Img_Processobj.Proc_main(pArrimg, imgwidth, imgheight, pmap255);


	//图像的转换成mat型
	cv::Mat imgmat0(imgheight, imgwidth, CV_8UC1, pmap255);
	cv::resize(imgmat0, imgPNG8UC1_Size512, Size(512, 512), 0, 0, cv::INTER_LINEAR);//将image的大小resize至(512, 512)的Mat格式

	//cv::imshow("分割前显示图像", imgPNG8UC1_Size512);
	//waitKey(0);

	delete[] pArrimg;
	pArrimg = nullptr;

	delete[] pmap255;
	pmap255 = nullptr;

}

void Unetseg_lung::Unetseg_lung_proimg(Mat& imgPNG8UC1_Size512, Mat& result_img)
{	
	int channel = 1;
	//开始调用Unet进行分割
	torch::Tensor tensor_image = torch::from_blob(imgPNG8UC1_Size512.data, { 1, imgPNG8UC1_Size512.rows, imgPNG8UC1_Size512.cols, channel }, torch::kByte);//杨英健20230913，将cv::Mat 转成Tensor
	tensor_image = tensor_image.permute({ 0, 3, 1, 2 });
	tensor_image = tensor_image.toType(torch::kFloat);
	tensor_image = tensor_image.div(255);

	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(tensor_image.to(this->m_device));

	//前向计算
	torch::Tensor output = this->m_module.forward(inputs).toTensor();
	torch::Tensor output_max = output.argmax(1);

	//tensor to Mat  
	output_max = output_max.squeeze();
	output_max = output_max.mul(255).to(torch::kU8);
	output_max = output_max.to(torch::kCPU);
	memcpy((void*)result_img.data, output_max.data_ptr(), sizeof(torch::kU8) * output_max.numel());
}



void Unetseg_lung::img_show(unsigned char* pintimgchar, int params[], unsigned short* pdiaph_Line_imgorg, unsigned short* pleftrightlung_imgmaskorg)
{
	int imgwidth = params[0];
	int imgheight = params[1];

	unsigned char* pmap255 = new(std::nothrow) unsigned char[imgwidth*imgheight]();//8位输出图像地址
	unsigned short* pinp = (unsigned short*)pintimgchar;

	int imglength = imgwidth * imgheight;
	double minvalue = *min_element(pinp, pinp + imglength);
	double maxvalue = *max_element(pinp, pinp + imglength);

	for (int i = 0; i < imglength; i++)
	{
		double midtempvalue = (pinp[i] - minvalue) / (maxvalue - minvalue);
		double resultdl = 255 * midtempvalue;
		pmap255[i] = unsigned char(resultdl + 0.5);
	}

	cv::Mat imgmat4(imgwidth, imgheight,CV_8UC1, pmap255);// (imgPNG8UC3_Size512.rows, imgPNG8UC3_Size512.cols, CV_8UC1, imgPNG8UC3_Size512.pdata);
	cv::Mat garay4;
	cv::cvtColor(imgmat4, garay4, cv::COLOR_GRAY2BGR);//  图像转成RBG做标记显示

	//cv::Point label1(params[0], params[1]);//右膈肌点
    //cv::Point label2(params[2], params[3]);//右膈肌点

	cv::Scalar colorrr1(0, 255, 0);
	cv::Scalar colorrr2(0, 0, 255);//绿色标点---标的点
	int poinsize = 10;
	int thickness = -5;

    cv::Point label3(params[4], params[3]);//左膈肌点
	cv::Point label4(params[6], params[5]);//左膈肌点
	cv::Point label5(params[8], params[7]);//右心点
	cv::Point label6(params[10], params[9]);//左心点
	cv::Point label7(params[12], params[11]);//右肋骨边缘
	cv::Point label8(params[14], params[13]);//左肋骨边缘            

	cv::Point label9(params[16], params[15]);//右测心点
	cv::Point label10(params[18], params[17]);//左测心点
	cv::Point label11(params[20], params[19]);//右测肋骨边缘   
	cv::Point label12(params[22], params[21]);//左测肋骨边缘
	cv::Point label13(params[24], params[23]);//左侧低点  
	cv::Point label14(params[26], params[25]);//右侧低点

	cv::circle(garay4, label3, poinsize, colorrr1, thickness);//左膈肌绘制标记
	cv::circle(garay4, label4, poinsize, colorrr1, thickness);//左膈肌绘制标记
	cv::circle(garay4, label5, poinsize, colorrr1, thickness);//右心绘制标记
	cv::circle(garay4, label6, poinsize, colorrr1, thickness);//左心绘制标记
	cv::circle(garay4, label7, poinsize, colorrr1, thickness);//右心绘制标记
	cv::circle(garay4, label8, poinsize, colorrr1, thickness);//左心绘制标记
	
	cv::circle(garay4, label9, poinsize, colorrr1, thickness);//右心测标记
	cv::circle(garay4, label10, poinsize, colorrr1, thickness);//左心测标记
	cv::circle(garay4, label11, poinsize, colorrr1, thickness);//右测肋骨标记
	cv::circle(garay4, label12, poinsize, colorrr1, thickness);//左心绘制标记
	cv::circle(garay4, label13, poinsize, colorrr1, thickness);//右测肋骨标记
	cv::circle(garay4, label14, poinsize, colorrr1, thickness);//左心绘制标记

	cv::namedWindow("各个点的显示1", cv::WINDOW_FREERATIO);
	cv::imshow("各个点的显示1", garay4);
	cv::waitKey(0);
	//-------------------------横膈膜边界显示--------------------------------------------------
	cv::Mat garay5 = garay4;
	vector<cv::Point> leftmaxpoint;
	vector<cv::Point> rightmaxpoint;
	leftmaxpoint.clear();
	rightmaxpoint.clear();

	cv::Point ppoint;
	for (int i=0;i<imgheight;i++)
	{
		for (int j = 0; j < imgwidth; j++)
		{

			if (pdiaph_Line_imgorg[i*imgwidth + j] == 128)
			{
				ppoint.x = j;
				ppoint.y = i;
				leftmaxpoint.push_back(ppoint);
			}
			else if (pdiaph_Line_imgorg[i*imgwidth + j] == 255)
			{
				ppoint.x = j;
				ppoint.y = i;
				rightmaxpoint.push_back(ppoint);
			}

		}
	}

	for (int i = 0; i < leftmaxpoint.size(); i++)
	{
		cv::circle(garay5, leftmaxpoint[i], 3, colorrr2, -3);//图像显示线段标记。
	}
	for (int i = 0; i < rightmaxpoint.size(); i++)
	{
		cv::circle(garay5, rightmaxpoint[i], 3, colorrr2, -3);//图像显示线段标记。
	}
	
	cv::namedWindow("各个线的显示", cv::WINDOW_FREERATIO);	
	cv::imshow("各个线的显示", garay5);
	cv::waitKey(0);

	cv::Mat imgmat(imgwidth, imgheight, CV_16UC1, pleftrightlung_imgmaskorg);	
	cv::Mat garay;
	//cv::normalize(imgmat12, garay12, 0, 1,cv::NORM_MINMAX);//归一化   
	cv::threshold(imgmat, garay, 0.5,65535 , cv::THRESH_BINARY);//二值化
	cv::namedWindow("左右肺显示", cv::WINDOW_FREERATIO);
	cv::imshow("左右肺显示", garay);
	cv::waitKey(0);
}
