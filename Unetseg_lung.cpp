#include "Unetseg_lung.h"

//���캯��
Unetseg_lung::Unetseg_lung()
{

	//����ģ�ͺͼ���ģ��
	if (torch::cuda::is_available() && torch::cuda::cudnn_is_available())//������cuda��������
	{
		//�˴�����ܼ�����־���ɹ۲�ʹ�õ���CPU����GPUģ��
		this->m_device = torch::kCUDA;//��GPU����ģ�͵�����
	}
	else
	{
		//�˴�����ܼ�����־���ɹ۲�ʹ�õ���CPU����GPUģ��
		this->m_device = torch::kCPU;//ֻ����cpu����ģ�ͼ���
	}

	//std::cout << "cudu support:" << (torch::cuda::is_available() ? "ture" : "false") << std::endl;
	//std::cout << "cudnn support:" << (torch::cuda::cudnn_is_available() ? "ture" : "false") << std::endl;
	//����ģ��
	this->m_module = torch::jit::load("G:\\Yangyingjian\\tianqi\\pth\\segnet\\260.pt", this->m_device);
}

//��������
Unetseg_lung::~Unetseg_lung()
{
}


void Unetseg_lung::Model_main(unsigned char* pImagechar, int params[],  unsigned char* presult_img,
	int& downsampwidth, int& downsampheight)
{

	//-----------------------�ι���ģ��������ʼ������16λ�����ַpintimgchar-----------------------
	CTRUnet_Detection CTRUnet_Detectionobj;
	cv::Mat imgPNG8UC1_Size512(512, 512, CV_8UC1);//ת����512*512
	Imgformat_Process(pImagechar, params, imgPNG8UC1_Size512);//��ʽλ��ת����Ԥ����		

	cv::Mat result_img(512, 512, CV_8UC1);//����512*512����Ĥ0��1�����������ģͼ
	Unetseg_lung_proimg(imgPNG8UC1_Size512, result_img);//��ʼ�ָ����ģ�ͷ�Ұ�ָ��	
	//-----------------------��ɷ�Ұ�ָΪ��һ������ʵ��������,����ӿ�result_img

	//Unet���ѧϰ�ָ��Ľӿ�ת�������ɷι��ܸ�������������ӿڡ�
	CTRUnet_Detectionobj.Lungmask_clean(result_img);//�Էָ�õķ�Ұ���������ͨ������
	downsampwidth = result_img.cols;
	downsampheight = result_img.rows;

    //�ι��ܺ����������ַ
	for (int i = 0; i < downsampwidth*downsampheight; i++)
	{
		presult_img[i] = ((unsigned char*)result_img.data)[i];
	}
}

void Unetseg_lung::Imgformat_Process(unsigned char* pintimgchar, int params[], Mat& imgPNG8UC1_Size512)
{
	//���ݸ�ʽת����Ԥ����
	int imgwidth=params[0];
	int imgheight = params[1];
	int bitdeph = params[2];

	int widthheight = imgwidth * imgheight;
	Img_Process Img_Processobj;//
	unsigned short* pArrimg = new unsigned short[imgwidth * imgheight]();//16λ����ͼ��

	//λ��ת��
	if (bitdeph==16 || bitdeph == 12)//�����16����12λ����ʽת����ֵ
	{
		for (size_t i = 0; i < imgwidth * imgheight; i++)
		{
			pArrimg[i] = ((unsigned short*)pintimgchar)[i];
		}
	}
	else // �����8λ
	{
		for (size_t i = 0; i < imgwidth * imgheight; i++)
		{
			pArrimg[i] = float(pintimgchar[i])/255.0*65535.0;
		}
	}
	
	unsigned char* pmap255 = new(std::nothrow) unsigned char[widthheight]();//8λ���ͼ���ַ

	//���������Ǵ����ͼ�����Ƿ�ͼ�񷴰��ٽ��зָ�
	//Img_Processobj.Isinverimg(pArrimg, pmap255, imgwidth, imgheight);

	//�������ԭʼͼ����ԭʼͼ�����Ԥ�����ٽ��зָ�
	Img_Processobj.Proc_main(pArrimg, imgwidth, imgheight, pmap255);


	//ͼ���ת����mat��
	cv::Mat imgmat0(imgheight, imgwidth, CV_8UC1, pmap255);
	cv::resize(imgmat0, imgPNG8UC1_Size512, Size(512, 512), 0, 0, cv::INTER_LINEAR);//��image�Ĵ�Сresize��(512, 512)��Mat��ʽ

	//cv::imshow("�ָ�ǰ��ʾͼ��", imgPNG8UC1_Size512);
	//waitKey(0);

	delete[] pArrimg;
	pArrimg = nullptr;

	delete[] pmap255;
	pmap255 = nullptr;

}

void Unetseg_lung::Unetseg_lung_proimg(Mat& imgPNG8UC1_Size512, Mat& result_img)
{	
	int channel = 1;
	//��ʼ����Unet���зָ�
	torch::Tensor tensor_image = torch::from_blob(imgPNG8UC1_Size512.data, { 1, imgPNG8UC1_Size512.rows, imgPNG8UC1_Size512.cols, channel }, torch::kByte);//��Ӣ��20230913����cv::Mat ת��Tensor
	tensor_image = tensor_image.permute({ 0, 3, 1, 2 });
	tensor_image = tensor_image.toType(torch::kFloat);
	tensor_image = tensor_image.div(255);

	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(tensor_image.to(this->m_device));

	//ǰ�����
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

	unsigned char* pmap255 = new(std::nothrow) unsigned char[imgwidth*imgheight]();//8λ���ͼ���ַ
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
	cv::cvtColor(imgmat4, garay4, cv::COLOR_GRAY2BGR);//  ͼ��ת��RBG�������ʾ

	//cv::Point label1(params[0], params[1]);//��������
    //cv::Point label2(params[2], params[3]);//��������

	cv::Scalar colorrr1(0, 255, 0);
	cv::Scalar colorrr2(0, 0, 255);//��ɫ���---��ĵ�
	int poinsize = 10;
	int thickness = -5;

    cv::Point label3(params[4], params[3]);//��������
	cv::Point label4(params[6], params[5]);//��������
	cv::Point label5(params[8], params[7]);//���ĵ�
	cv::Point label6(params[10], params[9]);//���ĵ�
	cv::Point label7(params[12], params[11]);//���߹Ǳ�Ե
	cv::Point label8(params[14], params[13]);//���߹Ǳ�Ե            

	cv::Point label9(params[16], params[15]);//�Ҳ��ĵ�
	cv::Point label10(params[18], params[17]);//����ĵ�
	cv::Point label11(params[20], params[19]);//�Ҳ��߹Ǳ�Ե   
	cv::Point label12(params[22], params[21]);//����߹Ǳ�Ե
	cv::Point label13(params[24], params[23]);//���͵�  
	cv::Point label14(params[26], params[25]);//�Ҳ�͵�

	cv::circle(garay4, label3, poinsize, colorrr1, thickness);//���������Ʊ��
	cv::circle(garay4, label4, poinsize, colorrr1, thickness);//���������Ʊ��
	cv::circle(garay4, label5, poinsize, colorrr1, thickness);//���Ļ��Ʊ��
	cv::circle(garay4, label6, poinsize, colorrr1, thickness);//���Ļ��Ʊ��
	cv::circle(garay4, label7, poinsize, colorrr1, thickness);//���Ļ��Ʊ��
	cv::circle(garay4, label8, poinsize, colorrr1, thickness);//���Ļ��Ʊ��
	
	cv::circle(garay4, label9, poinsize, colorrr1, thickness);//���Ĳ���
	cv::circle(garay4, label10, poinsize, colorrr1, thickness);//���Ĳ���
	cv::circle(garay4, label11, poinsize, colorrr1, thickness);//�Ҳ��߹Ǳ��
	cv::circle(garay4, label12, poinsize, colorrr1, thickness);//���Ļ��Ʊ��
	cv::circle(garay4, label13, poinsize, colorrr1, thickness);//�Ҳ��߹Ǳ��
	cv::circle(garay4, label14, poinsize, colorrr1, thickness);//���Ļ��Ʊ��

	cv::namedWindow("���������ʾ1", cv::WINDOW_FREERATIO);
	cv::imshow("���������ʾ1", garay4);
	cv::waitKey(0);
	//-------------------------����Ĥ�߽���ʾ--------------------------------------------------
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
		cv::circle(garay5, leftmaxpoint[i], 3, colorrr2, -3);//ͼ����ʾ�߶α�ǡ�
	}
	for (int i = 0; i < rightmaxpoint.size(); i++)
	{
		cv::circle(garay5, rightmaxpoint[i], 3, colorrr2, -3);//ͼ����ʾ�߶α�ǡ�
	}
	
	cv::namedWindow("�����ߵ���ʾ", cv::WINDOW_FREERATIO);	
	cv::imshow("�����ߵ���ʾ", garay5);
	cv::waitKey(0);

	cv::Mat imgmat(imgwidth, imgheight, CV_16UC1, pleftrightlung_imgmaskorg);	
	cv::Mat garay;
	//cv::normalize(imgmat12, garay12, 0, 1,cv::NORM_MINMAX);//��һ��   
	cv::threshold(imgmat, garay, 0.5,65535 , cv::THRESH_BINARY);//��ֵ��
	cv::namedWindow("���ҷ���ʾ", cv::WINDOW_FREERATIO);
	cv::imshow("���ҷ���ʾ", garay);
	cv::waitKey(0);
}
