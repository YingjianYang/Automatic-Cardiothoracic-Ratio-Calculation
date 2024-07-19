// ��cpp���ڴ�������ϵ�������������
#pragma once

#include"Unetseg_lung.h"
#include"Dicomfile_readwrite.h"
#include <chrono>
//#include<opencv2/opencv.hpp>

using namespace cv;//yangyingjian,�ùؼ���namespace�����������ռ�cv
using namespace std;

/*
	�����������
	pImagechar:		������Ƭ��λԭʼͼ��
	params[0]:	����ͼ��Ŀ�
	params[1]:	����ͼ��ĸ�
	params[2]:	����ͼ���λ��

	params[3]:	��������ǻ��Y����
	params[4]:	��������ǻ��X����
	params[5]:	����Ҳ���ǻ��Y����
	params[6]:	����Ҳ���ǻ��X����

	params[7]:	�����������Y����
	params[8]:	�����������X����
	params[9]:	����Ҳ������Y����
	params[10]:	����Ҳ������X����

	params[11]: �����������Y����
	params[12]: �����������X����
	params[13]: ����Ҳ������Y����
	params[14]: ����Ҳ������X����

	params[15]: ������յ�y����
	params[16]: ������յ�x����
	params[17]: ����Ҳ�յ�Y����
	params[18]: ����Ҳ�յ�x����

	params[19]: �������Ұ����Y����
	params[20]: �������Ұ����x����
	params[21]: ����Ҳ��Ұ����Y����
	params[22]: ����Ҳ��Ұ����x����

	params[19]: �������Ұ����Y����
	params[20]: �������Ұ����x����
	params[21]: ����Ҳ��Ұ����Y����
	params[22]: ����Ҳ��Ұ����x����

	params[23]: �������Ұ�׵�Y����
	params[24]: �������Ұ�׵�x����
	params[25]: ����Ҳ��Ұ�׵�Y����
	params[26]: ����Ҳ��Ұ�׵�x����

	params[27]: ������ͼ������
	params[28]: ����Ҳ�ͼ��������
	params[29]: �ж����ҷη�Ұ��λ�ã�ֵΪ1ʱ�����ļ�Ϊ�ҷΣ��Ҳ�ļ�Ϊ��Σ�ֵΪ2ʱ���ļ�Ϊ��Σ��Ҳ�ļ�Ϊ�ҷ�

	����������߽�ӿ�  pdiaph_Line_imgorg����ӿ�: //�������߽��ߣ����ͼ�������128���Ҳ�ͼ�������255
	������ҷ�ʶ����ģ�ӿ� pleftrightlung_imgmaskorg����ӿ�: //���ҷ�ʶ������������ģ���ҷ�1�����2

	return:		1:ִ�гɹ�;	0:ִ��ʧ��;

*/

int main()
{
	int params[30];

	int sig = 0;
	int imgwidth = 3072;//���
	int imgheight = 3072;//�߶�
	int bitwidth = 16;//λ��
	int widthheight = imgheight * imgwidth; //ͼ�����鳤��
	Dicomfile_readwrite Dcm_readwriteobj;
	Unetseg_lung Unetseg_lungobj;//�ӹ��캯���м������ѧϰģ�ͣ������������־�Ƿ����GPU
	CTRUnet_Detection CTRUnet_Detectionobj;
	unsigned short* pArrimg = new(std::nothrow) unsigned short[widthheight]();//  ������̬����洢��������  //   

	//��ȡ�ļ���������ͼ��
	string filePath_original = "G:\\Yangyingjian\\MatlabToC\\LUNG_ventilation_data\\original";
	//string write512path = "G:\\Train_images\\Chest_images_train";
	string format = ".dcm";
	vector<string> files_original;
	Dcm_readwriteobj.GetAllFormatFiles(filePath_original, files_original, format);//��ȡfilePath_mask·��������dcm��ʽ��Ӧ��ԭʼDRͼ��
	int size_original = files_original.size();//����dcm��ʽ��Ӧ��ԭʼDRͼ���Ӧ����Ŀ
	string readpathlist;

	//�������ѧϰģ��
	//������������ҷ�����ӿ����� 
	unsigned short* pdiaph_Line_imgorg = new unsigned short[imgwidth*imgheight]();//�������߽��ߣ����ͼ�������128���Ҳ�ͼ�������255
	unsigned short* pleftrightlung_imgmaskorg = new unsigned short[imgwidth*imgheight]();//���ҷ�ʶ�������ģ���ҷ�1�����2

	//����ÿ��ͼ��
	for (size_t i = 0; i < size_original; i++)//files_original.size()
	{
		readpathlist = files_original[i];
		cout << "�ļ�����" << files_original[i].c_str() << endl;
		string strpath;
		strpath.append(filePath_original).append("\\").append(readpathlist);//ԭʼͼ��Ķ�ȡ·��
		//  wstring wp = s2ws(strpath);// stringת���ַ���wstring   
		cout << "����·������" << strpath.c_str() << endl;
		Dcm_readwriteobj.ReadSingleFormFiles(strpath, pArrimg, imgwidth, imgheight);//��ȡdcmͼ��򿪴����

		//��ֵ��ȡ���λ��
		params[0] = imgwidth;
		params[1] = imgheight;
		params[2] = 16;
		int signum = 1;//  ���ܷ���ֵ��ֵΪ1���гɹ���ֵΪ0����ʧ��
		unsigned char* pImagechar = (unsigned char*)&pArrimg[0];//  ��ȡdcm����Ϊ��������ĵ�ַ; 

		//----------------�ι��ܿ������㣬����ģ�ͻ�ȡ�²�����ķ�Ұ��ģͼ--------
		int downsampwidth = 3072;
		int downsampheight = 3072;
		unsigned char* presult_img = new unsigned char[downsampwidth*downsampheight]();//�ι��ܺ����������ַ���²����Ŀ��downsampwidth downsampheight
		//��ȡʱ�俪ʼ�ڵ�
		auto start = std::chrono::high_resolution_clock::now();
		Unetseg_lungobj.Model_main(pImagechar, params, presult_img, downsampwidth, downsampheight);//
		// ��ȡ����ʱ���
		auto end = std::chrono::high_resolution_clock::now();
		// �������ʱ�䲢ת��Ϊ����
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		// �������ʱ��
		std::cout << "Code segment took " << duration.count() << " milliseconds." << std::endl;


		cv::Mat imgmat(512, 512, CV_8UC1, presult_img);
		cv::namedWindow("multiwindow", cv::WINDOW_FREERATIO);
		cv::Mat garay;
		//cv::normalize(imgmat12, garay12, 0, 1,cv::NORM_MINMAX);//��һ��   
		cv::threshold(imgmat, garay, 0, 255, cv::THRESH_BINARY);//��ֵ��
		cv::imshow("multiwindow", garay);
		cv::waitKey(0);

		//���ر��Լ��ι��ܸ���������ļ��ͼ��㣬���params����
		start = std::chrono::high_resolution_clock::now();
		signum = 1;
		signum = CTRUnet_Detectionobj.CTR_main(presult_img, downsampwidth, downsampheight , params);//Ԥ��512*512��Ұ��Ĥ������ڣ��ҷΣ������Ϊ1����Σ����С��Ϊ2�����·���Ϊӳ��ԭʼͼ����뺯���ڼ���
		if (signum != 1)
		{
			//���Լ���־
			return 0;
		}
		end = std::chrono::high_resolution_clock::now();
		// �������ʱ�䲢ת��Ϊ����
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		// �������ʱ��
		std::cout << "Code segment took " << duration.count() << " milliseconds." << std::endl;

		//------------------------------------------ͼ�����ʾ-----------------------------------------///
		float ri = abs(float(params[10] - params[8] + 1)) / abs(float(params[6] - params[4] + 1));
		cout << "���رȵ���ֵΪ��" << ri << endl;

		//������ļ�⣬���pdiaph_Line_img�������߽�ͼ�����ͼ�����������ֵΪ128���Ҳ�ͼ�����������ֵΪ255
		signum = 1;
		signum = CTRUnet_Detectionobj.Diaphragm_detect(presult_img, downsampwidth, downsampheight, params, pdiaph_Line_imgorg);
		if (signum != 1)
		{
			//���Լ���־
			return 0;
		}

		//���ҷε�ʶ�����������㣬������ҷ���ģpleftrightlung_imgmask���ҷ�1�����2��
		signum = 1;
		signum = CTRUnet_Detectionobj.Lung_Areacalculate(presult_img, downsampwidth, downsampheight, pleftrightlung_imgmaskorg, params);// ���ҷ�ʶ����������
		if (signum != 1)
		{
			//���Լ���־
			return 0;
		}

		//Ч��������ʾ
		//Unetseg_lungobj.img_show(pImagechar, params, pdiaph_Line_imgorg, pleftrightlung_imgmaskorg); //                    

		delete[] presult_img;
		presult_img = nullptr;
	}
	getchar();
	return 1;
}