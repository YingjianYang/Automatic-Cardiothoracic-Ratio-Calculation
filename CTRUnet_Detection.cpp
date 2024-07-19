#pragma once
#include "CTRUnet_Detection.h"


void CTRUnet_Detection::Lungmask_clean(cv::Mat& result_img)
{
	int width = result_img.cols;
	int height = result_img.rows;
	Matrix<unsigned short> matmasklungcopy(width, height, result_img.data);
	for (int i = 0; i < 512 * 512; i++)
	{
		if ((&result_img.at<unsigned char>(0, 0))[i] != 0)
		{
			matmasklungcopy.pdata[i] = 1;
		}
	}
	//��ͨ��ȥ��
	Erase_holl(matmasklungcopy, 0.05);//������ͨ��С��0.05������
	for (int i = 0; i < 512 * 512; i++)
	{
		if (matmasklungcopy.pdata[i] != 0)
		{
			(&result_img.at<unsigned char>(0, 0))[i] = 255;
		}
		else
		{
			(&result_img.at<unsigned char>(0, 0))[i] = 0;
		}
	}
}


int CTRUnet_Detection::CTR_main(unsigned char* presult_img, int downsampwidth, int downsampheight,  int params[])
{
	int imgwidth = params[0];
	int imgheight = params[1];

	//����ת����
	Matrix<unsigned short> matmasklung(downsampwidth, downsampheight);
	//�������ҷ�Ұ����
	Matrix<unsigned short> matleftlungmask(downsampwidth, downsampheight);
	Matrix<unsigned short> matrightlungmask(downsampwidth, downsampheight);

	for (int i = 0; i < matmasklung.Matrix_length(); i++)
	{
		if (presult_img[i] != 0)
		{
			matmasklung.pdata[i] = 1;
		}
	}

	int signum = 1;
	vector<int> xvec_left;
	vector<int> yvec_left;
	vector<int> xvec_right;
	vector<int> yvec_right;
	signum = LeftRightSegimg(matmasklung, matleftlungmask, matrightlungmask, xvec_left, yvec_left, xvec_right, yvec_right);
	if (signum != 1)
	{
		return 0;
	} 
	//       
	
	//��������Ķ���λ�á�
	int lefttop_Y = *min_element(yvec_left.begin(), yvec_left.end());
	int lefttop_X = xvec_left[min_element(yvec_left.begin(), yvec_left.end())- yvec_left.begin()];
	//��������ĵ׵�λ�á�
	vector<int> tempribX;
	vector<int> tempribY; //    

	for (int i = 0; i < xvec_left.size(); i++)
	{
		if (xvec_left[i] < lefttop_X)
		{
			tempribX.push_back(xvec_left[i]);
			tempribY.push_back(yvec_left[i]);
		}
	}
	int leftbot_Y = *max_element(tempribY.begin(), tempribY.end());
	int leftbot_X = tempribX[max_element(tempribY.begin(), tempribY.end()) - tempribY.begin()];

	//�Ҳ������Ķ���λ��
	int righttop_Y= *min_element(yvec_right.begin(), yvec_right.end());
	int righttop_X = xvec_right[min_element(yvec_right.begin(), yvec_right.end())- yvec_right.begin()];
	//�Ҳ������ĵ׵�λ��
	tempribX.clear();
	tempribY.clear();
	for (int i = 0; i < xvec_right.size(); i++)
	{
		if (xvec_right[i] > righttop_X)
		{
			tempribX.push_back(xvec_right[i]);
			tempribY.push_back(yvec_right[i]);
		}
	}

	int rightbot_Y = *max_element(tempribY.begin(), tempribY.end());
	int rightbot_X = tempribX[max_element(tempribY.begin(), tempribY.end()) - tempribY.begin()];

	vector<int> xvec_leftrib;
	vector<int> yvec_leftrib;
	vector<int> xvec_leftdiaph;
	vector<int> yvec_leftdiaph;//���ָ�
	signum = RibDiaphSeg_left(matleftlungmask, xvec_left, yvec_left, xvec_leftrib, yvec_leftrib, xvec_leftdiaph, yvec_leftdiaph);
	if (signum != 1)
	{
		return 0;
	}

	vector<int> xvec_rightrib;
	vector<int> yvec_rightrib;
	vector<int> xvec_rightdiaph;
	vector<int> yvec_rightdiaph;//�Ҳ�ָ�
	signum = RibDiaphSeg_right(matrightlungmask, xvec_right, yvec_right, xvec_rightrib, yvec_rightrib, xvec_rightdiaph, yvec_rightdiaph);
	if (signum != 1)
	{
		return 0;
	}

	//Ѱ�����ķָ���
	int Rib_Leftind = *max_element(xvec_right.begin(), xvec_right.end());//�ҵ��������߹�λ��
	int Rib_Rightind = *min_element(xvec_left.begin(), xvec_left.end());//�ҵ����Ҳ���߹�λ��
	//���ر��м��ߵ�X����ָ���λ��
	int Midseg_X = (Rib_Leftind + Rib_Rightind) / 2;

	//�����ݸ�Ĥ�߽�����
	vector<int> Xdiaph_Leftorder;
	vector<int> Ydiaph_Leftorder;
	Matrix<unsigned short> matLableimg_left(downsampwidth, downsampheight);
	GetorderDiaph(matLableimg_left, xvec_leftdiaph, yvec_leftdiaph, Xdiaph_Leftorder, Ydiaph_Leftorder);//  

	//cv::Mat imgmat1(downsampheight, downsampwidth, CV_16UC1, matLableimg_left.pdata);
	//cv::Mat garay1;
	//cv::threshold(imgmat1, garay1, 0, 65535, cv::THRESH_BINARY);//��ֵ��
	//cv::imshow("������", garay1);
	//cv::waitKey(0);

	//�Ҳ���ݸ�Ĥ�߽�����
	vector<int> Xdiaph_Rightorder;
	vector<int> Ydiaph_Rightorder;
	Matrix<unsigned short> matLableimg_right(downsampwidth, downsampheight);
	GetorderDiaph(matLableimg_right, xvec_rightdiaph, yvec_rightdiaph, Xdiaph_Rightorder, Ydiaph_Rightorder);//  

	//cv::Mat imgmat2(downsampheight, downsampwidth, CV_16UC1, matLableimg_right.pdata);
	//cv::Mat garay2;
	//cv::threshold(imgmat2, garay2, 0, 65535, cv::THRESH_BINARY);//��ֵ��
	//cv::imshow("�Ҳ����", garay2);
	//cv::waitKey(0);

	//�ж�ͼ����������λ����ҷ�,�߽������ڷ�Ұ���У�ֻ��ȡ����һ�����жϼ���
	int diaph_Leftind = 0; //���������Ч���λ�� �� //�����ȡ�յ�
	int hart_Leftind = 0; //���������Ч���λ��
	int leftmaxdisind;//���յ�������������λ��

	int diaph_Rightind = 0; //�Ҳ���Ч���λ��
	int hart_Rightind = 0; //�Ҳ���Ч���λ��
	int rightmaxdisind;//�Ҳ�յ����������λ��

	int deta_Y = 20;

	//�ж����ҷβ�����
	int LeftdisX = 0;
	int RightdisX = 0;
	Matrix<unsigned short> matLeftdis(matleftlungmask.width, matleftlungmask.height);
	Matrix<unsigned short> matRightdis(matrightlungmask.width, matrightlungmask.height);
	LeftRight_dist(LeftdisX, RightdisX, Midseg_X, matLeftdis, matRightdis,
		xvec_leftrib, yvec_leftrib, xvec_rightrib, yvec_rightrib,
		Xdiaph_Leftorder, Ydiaph_Leftorder, Xdiaph_Rightorder, Ydiaph_Rightorder);

	if (LeftdisX < RightdisX)//(matmasklung.pdata[index] == 1)//����1Ϊ�ҷΣ�����Ӧ�ҷ�
	{
		//�����ҷ�
		vector<float> nvect_leftratio;//�ҷβ�ÿ�����б��ֵ,leftmaxdisindΪ�ҷε�б��
		GetHeartDiaphPoint_rightlung(Xdiaph_Leftorder, Ydiaph_Leftorder, leftmaxdisind, nvect_leftratio, 1);//ͼ����ഫ1�����Ϊ��    
		//�ҷ�б�ʶ�Ӧ��Y����
		int leftmaxdis_Y = Ydiaph_Leftorder[leftmaxdisind];//��εĹյ�Ѱ�Ҿ��ڴ˵��λ�����½���Ѱ��
		leftmaxdis_Y = leftmaxdis_Y - deta_Y;
		//�������
		vector<float> nvect_rightratio;//��β���ͨ���ÿ�����б��ֵ
		GetHeartDiaphPoint_leftlung(Xdiaph_Rightorder, Ydiaph_Rightorder, rightmaxdisind, nvect_rightratio, leftmaxdis_Y, -1); //ͼ���Ҳഫ-1�����Ϊ��

		//���ݴ��������Ǹ���ͨ��б���ҵ������Եλ�úͺ������Եλ�� 
		Find_hartdiaphind_smallheart(Xdiaph_Leftorder, Ydiaph_Leftorder, nvect_leftratio, leftmaxdisind, hart_Leftind, diaph_Leftind, 1);//leftmaxdisind

		//���ݴ��������Ǹ���ͨ��б���ҵ������Եλ�úͺ������Եλ�� 
		Find_hartdiaphind_largeheart(Xdiaph_Rightorder, Ydiaph_Rightorder, nvect_rightratio, rightmaxdisind, hart_Rightind, diaph_Rightind, -1);
	}
	else //if (matmasklung.pdata[index] == 2)//����Ӧ���
	{
		//�����ҷ�
		vector<float> nvect_rightratio;//�ҷβ�ÿ�����б��ֵ,leftmaxdisindΪ�ҷε�б��
		GetHeartDiaphPoint_rightlung(Xdiaph_Rightorder, Ydiaph_Rightorder, rightmaxdisind, nvect_rightratio, -1);//ͼ���Ҳഫ-1  
		//�ҷ�б�ʶ�Ӧ��Y����
		int rightmaxdis_Y = Ydiaph_Rightorder[rightmaxdisind];//��εĹյ�Ѱ�Ҿ��ڴ˵��λ�����½���Ѱ��
		rightmaxdis_Y = rightmaxdis_Y - deta_Y;
		//�������
		vector<float> nvect_leftratio;
		GetHeartDiaphPoint_leftlung(Xdiaph_Leftorder, Ydiaph_Leftorder, leftmaxdisind, nvect_leftratio, rightmaxdis_Y, 1); //ͼ����ഫ1��

		//���ݴ��������Ǹ���ͨ��б���ҵ������Եλ�úͺ������Եλ�� 
		Find_hartdiaphind_smallheart(Xdiaph_Rightorder, Ydiaph_Rightorder, nvect_rightratio, rightmaxdisind, hart_Rightind, diaph_Rightind, -1);//leftmaxdisind

		//���ݴ��������Ǹ���ͨ��б���ҵ������Եλ�úͺ������Եλ�� 
		Find_hartdiaphind_largeheart(Xdiaph_Leftorder, Ydiaph_Leftorder, nvect_leftratio, leftmaxdisind, hart_Leftind, diaph_Leftind, 1);
	}
	/*else
	{
		;
	}*/

	//���ͼ�������ͺ�������λ�ã������                          
	int LefthartY = Ydiaph_Leftorder[hart_Leftind];
	int LefthartX = Xdiaph_Leftorder[hart_Leftind];

	//���ͼ�������
	int LeftdiaphY = Ydiaph_Leftorder[diaph_Leftind];
	int LeftdiaphX = Xdiaph_Leftorder[diaph_Leftind];

	//�Ҳ�ͼ�������ͺ��������Ҳ�λ�ã�����
	int RighthartY = Ydiaph_Rightorder[hart_Rightind];
	int RighthartX = Xdiaph_Rightorder[hart_Rightind];

	//�Ҳ�ͼ�������
	int RightdiaphY = Ydiaph_Rightorder[diaph_Rightind];
	int RightdiaphX = Xdiaph_Rightorder[diaph_Rightind];

	//���յ�
	int LeftmaxdisY = Ydiaph_Leftorder[leftmaxdisind];
	int LeftmaxdisX = Xdiaph_Leftorder[leftmaxdisind];

	//�Ҳ�յ�
	int RightmaxdisY = Ydiaph_Rightorder[rightmaxdisind];
	int RightmaxdisX = Xdiaph_Rightorder[rightmaxdisind];

	//���������߹ǵ��ȷ��
	if (LeftdisX < RightdisX)//(matmasklung.pdata[index] == 1)//���ͼ���Ӧ�ҷ�
	{
		int coutnum = 0;
		//����߹Ǳ߽���Ч����
		for (int i = 0; i < xvec_leftrib.size(); i++)
		{
			++coutnum;
			if (yvec_leftrib[i] == Ydiaph_Leftorder[diaph_Leftind])//�������������Yֵ��ͬ���߹ǵ�
			{
				params[3] = yvec_leftrib[i];
				params[4] = xvec_leftrib[i];
				break;
			}
		}

		if (coutnum == yvec_leftrib.size())
		{
			params[3] = yvec_leftrib.back();
			params[4] = xvec_leftrib.back();
		}

		//�Ҳ��߹Ǳ߽���Ч����
		int leftribeffect = Ydiaph_Leftorder[diaph_Leftind];
		vector<int> rightribX;
		vector<int> rightribY;
		for (int i = 0; i < yvec_rightrib.size(); i++)
		{
			if (yvec_rightrib[i] == leftribeffect)//�������������Yֵ��ͬ���߹ǵ�
			{
				rightribY.push_back(yvec_rightrib[i]);
				rightribX.push_back(xvec_rightrib[i]);
			}
		}

		if (rightribY.empty())
		{
			params[5] = yvec_rightrib.back();
			params[6] = xvec_rightrib.back();
		}
		else
		{
			int maxind = 0;
			maxind = max_element(rightribX.begin(), rightribX.end()) - rightribX.begin();
			params[5] = rightribY[maxind];
			params[6] = rightribX[maxind];
		}
	}
	else//���ַ�����
	{

		//����߹Ǳ߽���Ч����
		vector<int> leftribX;
		vector<int> leftribY;

		for (int i = 0; i < yvec_leftrib.size(); i++)
		{
			if (yvec_leftrib[i] == Ydiaph_Rightorder[diaph_Rightind])//�������������Yֵ��ͬ���߹ǵ�
			{
				leftribY.push_back(yvec_leftrib[i]);
				leftribX.push_back(xvec_leftrib[i]);
				break;
			}
		}
		if (leftribY.empty())
		{
			params[3] = yvec_leftrib.back();
			params[4] = xvec_leftrib.back();
		}
		else
		{
			int maxind = 0;
			maxind = max_element(leftribX.begin(), leftribX.end()) - leftribX.begin();
			params[3] = leftribY[maxind];
			params[4] = leftribX[maxind];
		}

		int coutnum = 0;

		//�Ҳ��߹Ǳ߽���Ч����
		for (int i = 0; i < yvec_rightrib.size(); i++)
		{
			++coutnum;
			if (yvec_rightrib[i] == Ydiaph_Rightorder[diaph_Rightind])//�������������Yֵ��ͬ���߹ǵ�
			{
				params[5] = yvec_rightrib[i];
				params[6] = xvec_rightrib[i];
				break;
			}
		}

		if (coutnum == yvec_rightrib.size())
		{
			params[5] = yvec_rightrib.back();
			params[6] = xvec_rightrib.back();
		}
	}

	params[7] = LefthartY;//������Y����
	params[8] = LefthartX;//������X����

	params[9] = RighthartY;//������Y����
	params[10] = RighthartX;//������X����

	params[11] = LeftdiaphY;//�������Y����
	params[12] = LeftdiaphX;//�������X����

	params[13] = RightdiaphY;//�Һ�����Y����
	params[14] = RightdiaphX;//�Һ�����X����

	params[15] = LeftmaxdisY;//���յ�y����
	params[16] = LeftmaxdisX;//���յ�x����

	params[17] = RightmaxdisY;//�Ҳ�յ�Y����
	params[18] = RightmaxdisX;//�Ҳ�յ�x����	

	params[19]= lefttop_Y;//����Ұ����Y����
	params[20]= lefttop_X;//����Ұ����x����
	params[21]= righttop_Y;//�Ҳ��Ұ����Y����
	params[22]= righttop_X;//�Ҳ��Ұ����x����

	params[23] = leftbot_Y;//����Ұ�׵�Y����
	params[24] = leftbot_X;//����Ұ�׵�x����
	params[25] = rightbot_Y;//�Ҳ��Ұ�׵�Y����
	params[26] = rightbot_X;//�Ҳ��Ұ�׵�x����

	// ӳ��ԭʼͼ���رȼ��������������ꡣӳ�䵽ԭʼͼ��
	MaptoOrg_CTR(imgwidth, imgheight, downsampwidth, downsampheight, params);

	return 1;
}

//���ͼ���������Ϊ128���Ҳ�ͼ���������Ϊ255
int CTRUnet_Detection::Diaphragm_detect(unsigned char* presult_img, int downsampwidth, int downsampheight,int params[],
	unsigned short* pdiaph_Line_imgorg)//downsampwidth, downsampheight,
{
	int imgwidth = params[0];
	int imgheight = params[1];

	unsigned char* pdiaph_Line_img = new unsigned char[512 * 512](); //�²����������ַ//512*512��ͼ����ģ
	//����ת��
	Matrix<unsigned short> matmasklung(downsampwidth, downsampheight);
	//�������ҷ�Ұ����
	Matrix<unsigned short> matleftlungmask(downsampwidth, downsampheight);
	Matrix<unsigned short> matrightlungmask(downsampwidth, downsampheight);

	for (int i = 0; i < matmasklung.Matrix_length(); i++)
	{
		if (presult_img[i] != 0)
		{
			matmasklung.pdata[i] = 1;
		}
	}

	int signum = 1;
	vector<int> xvec_left;
	vector<int> yvec_left;
	vector<int> xvec_right;
	vector<int> yvec_right;
	signum = LeftRightSegimg(matmasklung, matleftlungmask, matrightlungmask, xvec_left, yvec_left, xvec_right, yvec_right);//ֻ������ͼ��������Ҳ�ķ�Ұ
	if (signum != 1)
	{
		return 0;
	}

	//��������Ķ���λ��
	int lefttop_Y = *min_element(yvec_left.begin(), yvec_left.end());
	int lefttop_X = xvec_left[min_element(yvec_left.begin(), yvec_left.end()) - yvec_left.begin()];
	//�Ҳ������Ķ���λ��
	int righttop_Y = *min_element(yvec_right.begin(), yvec_right.end());
	int righttop_X = xvec_right[min_element(yvec_right.begin(), yvec_right.end()) - yvec_right.begin()];

	vector<int> xvec_leftrib;
	vector<int> yvec_leftrib;
	vector<int> xvec_leftdiaph;
	vector<int> yvec_leftdiaph;//���ָ�
	signum = RibDiaphSeg_left(matleftlungmask, xvec_left, yvec_left, xvec_leftrib, yvec_leftrib, xvec_leftdiaph, yvec_leftdiaph);
	if (signum != 1)
	{
		return 0;
	}

	vector<int> xvec_rightrib;
	vector<int> yvec_rightrib;
	vector<int> xvec_rightdiaph;
	vector<int> yvec_rightdiaph;//�Ҳ�ָ�
	signum = RibDiaphSeg_right(matrightlungmask, xvec_right, yvec_right, xvec_rightrib, yvec_rightrib, xvec_rightdiaph, yvec_rightdiaph);
	if (signum != 1)
	{
		return 0;
	}

	//Ѱ�����ķָ���
	int Rib_Leftind = *max_element(xvec_right.begin(), xvec_right.end());//  �ҵ��������߹�λ��
	int Rib_Rightind = *min_element(xvec_left.begin(), xvec_left.end());//  �ҵ����Ҳ���߹�λ��
	//���ر��м��ߵ�X����ָ���λ��
	int Midseg_X = (Rib_Leftind + Rib_Rightind) / 2;   

	//�����ݸ�Ĥ�߽�����
	vector<int> Xdiaph_Leftorder;
	vector<int> Ydiaph_Leftorder;
	Matrix<unsigned short> matLableimg_left(downsampwidth, downsampheight);
	GetorderDiaph(matLableimg_left, xvec_leftdiaph, yvec_leftdiaph, Xdiaph_Leftorder, Ydiaph_Leftorder);//                                  


	//�Ҳ���ݸ�Ĥ�߽�����
	vector<int> Xdiaph_Rightorder;
	vector<int> Ydiaph_Rightorder;
	Matrix<unsigned short> matLableimg_right(downsampwidth, downsampheight);
	GetorderDiaph(matLableimg_right, xvec_rightdiaph, yvec_rightdiaph, Xdiaph_Rightorder, Ydiaph_Rightorder);//  


	//�ж�ͼ����������λ����ҷ�,�߽������ڷ�Ұ���У�ֻ��ȡ����һ�����жϼ���
	int diaph_Leftind = 0; //���������Ч���λ�� �� //�����ȡ�յ�
	int hart_Leftind = 0; //���������Ч���λ��
	int leftmaxdisind;//���յ�������������λ��

	int diaph_Rightind = 0; //�Ҳ���Ч���λ��
	int hart_Rightind = 0; //�Ҳ���Ч���λ��
	int rightmaxdisind;//�Ҳ�յ����������λ��

	int deta_Y = 20;

	//�ж����ҷβ�����
	int LeftdisX = 0;
	int RightdisX = 0;
	Matrix<unsigned short> matLeftdis(matleftlungmask.width, matleftlungmask.height);
	Matrix<unsigned short> matRightdis(matrightlungmask.width, matrightlungmask.height);
	LeftRight_dist(LeftdisX, RightdisX, Midseg_X, matLeftdis, matRightdis,
		xvec_leftrib, yvec_leftrib, xvec_rightrib, yvec_rightrib,
		Xdiaph_Leftorder, Ydiaph_Leftorder, Xdiaph_Rightorder, Ydiaph_Rightorder);

	if (LeftdisX <= RightdisX)//(matmasklung.pdata[index] == 1)//����1Ϊ�ҷΣ�����Ӧ�ҷ�
	{
		//�����ҷ�
		vector<float> nvect_leftratio;//�ҷβ�ÿ�����б��ֵ,leftmaxdisindΪ�ҷε�б��
		GetHeartDiaphPoint_rightlung(Xdiaph_Leftorder, Ydiaph_Leftorder, leftmaxdisind, nvect_leftratio, 1);//ͼ����ഫ1�����Ϊ��    
		//�ҷ�б�ʶ�Ӧ��Y����
		int leftmaxdis_Y = Ydiaph_Leftorder[leftmaxdisind];//��εĹյ�Ѱ�Ҿ��ڴ˵��λ�����½���Ѱ��
		leftmaxdis_Y = leftmaxdis_Y - deta_Y;
		//�������
		vector<float> nvect_rightratio;//��β���ͨ���ÿ�����б��ֵ
		GetHeartDiaphPoint_leftlung(Xdiaph_Rightorder, Ydiaph_Rightorder, rightmaxdisind, nvect_rightratio, leftmaxdis_Y, -1); //ͼ���Ҳഫ-1�����Ϊ��

	}
	else //if (matmasklung.pdata[index] == 2)//����Ӧ���
	{
		//�����ҷ�
		vector<float> nvect_rightratio;//�ҷβ�ÿ�����б��ֵ,leftmaxdisindΪ�ҷε�б��
		GetHeartDiaphPoint_rightlung(Xdiaph_Rightorder, Ydiaph_Rightorder, rightmaxdisind, nvect_rightratio, -1);//ͼ���Ҳഫ-1  
		//�ҷ�б�ʶ�Ӧ��Y����
		int rightmaxdis_Y = Ydiaph_Rightorder[rightmaxdisind];//��εĹյ�Ѱ�Ҿ��ڴ˵��λ�����½���Ѱ��
		rightmaxdis_Y = rightmaxdis_Y - deta_Y;

		//�������
		vector<float> nvect_leftratio;
		GetHeartDiaphPoint_leftlung(Xdiaph_Leftorder, Ydiaph_Leftorder, leftmaxdisind, nvect_leftratio, rightmaxdis_Y, 1); //ͼ����ഫ1��
	}

	//---------------------------��Ydiaph_Leftorder��Ydiaph_Rightorder������ĵ����ߺ͹յ�������,�ֱ��ȡ�����Ҳ��������----------------
	//�����
	memset(matLableimg_right.pdata, 0, sizeof(unsigned short)* matLableimg_right.Matrix_length());
	memset(matLableimg_left.pdata, 0, sizeof(unsigned short)* matLableimg_left.Matrix_length());
	for (int i = 0; i < Ydiaph_Leftorder.size(); i++)
	{
		matLableimg_left.pdata[Ydiaph_Leftorder[i] * downsampwidth + Xdiaph_Leftorder[i]] = 1;
	}

	int leftmaxdisindlimit = leftmaxdisind;
	if (leftmaxdisind + 10 < Ydiaph_Rightorder.size())
	{
		leftmaxdisindlimit = leftmaxdisind + 10;
	}
	for (int i = leftmaxdisind - 15; i < leftmaxdisindlimit; i++)
	{
		matLableimg_left.pdata[Ydiaph_Leftorder[i] * downsampwidth + Xdiaph_Leftorder[i]] = 0;
	}
	bool TF = 0;
	vector <int> labval;
	vector <int> labind;
	float sum1 = 0;
	float sum2 = 0;
	int labindvalue = 1;
	Conectchose(matLableimg_left, matLableimg_right, labval, labind, TF);
	//��ͨ���ж�ѡ���ȡ������
	if (labind.size() == 1)
	{
		for (size_t i = leftmaxdisind - 10; i < Ydiaph_Leftorder.size(); i++)
		{
			cv::Point pointright;
			pointright.y = Ydiaph_Leftorder[i];
			pointright.x = Xdiaph_Leftorder[i];
			pdiaph_Line_img[Ydiaph_Leftorder[i] * downsampwidth + Xdiaph_Leftorder[i]] = 128;
		}
	}
	else
	{

		for (int i = 0, step1 = 0; i < matLableimg_right.height; i++, step1 += downsampwidth)
		{
			for (int j = 0; j < matLableimg_right.width; j++)
			{
				if (matLableimg_right.pdata[step1 + j] == labind[labind.size() - 2])
				{
					sum1 = sum1 + i;
				}
				else if (matLableimg_right.pdata[step1 + j] == labind.back())
				{
					sum2 = sum2 + i;
				}
			}
		}
		sum1 = sum1 / labval[labval.size() - 2];
		sum2 = sum2 / labval.back();

		//�ж����ݴ��ֵΪ����Ҫ��ֵ
		if (sum1 > sum2)
		{
			labindvalue = labind[labind.size() - 2];
		}
		else
		{
			labindvalue = labind.back();
		}
		for (int i = 0, step1 = 0; i < matLableimg_right.height; i++, step1 += downsampwidth)
		{
			for (int j = 0; j < downsampwidth; j++)
			{
				if (matLableimg_right.pdata[step1 + j] == labindvalue)
				{
					cv::Point pointleft;
					pointleft.y = i;
					pointleft.x = j;
					pdiaph_Line_img[i * downsampwidth + j] = 128;
				}
			}
		}
	}


	//�Ҳ���
	memset(matLableimg_right.pdata, 0, sizeof(unsigned short)* matLableimg_right.Matrix_length());
	memset(matLableimg_left.pdata, 0, sizeof(unsigned short)* matLableimg_left.Matrix_length());
	for (int i = 0; i < Ydiaph_Rightorder.size(); i++)
	{
		matLableimg_right.pdata[Ydiaph_Rightorder[i] * downsampwidth + Xdiaph_Rightorder[i]] = 1;
	}

	int rightmaxdisindlimit = rightmaxdisind;
	if (rightmaxdisind + 10 < Ydiaph_Rightorder.size())
	{
		rightmaxdisindlimit = rightmaxdisind + 10;
	}

	for (int i = rightmaxdisind - 15; i < rightmaxdisindlimit; i++)
	{
		matLableimg_right.pdata[Ydiaph_Rightorder[i] * downsampwidth + Xdiaph_Rightorder[i]] = 0;
	}
	labval.clear();
	labind.clear();
	sum1 = 0;
	sum2 = 0;
	TF = 0;

	//�Ҳ���ͨ���ж�ѡ�񣬻�ȡ������
	Conectchose(matLableimg_right, matLableimg_left, labval, labind, TF);
	if (labind.size() == 1)
	{
		for (size_t i = rightmaxdisind - 10; i < Ydiaph_Rightorder.size(); i++)
		{
			cv::Point pointright;
			pointright.y = Ydiaph_Rightorder[i];
			pointright.x = Xdiaph_Rightorder[i];
			pdiaph_Line_img[Ydiaph_Rightorder[i] * downsampwidth + Xdiaph_Rightorder[i]] = 255;
		}
	}
	else
	{
		for (int i = 0, step1 = 0; i < matLableimg_left.height; i++, step1 += downsampwidth)
		{
			for (int j = 0; j < downsampwidth; j++)
			{
				if (matLableimg_left.pdata[step1 + j] == labind[labind.size() - 2])
				{
					sum1 = sum1 + i;
				}
				else if (matLableimg_left.pdata[step1 + j] == labind.back())
				{
					sum2 = sum2 + i;
				}
			}
		}
		sum1 = sum1 / labval[labval.size() - 2];
		sum2 = sum2 / labval.back();

		//�ж����ݴ��ֵΪ����Ҫ��ֵ
		if (sum1 > sum2)
		{
			labindvalue = labind[labind.size() - 2];
		}
		else
		{
			labindvalue = labind.back();
		}

		for (int i = 0, step1 = 0; i < matLableimg_left.height; i++, step1 += downsampwidth)
		{
			for (int j = 0; j < downsampwidth; j++)
			{
				if (matLableimg_left.pdata[step1 + j] == labindvalue)
				{
					cv::Point pointleft;
					pointleft.y = i;
					pointleft.x = j;
					pdiaph_Line_img[i * downsampwidth + j] = 255;
				}
			}
		}
	}

	//ӳ�䵽ԭʼͼ
	MaptoOrg_Diaph(imgwidth, imgheight, downsampwidth, downsampheight, pdiaph_Line_img, pdiaph_Line_imgorg);//ӳ�䵽ԭʼͼ�ĺ���Ĥ�߽�

	delete[] pdiaph_Line_img;
	pdiaph_Line_img = nullptr;

	return 1;
}

//���ҷε����ֺ�������㣬pleftrightlung_imgmask�����ģ���ҷ�1�����2
int CTRUnet_Detection::Lung_Areacalculate(unsigned char* presult_img, int downsampwidth, int downsampheight, 
	unsigned short* pleftrightlung_imgmaskorg, int params[])
{
	int imgwidth = params[0];
	int imgheight = params[1];

	unsigned char* pleftrightlung_imgmask = new unsigned char[512 * 512](); //�²�������ͼ��������ģ512-*512

	//����ת��
	Matrix<unsigned short> matmasklung(downsampwidth, downsampheight);
	//�������ҷ�Ұ����
	Matrix<unsigned short> matleftlungmask(downsampwidth, downsampheight);
	Matrix<unsigned short> matrightlungmask(downsampwidth, downsampheight);

	for (int i = 0; i < matmasklung.Matrix_length(); i++)
	{
		if (presult_img[i] != 0)
		{
			matmasklung.pdata[i] = 1;
		}
	}

	int signum = 1;
	vector<int> xvec_left;
	vector<int> yvec_left;
	vector<int> xvec_right;
	vector<int> yvec_right;
	//matmasklung���������Ϊʹ�������С���������ҷε������������Ϊ�ҷ���Ϊ1�����С��Ϊ�����Ϊ2
	signum = LeftRightSegimg(matmasklung, matleftlungmask, matrightlungmask, xvec_left, yvec_left, xvec_right, yvec_right);//ֻ������ͼ��������Ҳ�ķ�Ұ
	if (signum != 1)
	{
		return 0;
	}

	vector<int> xvec_leftrib;
	vector<int> yvec_leftrib;
	vector<int> xvec_leftdiaph;
	vector<int> yvec_leftdiaph;//���ָ���Ե�ͺ��ݸ�Ĥ��
	signum = RibDiaphSeg_left(matleftlungmask, xvec_left, yvec_left, xvec_leftrib, yvec_leftrib, xvec_leftdiaph, yvec_leftdiaph);
	if (signum != 1)
	{
		return 0;
	}

	vector<int> xvec_rightrib;
	vector<int> yvec_rightrib;
	vector<int> xvec_rightdiaph;
	vector<int> yvec_rightdiaph;//�Ҳ�ָ���Ե�ͺ��ݸ�Ĥ��
	signum = RibDiaphSeg_right(matrightlungmask, xvec_right, yvec_right, xvec_rightrib, yvec_rightrib, xvec_rightdiaph, yvec_rightdiaph);
	if (signum != 1)
	{
		return 0;
	}

	//Ѱ�����ķָ���
	int Rib_Leftind = *max_element(xvec_right.begin(), xvec_right.end());//  �ҵ��������߹�λ��
	int Rib_Rightind = *min_element(xvec_left.begin(), xvec_left.end());//  �ҵ����Ҳ���߹�λ��
	//���ر��м��ߵ�X����ָ���λ��
	int Midseg_X = (Rib_Leftind + Rib_Rightind) / 2;

	//�����ݸ�Ĥ�߽�����
	vector<int> Xdiaph_Leftorder;
	vector<int> Ydiaph_Leftorder;
	Matrix<unsigned short> matLableimg_left(downsampwidth, downsampheight);
	GetorderDiaph(matLableimg_left, xvec_leftdiaph, yvec_leftdiaph, Xdiaph_Leftorder, Ydiaph_Leftorder);//                                  

	//�Ҳ���ݸ�Ĥ�߽�����
	vector<int> Xdiaph_Rightorder;
	vector<int> Ydiaph_Rightorder;
	Matrix<unsigned short> matLableimg_right(downsampwidth, downsampheight);
	GetorderDiaph(matLableimg_right, xvec_rightdiaph, yvec_rightdiaph, Xdiaph_Rightorder, Ydiaph_Rightorder);//  

	//�ж�ͼ����������λ����ҷ�,�߽������ڷ�Ұ���У�ֻ��ȡ����һ�����жϼ���
	int diaph_Leftind = 0; //���������Ч���λ�� �� //�����ȡ�յ�
	int hart_Leftind = 0; //���������Ч���λ��
	int leftmaxdisind; //���յ�������������λ��

	int diaph_Rightind = 0; //�Ҳ���Ч���λ��
	int hart_Rightind = 0; //�Ҳ���Ч���λ��
	int rightmaxdisind; //�Ҳ�յ����������λ��

	int deta_Y = 20; 

	//�ж����ҷβ�����
	int LeftdisX = 0;
	int RightdisX = 0;
	Matrix<unsigned short> matLeftdis(matleftlungmask.width, matleftlungmask.height);
	Matrix<unsigned short> matRightdis(matrightlungmask.width, matrightlungmask.height);
	LeftRight_dist(LeftdisX, RightdisX, Midseg_X, matLeftdis, matRightdis,
		xvec_leftrib, yvec_leftrib, xvec_rightrib, yvec_rightrib,
		Xdiaph_Leftorder, Ydiaph_Leftorder, Xdiaph_Rightorder, Ydiaph_Rightorder);

	int rightlungarea = 0;
	int leftlungarea = 0;

	//��������
   //�������ҷ�Ұ����
	Matrix<unsigned short> matleftlungmask_copy(downsampwidth, downsampheight);
	Matrix<unsigned short> matrightlungmask_copy(downsampwidth, downsampheight);
	signum = LeftRightsideimg(matmasklung, matleftlungmask_copy, matrightlungmask_copy);
	if (signum != 1)
	{
		return 0;
	}

	if (LeftdisX <= RightdisX)////����Ӧ�ҷΣ��ҷ�1�����2
	{
		//�ȸ�ֵ�ҷ�1
		for (int i = 0; i < matleftlungmask_copy.Matrix_length(); i++)
		{
			if (matleftlungmask_copy.pdata[i]!=0)
			{
				pleftrightlung_imgmask[i] = 1;
				rightlungarea++;
			}
		}
		//�ٸ�ֵ���2
		for (int i = 0; i < matrightlungmask_copy.Matrix_length(); i++)
		{
			if (matrightlungmask_copy.pdata[i] != 0)
			{
				pleftrightlung_imgmask[i] = 2;
				leftlungarea++;
			}
		}
		params[27] = rightlungarea;//����Ӧ�ҷ����
		params[28] = leftlungarea;//�Ҳ��Ӧ������
		params[29] = 1;//ͼ�������Ӧ�ҷΣ��Ҳ��Ӧ���
	}
	else //����Ӧ��Σ��ҷ�1�����2
	{

		//�ȸ�ֵ���2
		for (int i = 0; i < matleftlungmask_copy.Matrix_length(); i++)
		{
			if (matleftlungmask_copy.pdata[i] != 0)
			{
				pleftrightlung_imgmask[i] = 2;
				leftlungarea++;
			}
		}
		//�ٸ�ֵ�ҷ�1
		for (int i = 0; i < matrightlungmask_copy.Matrix_length(); i++)
		{
			if (matrightlungmask_copy.pdata[i] != 0)
			{
				pleftrightlung_imgmask[i] = 1;
				rightlungarea++;
			}
		}
		params[27] = leftlungarea;//����Ӧ������
		params[28] = rightlungarea;//�Ҳ��Ӧ�ҷ����
		params[29] = 2;//ͼ�������Ӧ��Σ��Ҳ��Ӧ�ҷ�
	}	

	//���ҷ�Ұʶ��ӳ�䵽ԭʼͼ��
	MaptoOrg_Areacal(imgwidth, imgheight, downsampwidth, downsampheight, params, pleftrightlung_imgmask, pleftrightlung_imgmaskorg);//���ҷ�ʶ����������ӳ�� 
	
	delete[] pleftrightlung_imgmask;
	pleftrightlung_imgmask = nullptr;

	return 1;   //     
}   

int CTRUnet_Detection::LeftRightSegimg(Matrix<unsigned short>& matmasklung, Matrix<unsigned short>& matLeftChest, Matrix<unsigned short>& matRightChest,
	vector<int>& xvec_left, vector<int>& yvec_left, vector<int>& xvec_right, vector<int>& yvec_right)
{
	int width = matmasklung.width;
	int height = matmasklung.height;
	int imglength = width * height;

	Matrix<unsigned short> matmasklungcopy(width, height);
	memcpy(matmasklungcopy.pdata, matmasklung.pdata, sizeof(unsigned short) * imglength);

	//��������Է�Ұ�����ҽ����ж���//
	vector<int> labval;
	vector<int> labind;
	bool TF = 0;
	Matrix<unsigned short> matLabelimg(width, height);
	Conectchose(matmasklungcopy, matLabelimg, labval, labind, TF);//����ͨ
	if (labind.size() <= 1)
	{
		return 0;
	}
	//�����ҷ�Ұ���������ͳ��
	for (int i = 0, step1 = 0; i < matmasklungcopy.height; i++, step1 += matmasklungcopy.width)
	{
		for (int j = 0; j < matmasklungcopy.width; j++)
		{
			if (matLabelimg.pdata[step1 + j] == labind.back())//����
			{
				matmasklung.pdata[step1 + j] = 1;//�������ҷδ�����Ϊ1
			}
			else if (matLabelimg.pdata[step1 + j] == labind[labind.size() - 2])//�����ڶ����
			{
				matmasklung.pdata[step1 + j] = 2;//���С����Σ�����Ϊ2
			}
			else
			{
				matmasklung.pdata[step1 + j] = 0;//����������Ϊ0
			}
		}
	}
	for (int i = 0; i < matmasklung.Matrix_length(); i++)
	{
		if (matmasklung.pdata[i] != 0)
		{
			matmasklungcopy.pdata[i] = 1;
		}
		else
		{
			matmasklungcopy.pdata[i] = 0;
		}
	}

	//������
	Matrix<unsigned short> strel(3, 3);
	for (int i = 0; i < 9; i++)
	{
		strel.pdata[i] = 1;
	}

	Matrix<unsigned short> dilateimg(width, height);
	Dilateimg(matmasklungcopy, strel, dilateimg);
	for (int i = 0; i < dilateimg.Matrix_length(); i++)
	{
		dilateimg.pdata[i] = dilateimg.pdata[i] - matmasklungcopy.pdata[i];
	}

	//��ͨ��ѡ��
	labval.clear();
	labind.clear();
	TF = 0;	
	memset(matLabelimg.pdata, 0, sizeof(unsigned short) * matLabelimg.Matrix_length());
	Conectchose(dilateimg, matLabelimg, labval, labind, TF);//����ͨ
	if (labind.size() <= 1)
	{
		return 0;
	}

	vector<int> xvec_labfirst;
	vector<int> yvec_labfirst;
	vector<int> xvec_labsec;
	vector<int> yvec_labsec;

	//�����ҷ�Ұ���������ͳ��
	for (int i = 0, step1 = 0; i < matLabelimg.height; i++, step1 += matLabelimg.width)
	{
		for (int j = 0; j < matLabelimg.width; j++)
		{
			if (matLabelimg.pdata[step1 + j] == labind.back())//����
			{
				xvec_labfirst.push_back(j);
				yvec_labfirst.push_back(i);
			}
			else if (matLabelimg.pdata[step1 + j] == labind[labind.size() - 2])//�����ڶ����
			{
				xvec_labsec.push_back(j);
				yvec_labsec.push_back(i);
			}
		}
	}

	//������������ж����ķָ��ߺ����Ҳ��Ұ
	int sumfirst = accumulate(xvec_labfirst.begin(), xvec_labfirst.end(), 0);
	sumfirst = sumfirst / xvec_labfirst.size();
	int sumsec = accumulate(xvec_labsec.begin(), xvec_labsec.end(), 0);
	sumsec = sumsec / xvec_labsec.size();

	int xmidseg = 0;
	if (sumfirst < sumsec)
	{
		//xvec_labfirstΪ���
		//�����������
		for (size_t i = 0; i < xvec_labfirst.size(); i++)//����������
		{
			matLeftChest.pdata[yvec_labfirst[i] * width + xvec_labfirst[i]] = 1;
		}
		for (size_t i = 0; i < xvec_labsec.size(); i++)//�Ҳ��������
		{
			matRightChest.pdata[yvec_labsec[i] * width + xvec_labsec[i]] = 1;
		}
		xvec_left.swap(xvec_labfirst);
		yvec_left.swap(yvec_labfirst);
		xvec_right.swap(xvec_labsec);
		yvec_right.swap(yvec_labsec);
	}
	else
	{
		//xvec_labsecΪ���,���������Ҳ�����С
		//�����������
		for (size_t i = 0; i < xvec_labsec.size(); i++)//����������
		{
			matLeftChest.pdata[yvec_labsec[i] * width + xvec_labsec[i]] = 1;
		}
		for (size_t i = 0; i < xvec_labfirst.size(); i++)//�Ҳ��������
		{
			matRightChest.pdata[yvec_labfirst[i] * width + xvec_labfirst[i]] = 1;
		}

		xvec_left.swap(xvec_labsec);
		yvec_left.swap(yvec_labsec);
		xvec_right.swap(xvec_labfirst);
		yvec_right.swap(yvec_labfirst);
	}

	memset(matLeftChest.pdata, 0, sizeof(unsigned short) * imglength);
	memset(matRightChest.pdata, 0, sizeof(unsigned short) * imglength);
	for (int i = 0; i < xvec_left.size(); i++)
	{
		matLeftChest.pdata[yvec_left[i] * width + xvec_left[i]] = 1;
	}
	for (int i = 0; i < xvec_right.size(); i++)
	{
		matRightChest.pdata[yvec_right[i] * width + xvec_right[i]] = 1;
	}

	return 1;
}

void CTRUnet_Detection::Conectchose(Matrix<unsigned short>& matInputimg, Matrix<unsigned short>& matLabelimg, vector<int>& labval, vector<int>& labind, bool TF)
{
	vector<int> labnum;
	if (TF)
	{
		//������
		int width = matInputimg.width;
		int height = matInputimg.height;

		vector<int> nxvect;//��ͨ����������ɺ�
		vector<int> nyvect;//��ͨ�����������ɺ�
		int clasenumber = 0;//��ͨ����
		//������������飬�����Ͻǿ�ʼ��˳ʱ�����������
		int neibx[8] = { -1, 0, 1, 1, 1, 0, -1, -1 };//������x����
		int neiby[8] = { -1, -1, -1, 0, 1, 1, 1, 0 };//������y����

		for (int i = 0, step1 = 0; i < height; i++, step1 += width)
		{
			unsigned short* inputimgrow = matInputimg.pdata + step1;//����ͼ������ֵ
			unsigned short* labelimgrow = matLabelimg.pdata + step1;//���ͼ������ֵ
			for (int j = 0; j < width; j++)
			{

				unsigned short* inputimgpos = inputimgrow + j;//����ͼ�����������
				unsigned short* labelimgpos = labelimgrow + j;//���ͼ��˵���������
				int deter = *inputimgpos;
				//�Ե�ǰֵ�����ж�
				if (deter == 1)//ֻҪ����ֵΪ1�������뿼�Ƿ���
				{
					clasenumber += 1;//��ͨ���ż�һ
					labnum.resize(clasenumber + 1);
					*inputimgpos = 0;//��ͼ��˵���Ϊ0
					*labelimgpos = clasenumber;//��ͼ��Դ˵���б�Ǽ�¼                

					//��ȡ�˴����ĵ�ĺ�������
					//�����ж�
					int runvalue = 0;
					int countnumber = 0; // �������
					nxvect.push_back(j);
					nyvect.push_back(i); // ������洢��vector����������             

					while (runvalue <= countnumber)
					{
						for (int k = 0; k < 8; k++)
						{//����
							int nx = nxvect[runvalue] + neibx[k];
							int ny = nyvect[runvalue] + neiby[k];
							if (nx < width && nx >= 0 && ny < height && ny >= 0)//����������ͼ��ߴ���������
							{
								unsigned short* nebinputimgpos = matInputimg.pdata + ny * width + nx;//����ͼ������λ��
								unsigned short* neblabelimgpos = matLabelimg.pdata + +ny * width + nx;//���ͼ���������λ��
								if (*nebinputimgpos == 1)
								{
									countnumber = countnumber + 1;
									nxvect.push_back(nx);//�洢�������ĺ�����
									nyvect.push_back(ny);//�洢��������������
									*nebinputimgpos = 0;//ԭʼ����ͼ���������ֵ��Ϊ0
									*neblabelimgpos = clasenumber;//���¾���Ĵ��������б�ű��
								}
							}
						}

						runvalue = runvalue + 1;//׷����ͨ�����������Ŀ��ֱ��׷��countnumber
					}

					labnum[clasenumber] = countnumber + 1; // ������ʼ��
				}

				nxvect.clear();
				nyvect.clear();
			}
		}
	}
	else
	{
		//������
		int width = matInputimg.width;
		int height = matInputimg.height;

		vector<int> nxvect;//��ͨ����������ɺ�
		vector<int> nyvect;//��ͨ�����������ɺ�
		int clasenumber = 0;//��ͨ�������

		//�������������飬�����Ͻǿ�ʼ��˳ʱ�����������
		int neibx[4] = { -1,0,1,0 };//������x����
		int neiby[4] = { 0,-1,0,1 };//������y����

		for (int i = 0; i < height; i++)
		{
			unsigned short* inputimgrow = matInputimg.pdata + i * width;//����ͼ������ֵ
			unsigned short* labelimgrow = matLabelimg.pdata + i * width;//���ͼ������ֵ
			for (int j = 0; j < width; j++)
			{
				unsigned short* inputimgpos = inputimgrow + j;//����ͼ�����������
				unsigned short* labelimgpos = labelimgrow + j;//���ͼ��˵���������
				int deter = *inputimgpos;

				//�Ե�ǰֵ�����ж�
				if (deter == 1)//ֻҪ����ֵΪ1�������뿼�Ƿ���
				{
					clasenumber = clasenumber + 1;//��ͨ���ż�һ
					labnum.resize(clasenumber + 1);//�ռ�ȱ�ŵ���Ŀ��1
					*inputimgpos = 0;//��ͼ��˵���Ϊ0
					*labelimgpos = clasenumber;//��ͼ��Դ˵���б�Ǽ�¼    

					int runvalue = 0;
					int countnumber = 0; // �������
					nxvect.push_back(j);
					nyvect.push_back(i); // ������洢��vector����������             

					while (runvalue <= countnumber)
					{
						for (int k = 0; k < 4; k++)
						{
							int nx = nxvect[runvalue] + neibx[k];
							int ny = nyvect[runvalue] + neiby[k];
							if (nx < width && nx >= 0 && ny < height && ny >= 0)//����������ͼ��ߴ���������
							{
								unsigned short* nebinputimgpos = matInputimg.pdata + ny * width + nx;//����ͼ������λ��
								unsigned short* neblabelimgpos = matLabelimg.pdata + +ny * width + nx;//���ͼ���������λ��
								if (*nebinputimgpos == 1)
								{
									countnumber = countnumber + 1;
									nxvect.push_back(nx);//�洢�������ĺ�����
									nyvect.push_back(ny);//�洢��������������
									*nebinputimgpos = 0;//ԭʼ����ͼ���������ֵ��Ϊ0
									*neblabelimgpos = clasenumber;//���¾���Ĵ��������б�ű��
								}
							}
						}
						runvalue = runvalue + 1;//׷����ͨ�����������Ŀ��ֱ��׷��countnumber
					}
					labnum[clasenumber] = countnumber + 1; //������ʼ�� 
				}
				nxvect.clear();
				nyvect.clear();
			}
		}
	}

	//����multimap��labsum�Ľ���ֵ����������������Ĺ�����ֵ�Ĵ�С����С����Ϊ˳��ͬʱ���ж�Ӧ������ֵ
	//�������������vector������
	multimap<int, int> labnummap;
	if (labnum.size() > 1)
	{
		for (int i = 1; i < labnum.size(); i++)
		{
			int keyind = i;
			int keyvalue = labnum[i];
			labnummap.insert(std::pair<int, int>(keyvalue, keyind));
		}

		// ��ֵ�Ĵ�С����
	   // ����ͳ�Ƶ���ֵ���������ŵ����� ����С����
		for (multimap<int, int>::iterator it = labnummap.begin(); it != labnummap.end(); it++)
		{
			labval.push_back((*it).first);
			labind.push_back((*it).second);
		}
	}
}


int CTRUnet_Detection::RibDiaphSeg_left(Matrix<unsigned short>& matLeftChest,
	vector<int>& xvec_left, vector<int>& yvec_left,
	vector<int>& xvec_leftrib, vector<int>& yvec_leftrib, vector<int>& xvec_leftdiaph, vector<int>& yvec_leftdiaph)
{

	//���β�����
	int yminleft = *min_element(yvec_left.begin(), yvec_left.end());//��ඥ��Y����
	int yminleft_ind = min_element(yvec_left.begin(), yvec_left.end()) - yvec_left.begin();
	int xminleft = xvec_left[yminleft_ind];//��ඥ��X����

	//��ȡ�����Ե���²�ĵ�
	vector<int> tempribX;
	vector<int> tempribY;
	for (int i = 0; i < xvec_left.size(); i++)
	{
		if (xvec_left[i]< xminleft)
		{
			tempribX.push_back(xvec_left[i]);
			tempribY.push_back(yvec_left[i]);
		}		
	}
	int ymaxleft = *max_element(tempribY.begin(), tempribY.end());
	int ymaxleft_ind = max_element(tempribY.begin(), tempribY.end()) - tempribY.begin();
	int xmaxleft = tempribX[ymaxleft_ind];

	//���������������ͨ����Ϊ0
	//������������飬�����Ͻǿ�ʼ��˳ʱ�����������
	int neibx[8] = { -1, 0, 1, 1, 1, 0, -1, -1 };//������x����
	int neiby[8] = { -1, -1, -1, 0, 1, 1, 1, 0 };//������y����
	vector<int> nxvect;//��ͨ����������ɺ�
	vector<int> nyvect;//��ͨ�����������ɺ�

	int indy = yminleft;
	int jndx = xminleft;
	if ((indy == yminleft) && (jndx == xminleft))//�Ϸ���16��ͨ��������Ϊ0
	{
		nxvect.push_back(jndx);
		nyvect.push_back(indy); // ������洢��vector����������  
		int runvalue = 0;
		while (runvalue < 16)
		{
			int sig = 1;
			for (int k = 0; k < 8; k++)
			{//����
				int nx = nxvect[runvalue] + neibx[k];
				int ny = nyvect[runvalue] + neiby[k];
				if (nx < matLeftChest.width && nx >= 0 && ny < matLeftChest.height && ny >= 0)//����������ͼ��ߴ���������
				{
					unsigned short* nebinputimgpos = matLeftChest.pdata + ny * matLeftChest.width + nx;//����ͼ������λ��
					if (*nebinputimgpos != 0)
					{
						nxvect.push_back(nx);//�洢�������ĺ�����
						nyvect.push_back(ny);//�洢��������������
						*nebinputimgpos = 0;//ԭʼ����ͼ���������ֵ��Ϊ0						
					}
				}
				else
				{
					sig = 0;
				}
			}
			if (sig==0)
			{
				break;
			}
			runvalue++;
		}
	}

	//��ζ�������10���߶�������ͨ����Ϊ0
	for (int i = yminleft; i < yminleft + 10; i++)
	{
		for (int j = 0; j < matLeftChest.width; j++)
		{
			matLeftChest.pdata[i * matLeftChest.width + j] = 0;
		}
	}

	//���������������ͨ����Ϊ0
	nxvect.clear();//��ͨ����������ɺ�
	nyvect.clear();//��ͨ�����������ɺ�
	indy = ymaxleft;
	jndx = xmaxleft;
	if ((indy == ymaxleft) && (jndx == xmaxleft))//�Ϸ���16��ͨ��������Ϊ0,matLeftChest.pdata[yminleft * matLeftChest.width + xminleft] == 1
	{
		nxvect.push_back(jndx);
		nyvect.push_back(indy); //������洢��vector����������  
		int runvalue = 0;
		while (nxvect.size() < 16)
		{
			int sig = 1;
			for (int k = 0; k < 8; k++)
			{//����
				int nx = nxvect[runvalue] + neibx[k];
				int ny = nyvect[runvalue] + neiby[k];
				if (nx < matLeftChest.width && nx >= 0 && ny < matLeftChest.height && ny >= 0)//����������ͼ��ߴ���������
				{
					unsigned short* nebinputimgpos = matLeftChest.pdata + ny * matLeftChest.width + nx;//����ͼ������λ��
					if (*nebinputimgpos != 0)
					{
						nxvect.push_back(nx);//�洢�������ĺ�����
						nyvect.push_back(ny);//�洢��������������
						*nebinputimgpos = 0;//ԭʼ����ͼ���������ֵ��Ϊ0
					}
				}
				else
				{
					sig = 0;
				}
			}
			if (sig==0)
			{
				break;
			}
			runvalue++;
		}
	}

	//��ͨ��ѡ��
	vector<int> labval;
	vector<int> labind;
	bool TF = 0;
	Matrix<unsigned short> matLabelimg(matLeftChest.width, matLeftChest.height);
	Conectchose(matLeftChest, matLabelimg, labval, labind, TF);//����ͨ
	if (labind.size() <= 1)
	{
		return 0;
	}

	//ͨ��X�����ۼ�ƽ�����жϱ߽�����Եor���ݸ��߽�
	vector<int> summean1;
	vector<int> summean2;
	for (int i = 0, step1 = 0; i < matLeftChest.height; i++, step1 += matLeftChest.width)
	{
		for (int j = 0; j < matLeftChest.width; j++)
		{
			if (matLabelimg.pdata[step1 + j] == labind.back())//�����ͨ��
			{
				summean1.push_back(j);

			}
			else if (matLabelimg.pdata[step1 + j] == labind[labind.size() - 2])//�����ڶ�����ͨ��
			{
				summean2.push_back(j);
			}
		}
	}
	int summean1value = accumulate(summean1.begin(), summean1.end(), 0) / summean1.size();
	int summean2value = accumulate(summean2.begin(), summean2.end(), 0) / summean2.size();

	//���ݺ������ƽ��ֵ�����ж�
	if (summean2value > summean1value)//1Ϊ��Ե
	{

		for (int i = 0, step1 = 0; i < matLeftChest.height; i++, step1 += matLeftChest.width)
		{
			for (int j = 0; j < matLeftChest.width; j++)
			{
				if (matLabelimg.pdata[step1 + j] == labind.back())//ƽ��ֵС������Ե
				{
					xvec_leftrib.push_back(j);
					yvec_leftrib.push_back(i);
				}
				else if (matLabelimg.pdata[step1 + j] == labind[labind.size() - 2])//ƽ��ֵ����Ǻ��ݸ�Ĥ�߽磬
				{
					xvec_leftdiaph.push_back(j);
					yvec_leftdiaph.push_back(i);
				}
			}
		}

	}
	else
	{
		for (int i = 0, step1 = 0; i < matLeftChest.height; i++, step1 += matLeftChest.width)
		{
			for (int j = 0; j < matLeftChest.width; j++)
			{
				if (matLabelimg.pdata[step1 + j] == labind.back())//ƽ��ֵ����Ǻ��ݸ�Ĥ�߽�
				{
					xvec_leftdiaph.push_back(j);
					yvec_leftdiaph.push_back(i);
				}
				else if (matLabelimg.pdata[step1 + j] == labind[labind.size() - 2])//ƽ��ֵС������Ե
				{
					xvec_leftrib.push_back(j);
					yvec_leftrib.push_back(i);
				}
			}
		}
	}
	return 1;
}


int CTRUnet_Detection::RibDiaphSeg_right(Matrix<unsigned short>& matRightChest,
	vector<int>& xvec_right, vector<int>& yvec_right,
	vector<int>& xvec_rightrib, vector<int>& yvec_rightrib,
	vector<int>& xvec_rightdiaph, vector<int>& yvec_rightdiaph)
{
	//�Ҳ�β�����
	int yminright = *min_element(yvec_right.begin(), yvec_right.end());//��ȡ�Ҳ����Ϸ����Y����
	int yminright_ind = min_element(yvec_right.begin(), yvec_right.end()) - yvec_right.begin();
	int xminright = xvec_right[yminright_ind];//��ȡ�Ҳ����Ϸ����X����

	//��ȡ�Ҳ�ͼ����Ե���·��ĵ������
	vector<int> tempribX;
	vector<int> tempribY;
	for (int i = 0; i < xvec_right.size(); i++)
	{
		if (xvec_right[i]> xminright)
		{
			tempribX.push_back(xvec_right[i]);
			tempribY.push_back(yvec_right[i]);
		}
	}

	int ymaxright = *max_element(tempribY.begin(), tempribY.end());
	int ymaxright_ind = max_element(tempribY.begin(), tempribY.end()) - tempribY.begin();
	int xmaxright = tempribX[ymaxright_ind];

	//���ҷ�����������ͨ����Ϊ0
	//������������飬�����Ͻǿ�ʼ��˳ʱ�����������
	int neibx[8] = { -1, 0, 1, 1, 1, 0, -1, -1 };//������x����
	int neiby[8] = { -1, -1, -1, 0, 1, 1, 1, 0 };//������y����
	vector<int> nxvect;//��ͨ����������ɺ�
	vector<int> nyvect;//��ͨ�����������ɺ�
	int indy = yminright;
	int jndx = xminright;
	if ((indy == yminright) && (jndx == xminright))//�Ϸ���16��ͨ��������Ϊ0
	{
		nxvect.push_back(jndx);
		nyvect.push_back(indy); // ������洢��vector����������  
		int runvalue = 0;
		while (nxvect.size() < 16)
		{
			for (int k = 0; k < 8; k++)
			{//����
				int nx = nxvect[runvalue] + neibx[k];
				int ny = nyvect[runvalue] + neiby[k];
				if (nx < matRightChest.width && nx >= 0 && ny < matRightChest.height && ny >= 0)//����������ͼ��ߴ���������
				{
					unsigned short* nebinputimgpos = matRightChest.pdata + ny * matRightChest.width + nx;//����ͼ������λ��
					if (*nebinputimgpos != 0)
					{
						nxvect.push_back(nx);//�洢�������ĺ�����
						nyvect.push_back(ny);//�洢��������������
						*nebinputimgpos = 0;//ԭʼ����ͼ���������ֵ��Ϊ0
					}
				}
			}
			runvalue++;
		}
	}

	//�ҷζ�������10���߶�������ͨ����Ϊ0
	for (int i = yminright; i < yminright + 10; i++)
	{
		for (int j = 0; j < matRightChest.width; j++)
		{
			matRightChest.pdata[i * matRightChest.width + j] = 0;
		}
	}

	//���ҷ�����������ͨ����Ϊ0
	nxvect.clear();//��ͨ����������ɺ�
	nyvect.clear();//��ͨ�����������ɺ�
	indy = ymaxright;
	jndx = xmaxright;
	if ((indy == ymaxright) && (jndx == xmaxright))//�Ϸ���16��ͨ��������Ϊ0,matRightChest.pdata[yminright * matRightChest.width + xminright] == 1
	{
		nxvect.push_back(jndx);
		nyvect.push_back(indy); //������洢��vector����������  
		int runvalue = 0;
		while (nxvect.size() < 16)
		{
			for (int k = 0; k < 8; k++)
			{//����
				int nx = nxvect[runvalue] + neibx[k];
				int ny = nyvect[runvalue] + neiby[k];
				if (nx < matRightChest.width && nx >= 0 && ny < matRightChest.height && ny >= 0)//����������ͼ��ߴ���������
				{
					unsigned short* nebinputimgpos = matRightChest.pdata + ny * matRightChest.width + nx;//����ͼ������λ��
					if (*nebinputimgpos != 0)
					{
						nxvect.push_back(nx);//�洢�������ĺ�����
						nyvect.push_back(ny);//�洢��������������
						*nebinputimgpos = 0;//ԭʼ����ͼ���������ֵ��Ϊ0
					}
				}
			}
			runvalue++;
		}
	}

	//��ͨ��ѡ��
	vector<int> labval;
	vector<int> labind;
	bool TF = 0;
	Matrix<unsigned short> matLabelimg(matRightChest.width, matRightChest.height);
	Conectchose(matRightChest, matLabelimg, labval, labind, TF);//����ͨ
	if (labind.size() <= 1)
	{
		return 0;
	}

	//ͨ��X�����ۼ�ƽ�����жϱ߽�����Եor���ݸ��߽�
	vector<int> summean1;
	vector<int> summean2;
	for (int i = 0, step1 = 0; i < matRightChest.height; i++, step1 += matRightChest.width)
	{
		for (int j = 0; j < matRightChest.width; j++)
		{
			if (matLabelimg.pdata[step1 + j] == labind.back())//�����ͨ��
			{
				summean1.push_back(j);

			}
			else if (matLabelimg.pdata[step1 + j] == labind[labind.size() - 2])//�����ڶ�����ͨ��
			{
				summean2.push_back(j);
			}
		}
	}
	int summean1value = accumulate(summean1.begin(), summean1.end(), 0) / summean1.size();
	int summean2value = accumulate(summean2.begin(), summean2.end(), 0) / summean2.size();


	//���ݺ������ƽ��ֵ�����ж�
	if (summean2value > summean1value)//1Ϊ���ݸ�Ĥ�߽�
	{

		for (int i = 0, step1 = 0; i < matRightChest.height; i++, step1 += matRightChest.width)
		{
			for (int j = 0; j < matRightChest.width; j++)
			{
				if (matLabelimg.pdata[step1 + j] == labind.back())//ƽ��ֵС���Ǻ��ݸ�Ĥ�߽�
				{
					xvec_rightdiaph.push_back(j);
					yvec_rightdiaph.push_back(i);

				}
				else if (matLabelimg.pdata[step1 + j] == labind[labind.size() - 2])//ƽ��ֵ�������Ե
				{
					xvec_rightrib.push_back(j);
					yvec_rightrib.push_back(i);
				}
			}
		}

	}
	else//2Ϊ���ݸ�Ĥ�߽磬1Ϊ��Ե
	{
		for (int i = 0, step1 = 0; i < matRightChest.height; i++, step1 += matRightChest.width)
		{
			for (int j = 0; j < matRightChest.width; j++)
			{
				if (matLabelimg.pdata[step1 + j] == labind.back())//ƽ��ֵ�������Ե
				{
					xvec_rightrib.push_back(j);
					yvec_rightrib.push_back(i);

				}
				else if (matLabelimg.pdata[step1 + j] == labind[labind.size() - 2])//ƽ��ֵС���Ǻ��ݸ�Ĥ�߽�
				{
					xvec_rightdiaph.push_back(j);
					yvec_rightdiaph.push_back(i);
				}
			}
		}
	}

	return 1;
}

void CTRUnet_Detection::Erodeimg(Matrix<unsigned short>& matInputimg, Matrix<unsigned short>& templatedilate, Matrix<unsigned short>& matOutputimg)
{
	int templateerodewidth = templatedilate.width;
	int templateerodeheight = templatedilate.height;

	int width = matInputimg.width;
	int height = matInputimg.height;

	//ģ���������б���������ȡ���׵�ַ
	int templatedis = templateerodewidth * templateerodeheight;

	int addwidth = templateerodewidth / 2;
	int addheight = templateerodeheight / 2;
	int widthheight = width * height;

	int tempwid = width + addwidth * 2;
	int tempheight = height + addheight * 2;
	int tempheigthwid = tempwid * tempheight;
	//����������ͼ��ռ䣬������ͼ������������ͼ��ռ�
	Matrix<unsigned short> matTempcalulate(tempwid, tempheight);

	//����¿ռ�
	//�м䲿�����
	for (int i = addheight, step1 = addheight * tempwid, step2 = 0; i < addheight + height; i++, step1 += tempwid, step2 += width)
	{
		for (int j = addwidth, step3 = 0; j < addwidth + width; j++, step3 += 1)
		{
			matTempcalulate.pdata[step1 + j] = matInputimg.pdata[step2 + step3];
		}
	}

	//���Ҳ�߽����
	for (int i = addheight, step1 = addheight * tempwid, step2 = 0; i < addheight + height; ++i, step1 += tempwid, step2 += width) //�����¿ռ�
	{
		//���߽����
		for (int j = 0; j < addwidth; ++j)
		{
			matTempcalulate.pdata[step1 + j] = matInputimg.pdata[step2];//���߽�����Ϊ��������������ͬ��
		}
		//�Ҳ�߽����
		for (int j = width + addwidth; j < tempwid; ++j)
		{
			matTempcalulate.pdata[step1 + j] = matInputimg.pdata[step2 + width - 1];//�Ҳ�߽�����Ϊ��������������ͬ��
		}
	}

	//�ϲ�߽����
	for (int i = 0, step1 = 0; i < addheight; i++, step1 += tempwid)//�ϲ���������
	{
		for (int j = 0; j < tempwid; j++)
		{
			matTempcalulate.pdata[step1 + j] = matTempcalulate.pdata[addheight * tempwid + j];
		}
	}
	//�²���������
	for (int i = height + addheight; i < tempheight; i++)
	{
		unsigned short* tempcalulaterow = matTempcalulate.pdata + i * tempwid;  //���ͼ���i�е��׵�ַ         
		unsigned short* botrow = matTempcalulate.pdata + (addheight + height - 1) * tempwid;  //���ͼ���i�е��׵�ַ    
		for (int j = 0; j < tempwid; j++)
		{
			tempcalulaterow[j] = botrow[j];
		}
	}

	//ע�⣺���ģ���г��ֵķ�ǰ��ֵ��Ĭ��Ϊ��������Ԫ�ء���������459ҳ
	//ģ���е�ǰ������
	int countnum = 0;
	for (int i = 0; i < templateerodewidth * templateerodeheight; i++)
	{
		if (templatedilate.pdata[i] == 1)
		{
			countnum = countnum + 1;//������һ
		}
	}

	//����ÿ���м�������ص㡣
	for (int j = addheight; j < addheight + height; j++)
	{
		unsigned short* inputimgrow = matTempcalulate.pdata + j * tempwid;//����ͼ�����ĵ�����׵�ַ
		unsigned short* outputimgrow = matOutputimg.pdata + (j - addheight) * width;//���ͼ�����ĵ�����׵�ַ
		for (int i = addwidth; i < addwidth + width; i++)//ÿһ�еı���
		{
			unsigned short* inputimgpos = inputimgrow + i;//����ͼ�����ĵ�ĵ�ַ
			unsigned short* outputimgpos = outputimgrow + i - addwidth;//���ͼ�����ĵ�ĵ�ַ

			int numcount = 0;//����������
			//����ģ�����
			for (int m = -addheight; m <= addheight; m++)//ģ������͵�ǰ����ƥ�����
			{
				unsigned short* templateeroderow = templatedilate.pdata + (m + addheight) * templateerodewidth;//ģ�������ֵ              
				unsigned short* temparryrow = inputimgpos + m * tempwid;//������������ֵ
				for (int n = -addwidth; n <= addwidth; n++)
				{
					unsigned short* templateerodepos = templateeroderow + n + addwidth;//ģ��ĵ�ǰ��ֵ
					unsigned short* temparrypos = temparryrow + n;//����ͼ��ǰֵ
					if (*templateerodepos == 1 && *temparrypos == 1) //ģ��ǰ��������ǰ���Ĵ�����һһ��Ӧ
					{
						numcount = numcount + 1;
					}
				}
			}//����ģ�����
			if (numcount == countnum)//���ģ������е�ǰ�����������ֵ���,��ζ��ģ���кͱ����������ж�һһ��Ӧ
			{
				*outputimgpos = 1;//���ֵ��ֵΪ1
			}
			else//
			{
				*outputimgpos = 0;//���ֵ��ֵΪ0
			}
		}
	}
}

void CTRUnet_Detection::Dilateimg(Matrix<unsigned short>& matInputimg, Matrix<unsigned short>& mattemplatedilate, Matrix<unsigned short>& matOutputimg)
{
	int templatedilatewidth = mattemplatedilate.width;
	int templatedilateheight = mattemplatedilate.height;

	int width = matInputimg.width;
	int height = matInputimg.height;

	//ģ���������б���������ȡ���׵�ַ
	int templatedis = templatedilatewidth * templatedilateheight;

	int addwidth = templatedilatewidth / 2;
	int addheight = templatedilateheight / 2;
	int widthheight = width * height;

	int tempwid = width + addwidth * 2;
	int tempheight = height + addheight * 2;
	int tempheigthwid = tempwid * tempheight;
	//����������ͼ��ռ䣬������ͼ������������ͼ��ռ�
	Matrix<unsigned short> matTempcalulate(tempwid, tempheight);

	//����¿ռ�
	//�м䲿�����
	for (int i = addheight, step1 = addheight * tempwid, step2 = 0; i < addheight + height; i++, step1 += tempwid, step2 += width)
	{
		for (int j = addwidth, step3 = 0; j < addwidth + width; j++, step3 += 1)
		{
			matTempcalulate.pdata[step1 + j] = matInputimg.pdata[step2 + step3];
		}
	}

	//���Ҳ�߽����
	for (int i = addheight, step1 = addheight * tempwid, step2 = 0; i < addheight + height; ++i, step1 += tempwid, step2 += width) //�����¿ռ�
	{
		//���߽����
		for (int j = 0; j < addwidth; ++j)
		{
			matTempcalulate.pdata[step1 + j] = matInputimg.pdata[step2];//���߽�����Ϊ��������������ͬ��
		}
		//�Ҳ�߽����
		for (int j = width + addwidth; j < tempwid; ++j)
		{
			matTempcalulate.pdata[step1 + j] = matInputimg.pdata[step2 + width - 1];//�Ҳ�߽�����Ϊ��������������ͬ��
		}
	}

	//�ϲ�߽����
	for (int i = 0, step1 = 0; i < addheight; i++, step1 += tempwid)//�ϲ���������
	{
		for (int j = 0; j < tempwid; j++)
		{
			matTempcalulate.pdata[step1 + j] = matTempcalulate.pdata[addheight * tempwid + j];
		}
	}
	//�²���������
	for (int i = height + addheight; i < tempheight; i++)
	{
		unsigned short* tempcalulaterow = matTempcalulate.pdata + i * tempwid;  //���ͼ���i�е��׵�ַ         
		unsigned short* botrow = matTempcalulate.pdata + (addheight + height - 1) * tempwid;  //���ͼ���i�е��׵�ַ    
		for (int j = 0; j < tempwid; j++)
		{
			tempcalulaterow[j] = botrow[j];
		}
	}

	//ģ�����
	for (int i = addheight; i < addheight + height; ++i) //����ģ���˲�         
	{
		unsigned short* inputrow = matTempcalulate.pdata + i * tempwid; //����ͼ���i�е��׵�ַ               
		unsigned short* outputimgrow = matOutputimg.pdata + (i - addheight) * width;  //���ͼ���i�е��׵�ַ                  
		for (int j = addwidth; j < addwidth + width; ++j)
		{
			unsigned short* inputpos = inputrow + j; //����ͼ���i�е�j����������Ԫ�صĵ�ַ
			unsigned short* outputimgpos = outputimgrow + j - addwidth; //���ͼ���i�е�j����������Ԫ�صĵ�ַ
			//��ģ���С�ľֲ������ڣ�ѭ�������ۼӡ�
			for (int m = -addheight; m <= addheight; m++)
			{
				unsigned short* starinputrow = inputpos + m * tempwid;//����ͼ��
				unsigned short* templatedilaterow = mattemplatedilate.pdata + (m + addheight) * templatedilatewidth;//ģ�����׵�ַ
				for (int n = -addwidth; n <= addwidth; n++)
				{
					unsigned short* templatedilatepos = templatedilaterow + n + addwidth;//ģ�嵱ǰֵ
					unsigned short* starinputpos = starinputrow + n;//ԭʼ���鵱ǰģ���еĶ�Ӧֵ
					//�ж�ֻҪģ������һ���ص���Ϊǰ��ֵ��������ĵ���Ϊ1
					if (*templatedilatepos == 1 && *starinputpos == 1)
					{
						*outputimgpos = 1;//ģ�����Ķ�Ӧ�����ͼ��Ĵ�������Ϊ1
						break;//������ǰģ��ڶ���ѭ��
					}
				}
				if (*outputimgpos == 1)//������ǰģ���һ��ѭ��
				{
					break;//����ģ��������һ������Ԫ�صı���������жϡ�
				}
			}
		}
	}
}

//matLableimg�����¶Ͽ�
void CTRUnet_Detection::GetorderDiaph(Matrix<unsigned short>& matLableimg, vector<int>& xvec_leftdiaph, vector<int>& yvec_leftdiaph,
	vector<int>& Xdiaph_Leftorder, vector<int>& Ydiaph_Leftorder)
{

	int width = matLableimg.width;
	int height = matLableimg.height;
	int imglength = width * height;

	int startX = xvec_leftdiaph[0];
	int startY = yvec_leftdiaph[0];

	//Matrix<unsigned short> matConectlabel(width, height);
	for (int i = 0; i < xvec_leftdiaph.size(); i++)
	{
		int index = yvec_leftdiaph[i] * width + xvec_leftdiaph[i];
		matLableimg.pdata[index] = 1;
	}

	////�����߶ε���㣬��������
	Search_Conectpoint(matLableimg, startX, startY, Xdiaph_Leftorder, Ydiaph_Leftorder);

	//��ǰ1/3������ȥ��
	int lengthstart = Xdiaph_Leftorder.size() / 3;
	for (int i = 0; i < lengthstart; i++)
	{
		Xdiaph_Leftorder.erase(Xdiaph_Leftorder.begin() + 0);
		Ydiaph_Leftorder.erase(Ydiaph_Leftorder.begin() + 0);
	}

	for (int i = 0; i < Ydiaph_Leftorder.size(); i++)
	{
		matLableimg.pdata[Ydiaph_Leftorder[i] * width + Xdiaph_Leftorder[i]] = 1;
	}

	int dd = 4;
}

void CTRUnet_Detection::Search_Conectpoint(Matrix<unsigned short>& matConectlabel, int startX, int startY, vector<int>& orderX, vector<int>& orderY)
{
	int width = matConectlabel.width;
	int height = matConectlabel.height;

	int clasenumber = 0;//��ͨ����
	//������������飬�����Ͻǿ�ʼ��˳ʱ�����������
	int neibx[8] = { -1, 0, 1, 1, 1, 0, -1, -1 };//������x����
	int neiby[8] = { -1, -1, -1, 0, 1, 1, 1, 0 };//������y����

	int i = startY;
	int j = startX;

	unsigned short* inputimgpos = matConectlabel.pdata + i * width + j;
	int deter = *inputimgpos;

	if (deter != 0)
	{
		clasenumber = clasenumber + 1;//��ͨ���ż�һ
		*inputimgpos = 0;//��ͼ��˵���Ϊ0        
		// 
	   //��ȡ�˴����ĵ�ĺ�������
	   //�����ж�
		int runvalue = 0;
		int countnumber = 0; // �������
		orderX.push_back(j);
		orderY.push_back(i); // ������洢��vector����������             

		while (runvalue <= countnumber)
		{
			for (int k = 0; k < 8; k++)
			{//����
				int nx = orderX[runvalue] + neibx[k];
				int ny = orderY[runvalue] + neiby[k];
				if (nx < width && nx >= 0 && ny < height && ny >= 0)//����������ͼ��ߴ���������
				{
					unsigned short* nebinputimgpos = matConectlabel.pdata + ny * width + nx;//����ͼ������λ��
					if (*nebinputimgpos == 1)
					{
						countnumber = countnumber + 1;
						orderX.push_back(nx);// �洢�������ĺ�����
						orderY.push_back(ny);// �洢��������������
						*nebinputimgpos = 0;//ԭʼ����ͼ���������ֵ��Ϊ0
					}
				}
			}
			runvalue = runvalue + 1;//  ׷����ͨ�����������Ŀ��ֱ��׷��countnumber 
		}

	}

}

void CTRUnet_Detection::GetHeartDiaphPoint_leftlung(vector<int>& Xdiaph_order, vector<int>& Ydiaph_order, int& maxdisind, vector<float>& nvect_ratio, int leftmaxdis_Y, int leftrightparm)
{
	//��δ���
	int w = Xdiaph_order.size();
	int detapix = 20;//����

	int numx = w / detapix;//����
	int startpix = detapix / 2;

	//���ߵ�ƽ������
	vector<float> nXXmeanvect;
	vector<float> nYYmeanvect;
	Line_Smooth(Xdiaph_order, Ydiaph_order, detapix, nXXmeanvect, nYYmeanvect);

	//б�ʼ���
	//���߿�ͷ��б�ʼ���
	Line_Ratio(nXXmeanvect, nYYmeanvect, detapix, nvect_ratio);

	//���ξ���ļ���
	vector<double> Vertical_disvector;//������ÿһ��ĵ���β�����Ĵ���
	ARC_Distance(nXXmeanvect, nYYmeanvect, Vertical_disvector, leftrightparm);

	int startind = 0;
	int endind = 0;
	for (int i = 0; i < nYYmeanvect.size(); i++)
	{
		int compr = Ydiaph_order[i];
		if (compr == leftmaxdis_Y)//�����ҷε�Y�����������20�����صľ�����б���
		{
			startind = i;
			break;
		}
		else
		{
			startind = Vertical_disvector.size() / 3;//��1/3����ʼѰ��
		}
	}

	//�ҵ����������Ǹ����λ��
	maxdisind = max_element(Vertical_disvector.begin() + startind, Vertical_disvector.end())
		- Vertical_disvector.begin();
}

void CTRUnet_Detection::GetHeartDiaphPoint_rightlung(vector<int>& Xdiaph_order, vector<int>& Ydiaph_order, int& maxdisind, vector<float>& nvect_ratio, int leftrightparm)
{
	int w = Xdiaph_order.size();
	int detapix = 20;//����

	int numx = w / detapix;//����
	int startpix = detapix / 2;

	//���ߵ�ƽ������
	vector<float> nXXmeanvect;
	vector<float> nYYmeanvect;
	Line_Smooth(Xdiaph_order, Ydiaph_order, detapix, nXXmeanvect, nYYmeanvect);

	//б�ʼ���
	//���߿�ͷ��б�ʼ���
	Line_Ratio(nXXmeanvect, nYYmeanvect, detapix, nvect_ratio);

	//���ξ���ļ���
	vector<double> Vertical_disvector;//������ÿһ��ĵ���β�����Ĵ���
	ARC_Distance(nXXmeanvect, nYYmeanvect, Vertical_disvector, leftrightparm);

	//�ҵ����������Ǹ����λ��
	maxdisind = max_element(Vertical_disvector.begin(), Vertical_disvector.end())
		- Vertical_disvector.begin();
}

void CTRUnet_Detection::Line_Smooth(vector<int>& Xdiaph_order, vector<int>& Ydiaph_order,
	int detapix, vector<float>& nXXmeanvect, vector<float>& nYYmeanvect)
{
	int w = Xdiaph_order.size();

	int numx = w / detapix;//����
	int startpix = detapix / 2;

	//���߿�ͷ���ƶ�ƽ������
	for (int i = 0; i < startpix; i++)
	{
		float sumtempx = 0;
		float sumtempy = 0;
		int sumcount = 0;
		for (int j = i; j < startpix; j++)
		{
			sumtempx = sumtempx + Xdiaph_order[j];
			sumtempy = sumtempy + Ydiaph_order[j];
			sumcount++;
		}

		float tempx = sumtempx / (sumcount + 1e-10);
		float tempy = sumtempy / (sumcount + 1e-10);

		nXXmeanvect.push_back(tempx);
		nYYmeanvect.push_back(tempy);

	}

	//����������ƶ�ƽ������
	for (int i = startpix; i < numx * detapix - startpix; i++)
	{
		float sumtempx = 0;
		float sumtempy = 0;
		int sumcount = 0;
		for (int j = i - startpix; j < i + startpix; j++)
		{
			sumtempx = sumtempx + Xdiaph_order[j];
			sumtempy = sumtempy + Ydiaph_order[j];
			sumcount++;
		}
		float tempx = sumtempx / (sumcount + 1e-10);
		float tempy = sumtempy / (sumcount + 1e-10);

		nXXmeanvect.push_back(tempx);
		nYYmeanvect.push_back(tempy);
	}

	//����ĩβ���ƶ�ƽ�����㣬ֱ�����һ������
	for (int i = numx * detapix - startpix; i < w; i++)
	{
		float sumtempx = 0;
		float sumtempy = 0;
		int sumcount = 0;
		for (int j = i; j < w; j++)
		{
			sumtempx = sumtempx + Xdiaph_order[j];
			sumtempy = sumtempy + Ydiaph_order[j];
			sumcount++;
		}

		float tempx = sumtempx / (sumcount + 1e-10);
		float tempy = sumtempy / (sumcount + 1e-10);

		nXXmeanvect.push_back(tempx);
		nYYmeanvect.push_back(tempy);
	}
}

void CTRUnet_Detection::Line_Ratio(vector<float>& nXXmeanvect, vector<float>& nYYmeanvect,
	int detapix, vector<float>& nvect_ratio)
{
	int w = nXXmeanvect.size();

	int numx = w / detapix;//����
	int startpix = detapix / 2;

	//���߿�ͷ��б�ʵļ���
	for (int i = 0; i < startpix; i++)
	{
		float nx1 = nXXmeanvect[i];
		float nx2 = nXXmeanvect[i + detapix];

		float ny1 = nYYmeanvect[i];
		float ny2 = nYYmeanvect[i + detapix];

		if (abs(ny1 - ny2) < 1e-5)
		{
			nvect_ratio.push_back(10);
		}
		else
		{
			float detax = nx2 - nx1;
			float detay = ny2 - ny1;

			float ratiovalue = detax / detay;
			if (ratiovalue >= 10)
			{
				ratiovalue = 10;
			}

			if (ratiovalue <= -10)
			{
				ratiovalue = -10;
			}
			nvect_ratio.push_back(ratiovalue);
		}
	}

	//�����������ݵ�б�ʼ���
	for (int i = startpix; i < numx * detapix - startpix; i++)
	{
		float nx1 = nXXmeanvect[i - startpix];
		float nx2 = nXXmeanvect[i + startpix];

		float ny1 = nYYmeanvect[i - startpix];
		float ny2 = nYYmeanvect[i + startpix];

		if (abs(ny1 - ny2) < 1e-5)
		{
			nvect_ratio.push_back(10);
		}
		else
		{
			float detax = nx2 - nx1;
			float detay = ny2 - ny1;

			float ratiovalue = detax / detay;
			if (ratiovalue >= 10)
			{
				ratiovalue = 10;
			}
			if (ratiovalue <= -10)
			{
				ratiovalue = -10;
			}
			nvect_ratio.push_back(ratiovalue);
		}
	}

	//���߽�β��б�ʼ���
	for (int i = numx * detapix - startpix; i < w; i++)
	{
		float nx1 = nXXmeanvect[i - detapix];
		float nx2 = nXXmeanvect[i];

		float ny1 = nYYmeanvect[i - detapix];
		float ny2 = nYYmeanvect[i];

		if (abs(ny1 - ny2) < 1e-5)
		{
			nvect_ratio.push_back(10);//               
		}
		else
		{
			float detax = nx2 - nx1;
			float detay = ny2 - ny1;

			float ratiovalue = detax / detay;
			if (ratiovalue >= 10)
			{
				ratiovalue = 10;
			}
			if (ratiovalue <= -10)
			{
				ratiovalue = -10;
			}
			nvect_ratio.push_back(ratiovalue);
		}
	}
}

void CTRUnet_Detection::ARC_Distance(vector<float>& nXXmeanvect, vector<float>& nYYmeanvect, vector<double>& Vertical_disvector, int posnegvalue)
{
	// ���ξ���
	//��β֮��Ļ�����
	vector<double> StEndvector;//��������Ԫ�أ���һ����X���ڶ�����Y
	StEndvector.resize(2);
	StEndvector[0] = nXXmeanvect[nXXmeanvect.size() - 1] - nXXmeanvect[0];
	StEndvector[1] = nYYmeanvect[nYYmeanvect.size() - 1] - nYYmeanvect[0];

	double Dis_StEndvector = 0;
	Dis_StEndvector = sqrtl(StEndvector[0] * StEndvector[0] +
		StEndvector[1] * StEndvector[1]);

	vector<double> Midvector;
	Midvector.resize(2);
	double Dis_Midvector = 0;
	double costheta = 0;
	double sintheta = 0;

	for (int i = 0; i < nYYmeanvect.size(); i++)
	{
		//������ÿһ�㵽��ʼ�������
		Midvector[0] = nXXmeanvect[i] - nXXmeanvect[0];
		Midvector[1] = nYYmeanvect[i] - nYYmeanvect[0];

		//������ÿһ�㵽��ʼ��ľ���
		Dis_Midvector = sqrtl(Midvector[0] * Midvector[0] +
			Midvector[1] * Midvector[1]);

		//������ÿһ�㵽��ʼ�����������β����֮��ļн�
		//������ÿһ�㵽��ʼ�����������β����֮��ļн�
		if ((Dis_StEndvector * Dis_Midvector) > 1e-10)
		{
			sintheta = posnegvalue * (Midvector[0] * StEndvector[1] -
				StEndvector[0] * Midvector[1]) /
				(Dis_Midvector * Dis_StEndvector);
		}
		else
		{
			sintheta = 0;
		}

		//������ÿһ��Ĵ���
		Vertical_disvector.push_back(Dis_Midvector * sintheta);
	}
}

void CTRUnet_Detection::Find_hartdiaphind_smallheart(vector<int>& nXXsmall_vect, vector<int>& nYYsmall_vect,
	vector<float>& nvect_ratio, int& maxdisind, int& hart_Leftind, int& diaph_Leftind, int leftrightparm)
{
	//����б�ʵľ���ֵ�����Ǹ�ֵ���ҵ�����λ�õ���Ч��
	int detadis = 100;//100
	hart_Leftind = maxdisind;
	if ((maxdisind - detadis) > 0)
	{
		int detapix = 10;
		//���߹յ��β����ǰ10����λ��б�����¼���
		for (int i = maxdisind - detapix; i < maxdisind; i++)
		{
			float nx1 = nXXsmall_vect[i - detapix];
			float nx2 = nXXsmall_vect[i];

			float ny1 = nYYsmall_vect[i - detapix];
			float ny2 = nYYsmall_vect[i];

			if (abs(ny1 - ny2) < 1e-5)
			{
				nvect_ratio[i] = 10;// �յ�ǰ��ǰ10����λ           
			}
			else
			{
				float detax = nx2 - nx1;
				float detay = ny2 - ny1;

				float ratiovalue = detax / detay;
				if (ratiovalue >= 10)
				{
					ratiovalue = 10;
				}
				if (ratiovalue <= -10)
				{
					ratiovalue = -10;
				}
				nvect_ratio[i] = ratiovalue;
			}
		}

		int tempind = maxdisind - detadis;
		hart_Leftind = tempind;
		double hartchest = abs(nvect_ratio[tempind]);

		int hart_smallX = nXXsmall_vect[tempind];
		int Current_smallX = tempind;

		for (int i = 0; i < detadis - 10; i++)//�������
		{
			if (hartchest > abs(nvect_ratio[tempind + i]))//��б��ģֵ��С��
			{
				if (abs(nvect_ratio[tempind + i]) != 10)
				{
					hartchest = abs(nvect_ratio[tempind + i]);
					hart_Leftind = tempind + i;
				}
			}

			if (leftrightparm == -1)//�Ҳ�
			{
				if (nXXsmall_vect[tempind + i] > hart_smallX)//�ҵ�����࣬�������������ߣ��Ҳ������ұ�
				{
					hart_smallX = nXXsmall_vect[tempind + i];
					Current_smallX = tempind + i;
				}
			}
			else//���
			{
				if (nXXsmall_vect[tempind + i] < hart_smallX)//�ҵ�����࣬�������������ߣ��Ҳ������ұ�
				{
					hart_smallX = nXXsmall_vect[tempind + i];
					Current_smallX = tempind + i;
				}
			}
		}

		if (leftrightparm == -1)//�Ҳ�
		{
			if (nXXsmall_vect[Current_smallX] > nXXsmall_vect[hart_Leftind])
			{
				hart_Leftind = Current_smallX;
			}
		}
		else//���
		{
			if (nXXsmall_vect[Current_smallX] < nXXsmall_vect[hart_Leftind])
			{
				hart_Leftind = Current_smallX;
			}
		}
	}
	else
	{
		int tempind = maxdisind ;
		hart_Leftind = tempind;
		double hartchest = abs(nvect_ratio[tempind]);

		int hart_smallX = nXXsmall_vect[tempind];
		int Current_smallX = tempind;

		for (int i = tempind; i >= 0; i--)//�������
		{
			if (hartchest > abs(nvect_ratio[i]))//��б��ģֵ��С��
			{
				if (abs(nvect_ratio[i]) != 10)
				{
					hartchest = abs(nvect_ratio[i]);
					hart_Leftind = i;
				}
			}

			if (leftrightparm == -1)//�Ҳ�
			{
				if (nXXsmall_vect[i] > hart_smallX)//�ҵ�����࣬�������������ߣ��Ҳ������ұ�
				{
					hart_smallX = nXXsmall_vect[i];
					Current_smallX = i;
				}
			}
			else//���
			{
				if (nXXsmall_vect[i] < hart_smallX)//�ҵ�����࣬�������������ߣ��Ҳ������ұ�
				{
					hart_smallX = nXXsmall_vect[i];
					Current_smallX = i;
				}
			}
		}

		if (leftrightparm == -1)//�Ҳ�
		{
			if (nXXsmall_vect[Current_smallX] > nXXsmall_vect[hart_Leftind])
			{
				hart_Leftind = Current_smallX;
			}
		}
		else//���
		{
			if (nXXsmall_vect[Current_smallX] < nXXsmall_vect[hart_Leftind])
			{
				hart_Leftind = Current_smallX;
			}
		}
	}

	//����б�ʵľ���ֵ�����Ǹ�ֵ���ҵ�������λ��
	//detadis = 50;
	//diaph_Leftind = maxdisind;
	//if ((maxdisind + detadis) < nvect_ratio.size())
	//{
	//	int tempind = maxdisind;
	//	double diaphchest = abs(nvect_ratio[tempind]);
	//	for (int i = 20; i < detadis; i++)
	//	{
	//		if (diaphchest < abs(nvect_ratio[tempind + i]))
	//		{
	//			diaphchest = abs(nvect_ratio[tempind + i]);
	//			diaph_Leftind = tempind + i;
	//		}
	//	}
	//}
	//else
	//{
	//	int tempind = maxdisind;
	//	double diaphchest = abs(nvect_ratio[tempind]);
	//	for (int i = maxdisind; i < nvect_ratio.size(); i++)//�ҵ�б��ģֵ����
	//	{
	//		if (diaphchest < abs(nvect_ratio[i]))
	//		{
	//			diaphchest = abs(nvect_ratio[i]);
	//			diaph_Leftind =  i;
	//		}
	//	}
	//}

	int tempind = maxdisind;
	int endstar = maxdisind + (nYYsmall_vect.size() - maxdisind) / 5 * 4;
	if (endstar > maxdisind + 10)
	{
		int diaphchest = abs(nYYsmall_vect[tempind + 10]);
		diaph_Leftind = maxdisind + 10;
		for (int i = endstar; i > maxdisind + 5; i--)
		{
			if (diaphchest > abs(nYYsmall_vect[i]))
			{
				diaphchest = abs(nYYsmall_vect[i]);
				diaph_Leftind = i;
			}
		}
	}
	else
	{
		diaph_Leftind = maxdisind;
	}
}


void CTRUnet_Detection::Find_hartdiaphind_largeheart(vector<int>& nXXlarge_vect, vector<int>& nYYlarge_vect,
	vector<float>& nvect_rightratio, int& maxdisind, int& hart_Rightind, int& diaph_Rightind, int leftrightparm)//leftrightparm���Ϊ1���Ҳ�Ϊ-1
{
	// ����б�ʵľ���ֵ��ƽ����λ�ôӶ��ҵ������Եλ�� 
	int detadis = 50;//50  
	hart_Rightind = maxdisind;
	if ((maxdisind - detadis) > 0)
	{
		int detapix = 10;
		//���߹յ��β����ǰ10����λ��б�����¼���
		for (int i = maxdisind - detapix; i < maxdisind; i++)
		{
			float nx1 = nXXlarge_vect[i - detapix];
			float nx2 = nXXlarge_vect[i];

			float ny1 = nYYlarge_vect[i - detapix];
			float ny2 = nYYlarge_vect[i];

			if (abs(ny1 - ny2) < 1e-5)
			{
				nvect_rightratio[i] = 10;// �յ�ǰ��ǰ10����λ           
			}
			else
			{
				float detax = nx2 - nx1;
				float detay = ny2 - ny1;

				float ratiovalue = detax / detay;
				if (ratiovalue >= 10)
				{
					ratiovalue = 10;
				}
				if (ratiovalue <= -10)
				{
					ratiovalue = -10;
				}
				nvect_rightratio[i] = ratiovalue;
			}
		}

		int tempind = maxdisind - detadis;
		hart_Rightind = tempind;
		double hartchest = abs(nvect_rightratio[tempind]);

		int hart_RightX = nXXlarge_vect[tempind];
		int Current_RightX = tempind;
		for (int i = 0; i < detadis - 5; i++)//��ǰdetadis�����ݵ�б����Ѱ��б��ģֵ��С���������
		{
			if (hartchest > abs(nvect_rightratio[tempind + i]))//�ҵ�б��ģֵ��С��
			{
				if (abs(nvect_rightratio[tempind + i]) != 10)
				{
					hartchest = abs(nvect_rightratio[tempind + i]);
					hart_Rightind = tempind + i;
				}
			}

			if (leftrightparm == -1)//�Ҳ�
			{
				if (nXXlarge_vect[tempind + i] > hart_RightX)//�ҵ�����࣬�������������ߣ��Ҳ������ұ�
				{
					hart_RightX = nXXlarge_vect[tempind + i];
					Current_RightX = tempind + i;
				}
			}
			else//���
			{
				if (nXXlarge_vect[tempind + i] < hart_RightX)//�ҵ�����࣬�������������ߣ��Ҳ������ұ�
				{
					hart_RightX = nXXlarge_vect[tempind + i];
					Current_RightX = tempind + i;
				}
			}

		}

		if (leftrightparm == -1)//�Ҳ�
		{
			if (nXXlarge_vect[Current_RightX] > nXXlarge_vect[hart_Rightind])
			{
				hart_Rightind = Current_RightX;
			}
		}
		else//���
		{
			if (nXXlarge_vect[Current_RightX] < nXXlarge_vect[hart_Rightind])
			{
				hart_Rightind = Current_RightX;
			}
		}

	}
	else
	{
		int tempind = maxdisind;
		hart_Rightind = tempind;
		double hartchest = abs(nvect_rightratio[tempind]);

		int hart_RightX = nXXlarge_vect[tempind];
		int Current_RightX = tempind;
		for (int i = tempind; i >=0; i--)//��ǰdetadis�����ݵ�б����Ѱ��б��ģֵ��С���������
		{
			if (hartchest > abs(nvect_rightratio[i]))//�ҵ�б��ģֵ��С��
			{
				if (abs(nvect_rightratio[i]) != 10)
				{
					hartchest = abs(nvect_rightratio[i]);
					hart_Rightind = i;
				}
			}

			if (leftrightparm == -1)//�Ҳ�
			{
				if (nXXlarge_vect[i] > hart_RightX)//�ҵ�����࣬�������������ߣ��Ҳ������ұ�
				{
					hart_RightX = nXXlarge_vect[ i];
					Current_RightX = i;
				}
			}
			else//���
			{
				if (nXXlarge_vect[i] < hart_RightX)//�ҵ�����࣬�������������ߣ��Ҳ������ұ�
				{
					hart_RightX = nXXlarge_vect[ i];
					Current_RightX = i;
				}
			}
		}

		if (leftrightparm == -1)//�Ҳ�
		{
			if (nXXlarge_vect[Current_RightX] > nXXlarge_vect[hart_Rightind])
			{
				hart_Rightind = Current_RightX;
			}
		}
		else//���
		{
			if (nXXlarge_vect[Current_RightX] < nXXlarge_vect[hart_Rightind])
			{
				hart_Rightind = Current_RightX;
			}
		}
	}

	////����б�ʵľ���ֵ�����Ǹ�ֵ���ҵ�������λ��
	//detadis = 30;
	//diaph_Rightind = maxdisind;
	//if ((maxdisind + detadis) < nvect_rightratio.size())
	//{
	//	int tempind = maxdisind;
	//	double diaphchest = abs(nvect_rightratio[tempind]);
	//	for (int i = 5; i < detadis; i++)
	//	{
	//		if (diaphchest < abs(nvect_rightratio[tempind + i]))//�ҵ�б������
	//		{
	//			diaphchest = abs(nvect_rightratio[tempind + i]);
	//			diaph_Rightind = tempind + i;
	//		}
	//	}
	//}
	//else
	//{
	//	int tempind = maxdisind;
	//	double diaphchest = abs(nvect_rightratio[tempind]);
	//	for (int i = maxdisind; i < nvect_rightratio.size(); i++)
	//	{
	//		if (diaphchest < abs(nvect_rightratio[i]))
	//		{
	//			diaphchest = abs(nvect_rightratio[i]);
	//			diaph_Rightind = i;
	//		}
	//	}
	//}

	
	int tempind = maxdisind;
	int endstar = maxdisind + (nYYlarge_vect.size() - maxdisind) / 5 * 4;

	if (endstar > maxdisind + 10)
	{
	    int diaphchest = abs(nYYlarge_vect[tempind+10]);
	    diaph_Rightind = maxdisind+10;
		for (int i = endstar; i > maxdisind + 5; i--)
		{
			if (diaphchest > abs(nYYlarge_vect[i]))
			{
				diaphchest = abs(nYYlarge_vect[i]);
				diaph_Rightind = i;
			}
		}
	}
	else
	{
		diaph_Rightind = maxdisind;
	}
}


void CTRUnet_Detection::LeftRight_dist(int& LeftdisX, int& RightdisX, int Midseg_X, Matrix<unsigned short>& matLeftdis, Matrix<unsigned short>& matRightdis,
	vector<int>& xribdataleft, vector<int>& yribdataleft,
	vector<int>& xribdataright, vector<int>& yribdataright,
	vector<int>& Xdiaph_Leftorder, vector<int>& Ydiaph_Leftorder,
	vector<int>& Xdiaph_Rightorder, vector<int>& Ydiaph_Rightorder)
{
	int leftrib_Ymin = *min_element(yribdataleft.begin(), yribdataleft.end());
	int leftrib_Ymax = *max_element(yribdataleft.begin(), yribdataleft.end());
	int rightrib_Ymin = *min_element(yribdataright.begin(), yribdataright.end());
	int rightrib_Ymax = *max_element(yribdataright.begin(), yribdataright.end());

	int min_Y = max(leftrib_Ymin, rightrib_Ymin);
	int max_Y = min(leftrib_Ymax, rightrib_Ymax);

	int mid_Yst = min_Y+abs(max_Y-min_Y) / 2;
	int mid_Yend = min_Y + abs(max_Y - min_Y) / 4 * 3;

	//��๹��
	for (int i = 0; i < Ydiaph_Leftorder.size(); i++)
	{
		matLeftdis.pdata[Ydiaph_Leftorder[i] * matLeftdis.width + Xdiaph_Leftorder[i]] = 1;
	}

	//�Ҳ๹��
	for (int i = 0; i < Ydiaph_Rightorder.size(); i++)
	{
		matRightdis.pdata[Ydiaph_Rightorder[i] * matRightdis.width + Xdiaph_Rightorder[i]] = 1;
	}

	//��������ݸ���������ľ���
	int sum_X = 1;
	int coutnum = 1;
	for (int i = mid_Yst, step1 = mid_Yst * matLeftdis.width; i < mid_Yend; i++, step1 += matLeftdis.width)
	{
		for (int j = matLeftdis.width - 1; j >= 0; j--)
		{
			if (matLeftdis.pdata[step1 + j] != 0)
			{
				sum_X = sum_X + (Midseg_X - j);
				coutnum++;
				break;
			}
		}
	}
	LeftdisX = sum_X / (coutnum+1e-6);

	//�����Ҳ��ݸ���������ľ���
	sum_X = 1;
	coutnum = 1;
	for (int i = mid_Yst, step1 = mid_Yst * matRightdis.width; i < mid_Yend; i++, step1 += matRightdis.width)
	{
		for (int j = 0; j < matRightdis.width; j++)
		{
			if (matRightdis.pdata[step1 + j] != 0)
			{
				sum_X = sum_X + (j - Midseg_X);
				coutnum++;
				break;
			}
		}
	}

	RightdisX = sum_X / (coutnum+1e-6);
}

int CTRUnet_Detection::Erase_holl(Matrix<unsigned short>& matInputimg, double ratiovalue)
{
	int signum = 1;

	int width = matInputimg.width;
	int height = matInputimg.height;

	int imglength = width * height;
	Matrix<unsigned short> imgcomple(width, height);

	//������ͼ�����ǰ���ͱ�����ת
	for (int i = 0; i < imglength; i++)
	{
		if (matInputimg.pdata[i] == 0)
		{
			imgcomple.pdata[i] = 1;
		}
		else
		{
			imgcomple.pdata[i] = 0;
		}
	}

	//���������׶�
	Matrix<unsigned short> imgcomple_inv(width, height);
	signum = Eliminateholes(imgcomple, imgcomple_inv, ratiovalue);//���������׶�
	if (signum != 1)
	{
		//MYLog.WriteLog(1,"���������׶�������ͼ��û����ͨ��");
		return 0;
	}

	//�ٴ�ǰ���ͱ�����ת
	for (int i = 0; i < imglength; i++)
	{
		if (imgcomple_inv.pdata[i] == 0)
		{
			imgcomple.pdata[i] = 1;
		}
		else
		{
			imgcomple.pdata[i] = 0;
		}
	}

	memset(imgcomple_inv.pdata, 0, sizeof(unsigned short) * imglength);
	signum = Eliminateholes(imgcomple, imgcomple_inv, ratiovalue);//����ǰ����  
	if (signum != 1)
	{
		//MYLog.WriteLog(1, "����ǰ���׶�������ͼ��û����ͨ��");
		return 0;
	}

	memcpy(matInputimg.pdata, imgcomple_inv.pdata, sizeof(unsigned short) * imglength);
	return 1;
}

int CTRUnet_Detection::Eliminateholes(Matrix<unsigned short>& matInputimg, Matrix<unsigned short>& matOutputimg, double ratiovalue)
{
	int width = matInputimg.width;
	int height = matInputimg.height;

	int widthheight = width * height;

	Matrix<unsigned short> matBWimgblabel(width, height);
	vector<int> labind;
	vector<int> labval;
	//labnum����ֵ������ͨ��Ĵ�С������ֵ������ͨ��ı�ţ����������bwimglabel����ͨ��ͼ��
	Conectchose(matInputimg, matBWimgblabel, labval, labind, FALSE);
	if (labind.empty())
	{
		return 0;
	}

	//��������ͨ��Ĵ�С���
	double connectsum = accumulate(labval.begin(), labval.end(), 0);
	double uptovalue = connectsum * ratiovalue;//����ֵ������ͨ����ֵ��ȡ��ͨ���ֵֹ
	int uptoind = -1;
	for (int i = 0; i < labval.size(); i++)
	{
		if (labval[i] > uptovalue)
		{
			uptoind = i; //��ȡ��ͨ��Ľ�ֹ���� ��
			break;
		}
		else
		{
			uptoind = -1;
		}
	}

	//�����׶����or��ͨ��ɸѡ���жϺ�ѭ��
	if (uptoind != -1)//˵����ȡ������ͨ��Ľ�������
	{
		for (int i = 0; i < widthheight; i++)
		{
			if (matBWimgblabel.pdata[i] != 0)
			{
				int labelnumber = matBWimgblabel.pdata[i];//ͼ���е���ͨ����
				for (int j = 0; j < uptoind; j++)//����С��ͨ��ı��
				{
					if (labelnumber == labind[j])//���С��ͨ��ı�ź�ͼ���еı�����Ӧ����и�ֵΪ0�Ĳ���
					{
						matBWimgblabel.pdata[i] = 0; // ֻҪ���㼴����Ϊ��ֵΪ0
						break;
					}
				}
			}
		}
	}
	else
	{
		//KrayLog.WriteLog(1, "Eliminateholes function is something wrong, the inputimg Conection is too small");
		return 0;
	}

	for (int i = 0; i < widthheight; i++)
	{
		if (matBWimgblabel.pdata[i] != 0)//���벻Ϊ0����ͨ��
		{
			matOutputimg.pdata[i] = 1;//���뵽�������ӿ���
		}
	}
	return 1;
}


int CTRUnet_Detection::LeftRightsideimg(Matrix<unsigned short>& matmasklung, Matrix<unsigned short>& matleftlungmask_copy, Matrix<unsigned short>& matrightlungmask_copy)
{
	int width = matmasklung.width;
	int height = matmasklung.height;
	Matrix<unsigned short> matmasklung_copy(matmasklung.width, matmasklung.height);
	Matrix<unsigned short> matinputmasklung(matmasklung.width, matmasklung.height);

	//��ֵmatinputmasklung������Ϊ��ͨ�������������
	for (int i = 0; i < width*height; i++)
	{
		if (matmasklung.pdata[i] != 0)
		{
			matinputmasklung.pdata[i] = 1;
		}
	}

	//
	vector<int> labind;
	vector<int> labval;
	//labnum����ֵ������ͨ��Ĵ�С������ֵ������ͨ��ı�ţ����������bwimglabel����ͨ��ͼ��
	Conectchose(matinputmasklung, matmasklung_copy, labval, labind, FALSE);
	if (labind.empty())
	{
		return 0;
	}

	int sigm = 0;
	int marknum = 0;

	//���ϵ��£������ң���ȡ���ķβ�����
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			if (matmasklung_copy.pdata[j*width + i] != 0)
			{
				marknum = matmasklung_copy.pdata[j*width + i];
				sigm = 1;
				break;
			}
		}
		if (sigm==1)
		{
			break;
		}
	}

	//����Ұ
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (matmasklung_copy.pdata[i*width + j] == marknum)
			{
				matleftlungmask_copy.pdata[i*width + j] = 1;
			}
		}
	}

	//�Ӵ��ϵ��£����ҵ��󣬻�ȡ�Ҳ�ķβ�����
	sigm = 0;
	for (int i = width-1; i >= 0; i--)
	{
		for (int j = 0; j < height; j++)
		{
			if (matmasklung_copy.pdata[j*width + i] != 0)
			{
				marknum = matmasklung_copy.pdata[j*width + i];
				sigm = 1;
				break;
			}
		}
		if (sigm == 1)
		{
			break;
		}
	}

	//�Ҳ��Ұ
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (matmasklung_copy.pdata[i*width + j] == marknum)
			{
				matrightlungmask_copy.pdata[i*width + j] = 1;
			}
		}
	}

	return 1;
}

//ֱ�������㷨
void CTRUnet_Detection::DDALine11(unsigned short* neblabelimgpos, int startx, int starty, int endx, int endy, int pixelvalue, int width, int height)
{
	double x, y, tmp, xIncre, yIncre;
	int dx = endx - startx;
	int dy = endy - starty;
	tmp = max(abs(dx), abs(dy));
	xIncre = 1.0 * dx / tmp;  // ����
	yIncre = 1.0 * dy / tmp;
	x = startx;  // ��ǰ��Ҫ���Ƶĵ�
	y = starty;
	for (int i = 0; i < tmp; i++)
	{
		x += xIncre;
		y += yIncre;
		neblabelimgpos[int(x + 0.5) + int(y + 0.5)*width] = pixelvalue;
	}
}

//���رȺ��������ӳ��
void CTRUnet_Detection::MaptoOrg_CTR(int imgwidth, int imgheight, int downsampwidth, int downsampheight, int params[])
{
	//��paramsY����ӳ��
	for (int i = 3; i <= 26; i += 2)
	{
		params[i] = round(float(params[i]+1) / downsampheight * imgheight);
	}
	//��paramsX����ӳ��
	for (int i = 4; i <= 26; i += 2)
	{
		params[i] = round(float(params[i]+1) / downsampwidth * imgwidth);
	}
}

//�����߽�ӳ��
void CTRUnet_Detection::MaptoOrg_Diaph(int imgwidth, int imgheight, int downsampwidth, int downsampheight, unsigned char* pdiaph_Line_img, unsigned short* pdiaph_Line_imgorg)
{
	//�Ժ���Ĥ�߽����ӳ��
	vector<int> nxvect;//��ͨ����������ɺ�
	vector<int> nyvect;//��ͨ�����������ɺ�
	int neibx[8] = { -1, 0, 1, 1, 1, 0, -1, -1 }; //������x����
	int neiby[8] = { -1, -1, -1, 0, 1, 1, 1, 0 }; //������y����
   
 	//�ԷŴ�Ĳ��������߽���ӳ��                                        
	for (int i = 0; i < imgheight; i++)
	{
		for (int j = 0; j < imgwidth; j++)
		{

			int mapwidth = float(j+1) / float(imgwidth) * (downsampwidth - 1);
			int mapheight = float(i+1) / float(imgheight) * (downsampheight - 1);
			int index = mapheight * downsampwidth + mapwidth;
			if (pdiaph_Line_img[index] != 0)//���Сͼ��Ϊ0
			{
				int orgindex = i * imgwidth + j;
				pdiaph_Line_imgorg[orgindex] = pdiaph_Line_img[index];//��Сͼ�ڴ������ֵ��ֵ����ͼ
				pdiaph_Line_img[index] = 0;//Сͼ��0��                                             

				//�����ж�
				int runvalue = 0;
				int countnumber = 0; // �������
				nxvect.push_back(mapwidth);//
				nyvect.push_back(mapheight); // ��Сͼ��Ч����洢��vector����������   

				while (runvalue <= countnumber)
				{

					for (int k = 0; k < 8; k++)
					{
						int nx = nxvect[runvalue] + neibx[k];
						int ny = nyvect[runvalue] + neiby[k];

						//��ͼ���ͼ�������������λ��
						int centerwith = float(nxvect[runvalue]) / (downsampwidth - 1) * float(imgwidth - 1);
						int centerheight = float(nyvect[runvalue]) / (downsampheight - 1) * float(imgheight - 1);


						if (nx < downsampwidth && nx >= 0 && ny < downsampheight && ny >= 0)//����������Сͼ��ߴ���������
						{
							unsigned char* nebinputimgpos = pdiaph_Line_img + ny * downsampwidth + nx;//����ͼ������λ��

							//���ͼ�������������λ��
							int currwith = float(nx) / (downsampwidth - 1) * float(imgwidth);
							int currheight = float(ny) / (downsampheight - 1) * float(imgheight);

							int currindex = currheight * imgwidth + currwith;
							unsigned short* neblabelimgpos = pdiaph_Line_imgorg + currindex;//��ͼ�����ͼ���������λ��
							if (*nebinputimgpos != 0)//  Сͼ������Ϊ�棬��ִ��������
							{
								countnumber = countnumber + 1;
								nxvect.push_back(nx);//�洢Сͼ�������ĺ�����
								nyvect.push_back(ny);//�洢Сͼ��������������

								//��ͼ�������������ʼ��
								DDALine11(pdiaph_Line_imgorg, currwith, currheight, centerwith, centerheight, int(*nebinputimgpos), imgwidth, imgheight);//ֱ�߻��߷��������������꣬����ֱ�߲���ֵ
								*nebinputimgpos = 0;//ԭʼ����ͼ���������ֵ��Ϊ0
							}
						}
					}
					runvalue = runvalue + 1;//׷����Ч��
				}
			}
			nxvect.clear();
			nyvect.clear();
		}
	}
}

//���ҷ�ʶ��ͷ����ӳ��
void CTRUnet_Detection::MaptoOrg_Areacal(int imgwidth, int imgheight, int downsampwidth, int downsampheight, int params[], unsigned char* pleftrightlung_imgmask, unsigned short* pleftrightlung_imgmaskorg)
{
	//�����ҷβ�ʶ���������ӳ��
	for (int i = 0; i < imgheight; i++)
	{
		for (int j = 0; j < imgwidth; j++)
		{

			int mapwidth = float(j + 1) / float(imgwidth) * (downsampwidth - 1);
			int mapheight = float(i + 1) / float(imgheight) * (downsampheight - 1);
			pleftrightlung_imgmaskorg[i * imgwidth + j] = pleftrightlung_imgmask[mapheight * downsampwidth + mapwidth];
		}
	}

	//���������ӳ��
	params[27] = params[27] * (float(imgwidth) / downsampwidth)*(float(imgheight) / downsampheight);
	params[28] = params[28] * (float(imgwidth) / downsampwidth)*(float(imgheight) / downsampheight);
}