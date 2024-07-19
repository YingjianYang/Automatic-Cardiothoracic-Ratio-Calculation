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
	//连通域去除
	Erase_holl(matmasklungcopy, 0.05);//消除连通域小于0.05的区域
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

	//数据转换。
	Matrix<unsigned short> matmasklung(downsampwidth, downsampheight);
	//构建左右肺野区域。
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
	
	//左侧轮廓的顶点位置。
	int lefttop_Y = *min_element(yvec_left.begin(), yvec_left.end());
	int lefttop_X = xvec_left[min_element(yvec_left.begin(), yvec_left.end())- yvec_left.begin()];
	//左侧轮廓的底点位置。
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

	//右侧轮廓的顶点位置
	int righttop_Y= *min_element(yvec_right.begin(), yvec_right.end());
	int righttop_X = xvec_right[min_element(yvec_right.begin(), yvec_right.end())- yvec_right.begin()];
	//右侧轮廓的底点位置
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
	vector<int> yvec_leftdiaph;//左侧分割
	signum = RibDiaphSeg_left(matleftlungmask, xvec_left, yvec_left, xvec_leftrib, yvec_leftrib, xvec_leftdiaph, yvec_leftdiaph);
	if (signum != 1)
	{
		return 0;
	}

	vector<int> xvec_rightrib;
	vector<int> yvec_rightrib;
	vector<int> xvec_rightdiaph;
	vector<int> yvec_rightdiaph;//右侧分割
	signum = RibDiaphSeg_right(matrightlungmask, xvec_right, yvec_right, xvec_rightrib, yvec_rightrib, xvec_rightdiaph, yvec_rightdiaph);
	if (signum != 1)
	{
		return 0;
	}

	//寻找中心分割线
	int Rib_Leftind = *max_element(xvec_right.begin(), xvec_right.end());//找到最左侧的肋骨位置
	int Rib_Rightind = *min_element(xvec_left.begin(), xvec_left.end());//找到最右侧的肋骨位置
	//心胸比中间线的X方向分割线位置
	int Midseg_X = (Rib_Leftind + Rib_Rightind) / 2;

	//左侧横纵隔膜边界排序
	vector<int> Xdiaph_Leftorder;
	vector<int> Ydiaph_Leftorder;
	Matrix<unsigned short> matLableimg_left(downsampwidth, downsampheight);
	GetorderDiaph(matLableimg_left, xvec_leftdiaph, yvec_leftdiaph, Xdiaph_Leftorder, Ydiaph_Leftorder);//  

	//cv::Mat imgmat1(downsampheight, downsampwidth, CV_16UC1, matLableimg_left.pdata);
	//cv::Mat garay1;
	//cv::threshold(imgmat1, garay1, 0, 65535, cv::THRESH_BINARY);//二值化
	//cv::imshow("左侧横纵", garay1);
	//cv::waitKey(0);

	//右侧横纵隔膜边界排序
	vector<int> Xdiaph_Rightorder;
	vector<int> Ydiaph_Rightorder;
	Matrix<unsigned short> matLableimg_right(downsampwidth, downsampheight);
	GetorderDiaph(matLableimg_right, xvec_rightdiaph, yvec_rightdiaph, Xdiaph_Rightorder, Ydiaph_Rightorder);//  

	//cv::Mat imgmat2(downsampheight, downsampwidth, CV_16UC1, matLableimg_right.pdata);
	//cv::Mat garay2;
	//cv::threshold(imgmat2, garay2, 0, 65535, cv::THRESH_BINARY);//二值化
	//cv::imshow("右侧横纵", garay2);
	//cv::waitKey(0);

	//判断图像的左侧是左肺还是右肺,边界点包含于肺野点中，只需取其中一个点判断即可
	int diaph_Leftind = 0; //左侧横隔肌有效点的位置 。 //计算获取拐点
	int hart_Leftind = 0; //左侧心脏有效点的位置
	int leftmaxdisind;//左侧拐点的最大距离的索引位置

	int diaph_Rightind = 0; //右侧有效点的位置
	int hart_Rightind = 0; //右侧有效点的位置
	int rightmaxdisind;//右侧拐点的最大的索引位置

	int deta_Y = 20;

	//判断左右肺部区域
	int LeftdisX = 0;
	int RightdisX = 0;
	Matrix<unsigned short> matLeftdis(matleftlungmask.width, matleftlungmask.height);
	Matrix<unsigned short> matRightdis(matrightlungmask.width, matrightlungmask.height);
	LeftRight_dist(LeftdisX, RightdisX, Midseg_X, matLeftdis, matRightdis,
		xvec_leftrib, yvec_leftrib, xvec_rightrib, yvec_rightrib,
		Xdiaph_Leftorder, Ydiaph_Leftorder, Xdiaph_Rightorder, Ydiaph_Rightorder);

	if (LeftdisX < RightdisX)//(matmasklung.pdata[index] == 1)//等于1为右肺，左侧对应右肺
	{
		//先算右肺
		vector<float> nvect_leftratio;//右肺部每个点的斜率值,leftmaxdisind为右肺的斜率
		GetHeartDiaphPoint_rightlung(Xdiaph_Leftorder, Ydiaph_Leftorder, leftmaxdisind, nvect_leftratio, 1);//图像左侧传1，叉乘为正    
		//右肺斜率对应的Y坐标
		int leftmaxdis_Y = Ydiaph_Leftorder[leftmaxdisind];//左肺的拐点寻找就在此点的位置上下进行寻找
		leftmaxdis_Y = leftmaxdis_Y - deta_Y;
		//再算左肺
		vector<float> nvect_rightratio;//左肺侧连通域的每个点的斜率值
		GetHeartDiaphPoint_leftlung(Xdiaph_Rightorder, Ydiaph_Rightorder, rightmaxdisind, nvect_rightratio, leftmaxdis_Y, -1); //图像右侧传-1，叉乘为负

		//根据垂距最大的那个点通过斜率找到心脏边缘位置和横隔肌边缘位置 
		Find_hartdiaphind_smallheart(Xdiaph_Leftorder, Ydiaph_Leftorder, nvect_leftratio, leftmaxdisind, hart_Leftind, diaph_Leftind, 1);//leftmaxdisind

		//根据垂距最大的那个点通过斜率找到心脏边缘位置和横隔肌边缘位置 
		Find_hartdiaphind_largeheart(Xdiaph_Rightorder, Ydiaph_Rightorder, nvect_rightratio, rightmaxdisind, hart_Rightind, diaph_Rightind, -1);
	}
	else //if (matmasklung.pdata[index] == 2)//左侧对应左肺
	{
		//先算右肺
		vector<float> nvect_rightratio;//右肺部每个点的斜率值,leftmaxdisind为右肺的斜率
		GetHeartDiaphPoint_rightlung(Xdiaph_Rightorder, Ydiaph_Rightorder, rightmaxdisind, nvect_rightratio, -1);//图像右侧传-1  
		//右肺斜率对应的Y坐标
		int rightmaxdis_Y = Ydiaph_Rightorder[rightmaxdisind];//左肺的拐点寻找就在此点的位置上下进行寻找
		rightmaxdis_Y = rightmaxdis_Y - deta_Y;
		//再算左肺
		vector<float> nvect_leftratio;
		GetHeartDiaphPoint_leftlung(Xdiaph_Leftorder, Ydiaph_Leftorder, leftmaxdisind, nvect_leftratio, rightmaxdis_Y, 1); //图像左侧传1，

		//根据垂距最大的那个点通过斜率找到心脏边缘位置和横隔肌边缘位置 
		Find_hartdiaphind_smallheart(Xdiaph_Rightorder, Ydiaph_Rightorder, nvect_rightratio, rightmaxdisind, hart_Rightind, diaph_Rightind, -1);//leftmaxdisind

		//根据垂距最大的那个点通过斜率找到心脏边缘位置和横隔肌边缘位置 
		Find_hartdiaphind_largeheart(Xdiaph_Leftorder, Ydiaph_Leftorder, nvect_leftratio, leftmaxdisind, hart_Leftind, diaph_Leftind, 1);
	}
	/*else
	{
		;
	}*/

	//左侧图像的心脏和横隔肌左侧位置：心脏点                          
	int LefthartY = Ydiaph_Leftorder[hart_Leftind];
	int LefthartX = Xdiaph_Leftorder[hart_Leftind];

	//左侧图像横膈肌
	int LeftdiaphY = Ydiaph_Leftorder[diaph_Leftind];
	int LeftdiaphX = Xdiaph_Leftorder[diaph_Leftind];

	//右侧图像的心脏和横膈肌的右侧位置：心脏
	int RighthartY = Ydiaph_Rightorder[hart_Rightind];
	int RighthartX = Xdiaph_Rightorder[hart_Rightind];

	//右侧图像横膈肌
	int RightdiaphY = Ydiaph_Rightorder[diaph_Rightind];
	int RightdiaphX = Xdiaph_Rightorder[diaph_Rightind];

	//左侧拐点
	int LeftmaxdisY = Ydiaph_Leftorder[leftmaxdisind];
	int LeftmaxdisX = Xdiaph_Leftorder[leftmaxdisind];

	//右侧拐点
	int RightmaxdisY = Ydiaph_Rightorder[rightmaxdisind];
	int RightmaxdisX = Xdiaph_Rightorder[rightmaxdisind];

	//左右两侧肋骨点的确认
	if (LeftdisX < RightdisX)//(matmasklung.pdata[index] == 1)//左侧图像对应右肺
	{
		int coutnum = 0;
		//左侧肋骨边界有效点检测
		for (int i = 0; i < xvec_leftrib.size(); i++)
		{
			++coutnum;
			if (yvec_leftrib[i] == Ydiaph_Leftorder[diaph_Leftind])//与右膈肌检测点的Y值相同的肋骨点
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

		//右侧肋骨边界有效点检测
		int leftribeffect = Ydiaph_Leftorder[diaph_Leftind];
		vector<int> rightribX;
		vector<int> rightribY;
		for (int i = 0; i < yvec_rightrib.size(); i++)
		{
			if (yvec_rightrib[i] == leftribeffect)//与右膈肌检测点的Y值相同的肋骨点
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
	else//出现反向检测
	{

		//左侧肋骨边界有效点检测
		vector<int> leftribX;
		vector<int> leftribY;

		for (int i = 0; i < yvec_leftrib.size(); i++)
		{
			if (yvec_leftrib[i] == Ydiaph_Rightorder[diaph_Rightind])//与右膈肌检测点的Y值相同的肋骨点
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

		//右侧肋骨边界有效点检测
		for (int i = 0; i < yvec_rightrib.size(); i++)
		{
			++coutnum;
			if (yvec_rightrib[i] == Ydiaph_Rightorder[diaph_Rightind])//与右膈肌检测点的Y值相同的肋骨点
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

	params[7] = LefthartY;//左心脏Y坐标
	params[8] = LefthartX;//左心脏X坐标

	params[9] = RighthartY;//右心脏Y坐标
	params[10] = RighthartX;//右心脏X坐标

	params[11] = LeftdiaphY;//左横膈肌Y坐标
	params[12] = LeftdiaphX;//左横膈肌X坐标

	params[13] = RightdiaphY;//右横膈肌Y坐标
	params[14] = RightdiaphX;//右横膈肌X坐标

	params[15] = LeftmaxdisY;//左侧拐点y坐标
	params[16] = LeftmaxdisX;//左侧拐点x坐标

	params[17] = RightmaxdisY;//右侧拐点Y坐标
	params[18] = RightmaxdisX;//右侧拐点x坐标	

	params[19]= lefttop_Y;//左侧肺野顶点Y坐标
	params[20]= lefttop_X;//左侧肺野顶点x坐标
	params[21]= righttop_Y;//右侧肺野顶点Y坐标
	params[22]= righttop_X;//右侧肺野顶点x坐标

	params[23] = leftbot_Y;//左侧肺野底点Y坐标
	params[24] = leftbot_X;//左侧肺野底点x坐标
	params[25] = rightbot_Y;//右侧肺野底点Y坐标
	params[26] = rightbot_X;//右侧肺野底点x坐标

	// 映射原始图心胸比及各个特殊点的坐标。映射到原始图像
	MaptoOrg_CTR(imgwidth, imgheight, downsampwidth, downsampheight, params);

	return 1;
}

//左侧图像的膈肌点为128，右侧图像的膈肌点为255
int CTRUnet_Detection::Diaphragm_detect(unsigned char* presult_img, int downsampwidth, int downsampheight,int params[],
	unsigned short* pdiaph_Line_imgorg)//downsampwidth, downsampheight,
{
	int imgwidth = params[0];
	int imgheight = params[1];

	unsigned char* pdiaph_Line_img = new unsigned char[512 * 512](); //下采样的输出地址//512*512的图像淹模
	//数据转换
	Matrix<unsigned short> matmasklung(downsampwidth, downsampheight);
	//构建左右肺野区域
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
	signum = LeftRightSegimg(matmasklung, matleftlungmask, matrightlungmask, xvec_left, yvec_left, xvec_right, yvec_right);//只是区别图像的左侧和右侧的肺野
	if (signum != 1)
	{
		return 0;
	}

	//左侧轮廓的顶点位置
	int lefttop_Y = *min_element(yvec_left.begin(), yvec_left.end());
	int lefttop_X = xvec_left[min_element(yvec_left.begin(), yvec_left.end()) - yvec_left.begin()];
	//右侧轮廓的顶点位置
	int righttop_Y = *min_element(yvec_right.begin(), yvec_right.end());
	int righttop_X = xvec_right[min_element(yvec_right.begin(), yvec_right.end()) - yvec_right.begin()];

	vector<int> xvec_leftrib;
	vector<int> yvec_leftrib;
	vector<int> xvec_leftdiaph;
	vector<int> yvec_leftdiaph;//左侧分割
	signum = RibDiaphSeg_left(matleftlungmask, xvec_left, yvec_left, xvec_leftrib, yvec_leftrib, xvec_leftdiaph, yvec_leftdiaph);
	if (signum != 1)
	{
		return 0;
	}

	vector<int> xvec_rightrib;
	vector<int> yvec_rightrib;
	vector<int> xvec_rightdiaph;
	vector<int> yvec_rightdiaph;//右侧分割
	signum = RibDiaphSeg_right(matrightlungmask, xvec_right, yvec_right, xvec_rightrib, yvec_rightrib, xvec_rightdiaph, yvec_rightdiaph);
	if (signum != 1)
	{
		return 0;
	}

	//寻找中心分割线
	int Rib_Leftind = *max_element(xvec_right.begin(), xvec_right.end());//  找到最左侧的肋骨位置
	int Rib_Rightind = *min_element(xvec_left.begin(), xvec_left.end());//  找到最右侧的肋骨位置
	//心胸比中间线的X方向分割线位置
	int Midseg_X = (Rib_Leftind + Rib_Rightind) / 2;   

	//左侧横纵隔膜边界排序
	vector<int> Xdiaph_Leftorder;
	vector<int> Ydiaph_Leftorder;
	Matrix<unsigned short> matLableimg_left(downsampwidth, downsampheight);
	GetorderDiaph(matLableimg_left, xvec_leftdiaph, yvec_leftdiaph, Xdiaph_Leftorder, Ydiaph_Leftorder);//                                  


	//右侧横纵隔膜边界排序
	vector<int> Xdiaph_Rightorder;
	vector<int> Ydiaph_Rightorder;
	Matrix<unsigned short> matLableimg_right(downsampwidth, downsampheight);
	GetorderDiaph(matLableimg_right, xvec_rightdiaph, yvec_rightdiaph, Xdiaph_Rightorder, Ydiaph_Rightorder);//  


	//判断图像的左侧是左肺还是右肺,边界点包含于肺野点中，只需取其中一个点判断即可
	int diaph_Leftind = 0; //左侧横隔肌有效点的位置 。 //计算获取拐点
	int hart_Leftind = 0; //左侧心脏有效点的位置
	int leftmaxdisind;//左侧拐点的最大距离的索引位置

	int diaph_Rightind = 0; //右侧有效点的位置
	int hart_Rightind = 0; //右侧有效点的位置
	int rightmaxdisind;//右侧拐点的最大的索引位置

	int deta_Y = 20;

	//判断左右肺部区域
	int LeftdisX = 0;
	int RightdisX = 0;
	Matrix<unsigned short> matLeftdis(matleftlungmask.width, matleftlungmask.height);
	Matrix<unsigned short> matRightdis(matrightlungmask.width, matrightlungmask.height);
	LeftRight_dist(LeftdisX, RightdisX, Midseg_X, matLeftdis, matRightdis,
		xvec_leftrib, yvec_leftrib, xvec_rightrib, yvec_rightrib,
		Xdiaph_Leftorder, Ydiaph_Leftorder, Xdiaph_Rightorder, Ydiaph_Rightorder);

	if (LeftdisX <= RightdisX)//(matmasklung.pdata[index] == 1)//等于1为右肺，左侧对应右肺
	{
		//先算右肺
		vector<float> nvect_leftratio;//右肺部每个点的斜率值,leftmaxdisind为右肺的斜率
		GetHeartDiaphPoint_rightlung(Xdiaph_Leftorder, Ydiaph_Leftorder, leftmaxdisind, nvect_leftratio, 1);//图像左侧传1，叉乘为正    
		//右肺斜率对应的Y坐标
		int leftmaxdis_Y = Ydiaph_Leftorder[leftmaxdisind];//左肺的拐点寻找就在此点的位置上下进行寻找
		leftmaxdis_Y = leftmaxdis_Y - deta_Y;
		//再算左肺
		vector<float> nvect_rightratio;//左肺侧连通域的每个点的斜率值
		GetHeartDiaphPoint_leftlung(Xdiaph_Rightorder, Ydiaph_Rightorder, rightmaxdisind, nvect_rightratio, leftmaxdis_Y, -1); //图像右侧传-1，叉乘为负

	}
	else //if (matmasklung.pdata[index] == 2)//左侧对应左肺
	{
		//先算右肺
		vector<float> nvect_rightratio;//右肺部每个点的斜率值,leftmaxdisind为右肺的斜率
		GetHeartDiaphPoint_rightlung(Xdiaph_Rightorder, Ydiaph_Rightorder, rightmaxdisind, nvect_rightratio, -1);//图像右侧传-1  
		//右肺斜率对应的Y坐标
		int rightmaxdis_Y = Ydiaph_Rightorder[rightmaxdisind];//左肺的拐点寻找就在此点的位置上下进行寻找
		rightmaxdis_Y = rightmaxdis_Y - deta_Y;

		//再算左肺
		vector<float> nvect_leftratio;
		GetHeartDiaphPoint_leftlung(Xdiaph_Leftorder, Ydiaph_Leftorder, leftmaxdisind, nvect_leftratio, rightmaxdis_Y, 1); //图像左侧传1，
	}

	//---------------------------对Ydiaph_Leftorder和Ydiaph_Rightorder所代表的的曲线和拐点做处理,分别获取左侧和右侧的膈肌点----------------
	//左侧检测
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
	//连通域判断选择获取膈肌点
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

		//判断数据大的值为所需要的值
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


	//右侧检测
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

	//右侧连通域判断选择，获取膈肌点
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

		//判断数据大的值为所需要的值
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

	//映射到原始图
	MaptoOrg_Diaph(imgwidth, imgheight, downsampwidth, downsampheight, pdiaph_Line_img, pdiaph_Line_imgorg);//映射到原始图的横膈膜边界

	delete[] pdiaph_Line_img;
	pdiaph_Line_img = nullptr;

	return 1;
}

//左右肺的区分和面积计算，pleftrightlung_imgmask输出掩模中右肺1，左肺2
int CTRUnet_Detection::Lung_Areacalculate(unsigned char* presult_img, int downsampwidth, int downsampheight, 
	unsigned short* pleftrightlung_imgmaskorg, int params[])
{
	int imgwidth = params[0];
	int imgheight = params[1];

	unsigned char* pleftrightlung_imgmask = new unsigned char[512 * 512](); //下采样左右图的区别掩模512-*512

	//数据转换
	Matrix<unsigned short> matmasklung(downsampwidth, downsampheight);
	//构建左右肺野区域
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
	//matmasklung数组可以作为使用面积大小来区域左右肺的输出，面积大的为右肺置为1，面积小的为左肺置为2
	signum = LeftRightSegimg(matmasklung, matleftlungmask, matrightlungmask, xvec_left, yvec_left, xvec_right, yvec_right);//只是区别图像的左侧和右侧的肺野
	if (signum != 1)
	{
		return 0;
	}

	vector<int> xvec_leftrib;
	vector<int> yvec_leftrib;
	vector<int> xvec_leftdiaph;
	vector<int> yvec_leftdiaph;//左侧分割肋缘和横纵隔膜。
	signum = RibDiaphSeg_left(matleftlungmask, xvec_left, yvec_left, xvec_leftrib, yvec_leftrib, xvec_leftdiaph, yvec_leftdiaph);
	if (signum != 1)
	{
		return 0;
	}

	vector<int> xvec_rightrib;
	vector<int> yvec_rightrib;
	vector<int> xvec_rightdiaph;
	vector<int> yvec_rightdiaph;//右侧分割肋缘和横纵隔膜。
	signum = RibDiaphSeg_right(matrightlungmask, xvec_right, yvec_right, xvec_rightrib, yvec_rightrib, xvec_rightdiaph, yvec_rightdiaph);
	if (signum != 1)
	{
		return 0;
	}

	//寻找中心分割线
	int Rib_Leftind = *max_element(xvec_right.begin(), xvec_right.end());//  找到最左侧的肋骨位置
	int Rib_Rightind = *min_element(xvec_left.begin(), xvec_left.end());//  找到最右侧的肋骨位置
	//心胸比中间线的X方向分割线位置
	int Midseg_X = (Rib_Leftind + Rib_Rightind) / 2;

	//左侧横纵隔膜边界排序
	vector<int> Xdiaph_Leftorder;
	vector<int> Ydiaph_Leftorder;
	Matrix<unsigned short> matLableimg_left(downsampwidth, downsampheight);
	GetorderDiaph(matLableimg_left, xvec_leftdiaph, yvec_leftdiaph, Xdiaph_Leftorder, Ydiaph_Leftorder);//                                  

	//右侧横纵隔膜边界排序
	vector<int> Xdiaph_Rightorder;
	vector<int> Ydiaph_Rightorder;
	Matrix<unsigned short> matLableimg_right(downsampwidth, downsampheight);
	GetorderDiaph(matLableimg_right, xvec_rightdiaph, yvec_rightdiaph, Xdiaph_Rightorder, Ydiaph_Rightorder);//  

	//判断图像的左侧是左肺还是右肺,边界点包含于肺野点中，只需取其中一个点判断即可
	int diaph_Leftind = 0; //左侧横隔肌有效点的位置 。 //计算获取拐点
	int hart_Leftind = 0; //左侧心脏有效点的位置
	int leftmaxdisind; //左侧拐点的最大距离的索引位置

	int diaph_Rightind = 0; //右侧有效点的位置
	int hart_Rightind = 0; //右侧有效点的位置
	int rightmaxdisind; //右侧拐点的最大的索引位置

	int deta_Y = 20; 

	//判断左右肺部区域
	int LeftdisX = 0;
	int RightdisX = 0;
	Matrix<unsigned short> matLeftdis(matleftlungmask.width, matleftlungmask.height);
	Matrix<unsigned short> matRightdis(matrightlungmask.width, matrightlungmask.height);
	LeftRight_dist(LeftdisX, RightdisX, Midseg_X, matLeftdis, matRightdis,
		xvec_leftrib, yvec_leftrib, xvec_rightrib, yvec_rightrib,
		Xdiaph_Leftorder, Ydiaph_Leftorder, Xdiaph_Rightorder, Ydiaph_Rightorder);

	int rightlungarea = 0;
	int leftlungarea = 0;

	//复制留用
   //构建左右肺野区域
	Matrix<unsigned short> matleftlungmask_copy(downsampwidth, downsampheight);
	Matrix<unsigned short> matrightlungmask_copy(downsampwidth, downsampheight);
	signum = LeftRightsideimg(matmasklung, matleftlungmask_copy, matrightlungmask_copy);
	if (signum != 1)
	{
		return 0;
	}

	if (LeftdisX <= RightdisX)////左侧对应右肺，右肺1，左肺2
	{
		//先赋值右肺1
		for (int i = 0; i < matleftlungmask_copy.Matrix_length(); i++)
		{
			if (matleftlungmask_copy.pdata[i]!=0)
			{
				pleftrightlung_imgmask[i] = 1;
				rightlungarea++;
			}
		}
		//再赋值左肺2
		for (int i = 0; i < matrightlungmask_copy.Matrix_length(); i++)
		{
			if (matrightlungmask_copy.pdata[i] != 0)
			{
				pleftrightlung_imgmask[i] = 2;
				leftlungarea++;
			}
		}
		params[27] = rightlungarea;//左侧对应右肺面积
		params[28] = leftlungarea;//右侧对应左肺面积
		params[29] = 1;//图像的左侧对应右肺，右侧对应左肺
	}
	else //左侧对应左肺，右肺1，左肺2
	{

		//先赋值左肺2
		for (int i = 0; i < matleftlungmask_copy.Matrix_length(); i++)
		{
			if (matleftlungmask_copy.pdata[i] != 0)
			{
				pleftrightlung_imgmask[i] = 2;
				leftlungarea++;
			}
		}
		//再赋值右肺1
		for (int i = 0; i < matrightlungmask_copy.Matrix_length(); i++)
		{
			if (matrightlungmask_copy.pdata[i] != 0)
			{
				pleftrightlung_imgmask[i] = 1;
				rightlungarea++;
			}
		}
		params[27] = leftlungarea;//左侧对应左肺面积
		params[28] = rightlungarea;//右侧对应右肺面积
		params[29] = 2;//图像的左侧对应左肺，右侧对应右肺
	}	

	//左右肺野识别映射到原始图中
	MaptoOrg_Areacal(imgwidth, imgheight, downsampwidth, downsampheight, params, pleftrightlung_imgmask, pleftrightlung_imgmaskorg);//左右肺识别和面积计算映射 
	
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

	//根据面积对肺野的左右进行判定。//
	vector<int> labval;
	vector<int> labind;
	bool TF = 0;
	Matrix<unsigned short> matLabelimg(width, height);
	Conectchose(matmasklungcopy, matLabelimg, labval, labind, TF);//四联通
	if (labind.size() <= 1)
	{
		return 0;
	}
	//对左右肺野的坐标进行统计
	for (int i = 0, step1 = 0; i < matmasklungcopy.height; i++, step1 += matmasklungcopy.width)
	{
		for (int j = 0; j < matmasklungcopy.width; j++)
		{
			if (matLabelimg.pdata[step1 + j] == labind.back())//最大的
			{
				matmasklung.pdata[step1 + j] = 1;//面积大的右肺大，设置为1
			}
			else if (matLabelimg.pdata[step1 + j] == labind[labind.size() - 2])//倒数第二大的
			{
				matmasklung.pdata[step1 + j] = 2;//面积小的左肺，设置为2
			}
			else
			{
				matmasklung.pdata[step1 + j] = 0;//其他区域置为0
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

	//先膨胀
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

	//连通域选择
	labval.clear();
	labind.clear();
	TF = 0;	
	memset(matLabelimg.pdata, 0, sizeof(unsigned short) * matLabelimg.Matrix_length());
	Conectchose(dilateimg, matLabelimg, labval, labind, TF);//四联通
	if (labind.size() <= 1)
	{
		return 0;
	}

	vector<int> xvec_labfirst;
	vector<int> yvec_labfirst;
	vector<int> xvec_labsec;
	vector<int> yvec_labsec;

	//对左右肺野的坐标进行统计
	for (int i = 0, step1 = 0; i < matLabelimg.height; i++, step1 += matLabelimg.width)
	{
		for (int j = 0; j < matLabelimg.width; j++)
		{
			if (matLabelimg.pdata[step1 + j] == labind.back())//最大的
			{
				xvec_labfirst.push_back(j);
				yvec_labfirst.push_back(i);
			}
			else if (matLabelimg.pdata[step1 + j] == labind[labind.size() - 2])//倒数第二大的
			{
				xvec_labsec.push_back(j);
				yvec_labsec.push_back(i);
			}
		}
	}

	//根据坐标进行判断中心分割线和左右侧肺野
	int sumfirst = accumulate(xvec_labfirst.begin(), xvec_labfirst.end(), 0);
	sumfirst = sumfirst / xvec_labfirst.size();
	int sumsec = accumulate(xvec_labsec.begin(), xvec_labsec.end(), 0);
	sumsec = sumsec / xvec_labsec.size();

	int xmidseg = 0;
	if (sumfirst < sumsec)
	{
		//xvec_labfirst为左侧
		//左右坐标填充
		for (size_t i = 0; i < xvec_labfirst.size(); i++)//左侧坐标填充
		{
			matLeftChest.pdata[yvec_labfirst[i] * width + xvec_labfirst[i]] = 1;
		}
		for (size_t i = 0; i < xvec_labsec.size(); i++)//右侧坐标填充
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
		//xvec_labsec为左侧,左侧找最大，右侧找最小
		//左右坐标填充
		for (size_t i = 0; i < xvec_labsec.size(); i++)//左侧坐标填充
		{
			matLeftChest.pdata[yvec_labsec[i] * width + xvec_labsec[i]] = 1;
		}
		for (size_t i = 0; i < xvec_labfirst.size(); i++)//右侧坐标填充
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
		//八邻域
		int width = matInputimg.width;
		int height = matInputimg.height;

		vector<int> nxvect;//连通域横坐标收纳盒
		vector<int> nyvect;//连通域纵坐标收纳盒
		int clasenumber = 0;//连通域表盒
		//定义八邻域数组，从左上角开始，顺时针遍历八邻域
		int neibx[8] = { -1, 0, 1, 1, 1, 0, -1, -1 };//八邻域x方向
		int neiby[8] = { -1, -1, -1, 0, 1, 1, 1, 0 };//八邻域y方向

		for (int i = 0, step1 = 0; i < height; i++, step1 += width)
		{
			unsigned short* inputimgrow = matInputimg.pdata + step1;//输入图像行首值
			unsigned short* labelimgrow = matLabelimg.pdata + step1;//输出图像行首值
			for (int j = 0; j < width; j++)
			{

				unsigned short* inputimgpos = inputimgrow + j;//输入图像此像素坐标
				unsigned short* labelimgpos = labelimgrow + j;//输出图像此点像素坐标
				int deter = *inputimgpos;
				//对当前值进行判定
				if (deter == 1)//只要像素值为1，即纳入考虑范畴
				{
					clasenumber += 1;//连通域标号加一
					labnum.resize(clasenumber + 1);
					*inputimgpos = 0;//旧图像此点置为0
					*labelimgpos = clasenumber;//新图像对此点进行标记记录                

					//获取此次中心点的横纵坐标
					//邻域判定
					int runvalue = 0;
					int countnumber = 0; // 邻域计数
					nxvect.push_back(j);
					nyvect.push_back(i); // 将坐标存储到vector向量数组中             

					while (runvalue <= countnumber)
					{
						for (int k = 0; k < 8; k++)
						{//坐标
							int nx = nxvect[runvalue] + neibx[k];
							int ny = nyvect[runvalue] + neiby[k];
							if (nx < width && nx >= 0 && ny < height && ny >= 0)//邻域坐标在图像尺寸上下限内
							{
								unsigned short* nebinputimgpos = matInputimg.pdata + ny * width + nx;//输入图像邻域位置
								unsigned short* neblabelimgpos = matLabelimg.pdata + +ny * width + nx;//输出图像此邻域点的位置
								if (*nebinputimgpos == 1)
								{
									countnumber = countnumber + 1;
									nxvect.push_back(nx);//存储此邻域点的横坐标
									nyvect.push_back(ny);//存储此领域点的纵坐标
									*nebinputimgpos = 0;//原始输入图像此邻域点的值赋为0
									*neblabelimgpos = clasenumber;//对新矩阵的此邻域点进行标号标记
								}
							}
						}

						runvalue = runvalue + 1;//追赶连通域的总像素数目，直到追上countnumber
					}

					labnum[clasenumber] = countnumber + 1; // 加上起始点
				}

				nxvect.clear();
				nyvect.clear();
			}
		}
	}
	else
	{
		//四邻域
		int width = matInputimg.width;
		int height = matInputimg.height;

		vector<int> nxvect;//连通域横坐标收纳盒
		vector<int> nyvect;//连通域纵坐标收纳盒
		int clasenumber = 0;//连通域个数盒

		//定义四邻域数组，从左上角开始，顺时针遍历四邻域
		int neibx[4] = { -1,0,1,0 };//四邻域x方向
		int neiby[4] = { 0,-1,0,1 };//四邻域y方向

		for (int i = 0; i < height; i++)
		{
			unsigned short* inputimgrow = matInputimg.pdata + i * width;//输入图像行首值
			unsigned short* labelimgrow = matLabelimg.pdata + i * width;//输出图像行首值
			for (int j = 0; j < width; j++)
			{
				unsigned short* inputimgpos = inputimgrow + j;//输入图像此像素坐标
				unsigned short* labelimgpos = labelimgrow + j;//输出图像此点像素坐标
				int deter = *inputimgpos;

				//对当前值进行判定
				if (deter == 1)//只要像素值为1，即纳入考虑范畴
				{
					clasenumber = clasenumber + 1;//连通域标号加一
					labnum.resize(clasenumber + 1);//空间比标号的数目多1
					*inputimgpos = 0;//旧图像此点置为0
					*labelimgpos = clasenumber;//新图像对此点进行标记记录    

					int runvalue = 0;
					int countnumber = 0; // 邻域计数
					nxvect.push_back(j);
					nyvect.push_back(i); // 将坐标存储到vector向量数组中             

					while (runvalue <= countnumber)
					{
						for (int k = 0; k < 4; k++)
						{
							int nx = nxvect[runvalue] + neibx[k];
							int ny = nyvect[runvalue] + neiby[k];
							if (nx < width && nx >= 0 && ny < height && ny >= 0)//邻域坐标在图像尺寸上下限内
							{
								unsigned short* nebinputimgpos = matInputimg.pdata + ny * width + nx;//输入图像邻域位置
								unsigned short* neblabelimgpos = matLabelimg.pdata + +ny * width + nx;//输出图像此邻域点的位置
								if (*nebinputimgpos == 1)
								{
									countnumber = countnumber + 1;
									nxvect.push_back(nx);//存储此邻域点的横坐标
									nyvect.push_back(ny);//存储此领域点的纵坐标
									*nebinputimgpos = 0;//原始输入图像此邻域点的值赋为0
									*neblabelimgpos = clasenumber;//对新矩阵的此邻域点进行标号标记
								}
							}
						}
						runvalue = runvalue + 1;//追赶连通域的总像素数目，直到追上countnumber
					}
					labnum[clasenumber] = countnumber + 1; //加上起始点 
				}
				nxvect.clear();
				nyvect.clear();
			}
		}
	}

	//利用multimap对labsum的进行值和索引的排序，排序的规则以值的大小（从小到大）为顺序同时排列对应的索引值
	//相关排序后的纳入vector数组中
	multimap<int, int> labnummap;
	if (labnum.size() > 1)
	{
		for (int i = 1; i < labnum.size(); i++)
		{
			int keyind = i;
			int keyvalue = labnum[i];
			labnummap.insert(std::pair<int, int>(keyvalue, keyind));
		}

		// 数值的大小排序
	   // 根据统计的数值进行索引号的排序 （从小到大）
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

	//左侧肺部处理
	int yminleft = *min_element(yvec_left.begin(), yvec_left.end());//左侧顶点Y坐标
	int yminleft_ind = min_element(yvec_left.begin(), yvec_left.end()) - yvec_left.begin();
	int xminleft = xvec_left[yminleft_ind];//左侧顶点X坐标

	//获取左侧肋缘最下侧的点
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

	//对左肺上区域点的连通域至为0
	//定义八邻域数组，从左上角开始，顺时针遍历八邻域
	int neibx[8] = { -1, 0, 1, 1, 1, 0, -1, -1 };//八邻域x方向
	int neiby[8] = { -1, -1, -1, 0, 1, 1, 1, 0 };//八邻域y方向
	vector<int> nxvect;//连通域横坐标收纳盒
	vector<int> nyvect;//连通域纵坐标收纳盒

	int indy = yminleft;
	int jndx = xminleft;
	if ((indy == yminleft) && (jndx == xminleft))//上方点16连通域像素置为0
	{
		nxvect.push_back(jndx);
		nyvect.push_back(indy); // 将坐标存储到vector向量数组中  
		int runvalue = 0;
		while (runvalue < 16)
		{
			int sig = 1;
			for (int k = 0; k < 8; k++)
			{//坐标
				int nx = nxvect[runvalue] + neibx[k];
				int ny = nyvect[runvalue] + neiby[k];
				if (nx < matLeftChest.width && nx >= 0 && ny < matLeftChest.height && ny >= 0)//邻域坐标在图像尺寸上下限内
				{
					unsigned short* nebinputimgpos = matLeftChest.pdata + ny * matLeftChest.width + nx;//输入图像邻域位置
					if (*nebinputimgpos != 0)
					{
						nxvect.push_back(nx);//存储此邻域点的横坐标
						nyvect.push_back(ny);//存储此领域点的纵坐标
						*nebinputimgpos = 0;//原始输入图像此邻域点的值赋为0						
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

	//左肺顶部区域10个高度区域连通域置为0
	for (int i = yminleft; i < yminleft + 10; i++)
	{
		for (int j = 0; j < matLeftChest.width; j++)
		{
			matLeftChest.pdata[i * matLeftChest.width + j] = 0;
		}
	}

	//对左肺下区域点的连通域至为0
	nxvect.clear();//连通域横坐标收纳盒
	nyvect.clear();//连通域纵坐标收纳盒
	indy = ymaxleft;
	jndx = xmaxleft;
	if ((indy == ymaxleft) && (jndx == xmaxleft))//上方点16连通域像素置为0,matLeftChest.pdata[yminleft * matLeftChest.width + xminleft] == 1
	{
		nxvect.push_back(jndx);
		nyvect.push_back(indy); //将坐标存储到vector向量数组中  
		int runvalue = 0;
		while (nxvect.size() < 16)
		{
			int sig = 1;
			for (int k = 0; k < 8; k++)
			{//坐标
				int nx = nxvect[runvalue] + neibx[k];
				int ny = nyvect[runvalue] + neiby[k];
				if (nx < matLeftChest.width && nx >= 0 && ny < matLeftChest.height && ny >= 0)//邻域坐标在图像尺寸上下限内
				{
					unsigned short* nebinputimgpos = matLeftChest.pdata + ny * matLeftChest.width + nx;//输入图像邻域位置
					if (*nebinputimgpos != 0)
					{
						nxvect.push_back(nx);//存储此邻域点的横坐标
						nyvect.push_back(ny);//存储此领域点的纵坐标
						*nebinputimgpos = 0;//原始输入图像此邻域点的值赋为0
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

	//连通域选择
	vector<int> labval;
	vector<int> labind;
	bool TF = 0;
	Matrix<unsigned short> matLabelimg(matLeftChest.width, matLeftChest.height);
	Conectchose(matLeftChest, matLabelimg, labval, labind, TF);//四联通
	if (labind.size() <= 1)
	{
		return 0;
	}

	//通过X坐标累加平均，判断边界是肋缘or横纵隔边界
	vector<int> summean1;
	vector<int> summean2;
	for (int i = 0, step1 = 0; i < matLeftChest.height; i++, step1 += matLeftChest.width)
	{
		for (int j = 0; j < matLeftChest.width; j++)
		{
			if (matLabelimg.pdata[step1 + j] == labind.back())//最大连通域
			{
				summean1.push_back(j);

			}
			else if (matLabelimg.pdata[step1 + j] == labind[labind.size() - 2])//倒数第二大连通域
			{
				summean2.push_back(j);
			}
		}
	}
	int summean1value = accumulate(summean1.begin(), summean1.end(), 0) / summean1.size();
	int summean2value = accumulate(summean2.begin(), summean2.end(), 0) / summean2.size();

	//根据横坐标的平均值进行判定
	if (summean2value > summean1value)//1为肋缘
	{

		for (int i = 0, step1 = 0; i < matLeftChest.height; i++, step1 += matLeftChest.width)
		{
			for (int j = 0; j < matLeftChest.width; j++)
			{
				if (matLabelimg.pdata[step1 + j] == labind.back())//平均值小的是肋缘
				{
					xvec_leftrib.push_back(j);
					yvec_leftrib.push_back(i);
				}
				else if (matLabelimg.pdata[step1 + j] == labind[labind.size() - 2])//平均值大的是横纵隔膜边界，
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
				if (matLabelimg.pdata[step1 + j] == labind.back())//平均值大的是横纵隔膜边界
				{
					xvec_leftdiaph.push_back(j);
					yvec_leftdiaph.push_back(i);
				}
				else if (matLabelimg.pdata[step1 + j] == labind[labind.size() - 2])//平均值小的是肋缘
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
	//右侧肺部处理
	int yminright = *min_element(yvec_right.begin(), yvec_right.end());//获取右侧最上方点的Y坐标
	int yminright_ind = min_element(yvec_right.begin(), yvec_right.end()) - yvec_right.begin();
	int xminright = xvec_right[yminright_ind];//获取右侧最上方点的X坐标

	//获取右侧图像肋缘最下方的点的坐标
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

	//对右肺上区域点的连通域至为0
	//定义八邻域数组，从右上角开始，顺时针遍历八邻域
	int neibx[8] = { -1, 0, 1, 1, 1, 0, -1, -1 };//八邻域x方向
	int neiby[8] = { -1, -1, -1, 0, 1, 1, 1, 0 };//八邻域y方向
	vector<int> nxvect;//连通域横坐标收纳盒
	vector<int> nyvect;//连通域纵坐标收纳盒
	int indy = yminright;
	int jndx = xminright;
	if ((indy == yminright) && (jndx == xminright))//上方点16连通域像素置为0
	{
		nxvect.push_back(jndx);
		nyvect.push_back(indy); // 将坐标存储到vector向量数组中  
		int runvalue = 0;
		while (nxvect.size() < 16)
		{
			for (int k = 0; k < 8; k++)
			{//坐标
				int nx = nxvect[runvalue] + neibx[k];
				int ny = nyvect[runvalue] + neiby[k];
				if (nx < matRightChest.width && nx >= 0 && ny < matRightChest.height && ny >= 0)//邻域坐标在图像尺寸上下限内
				{
					unsigned short* nebinputimgpos = matRightChest.pdata + ny * matRightChest.width + nx;//输入图像邻域位置
					if (*nebinputimgpos != 0)
					{
						nxvect.push_back(nx);//存储此邻域点的横坐标
						nyvect.push_back(ny);//存储此领域点的纵坐标
						*nebinputimgpos = 0;//原始输入图像此邻域点的值赋为0
					}
				}
			}
			runvalue++;
		}
	}

	//右肺顶部区域10个高度区域连通域置为0
	for (int i = yminright; i < yminright + 10; i++)
	{
		for (int j = 0; j < matRightChest.width; j++)
		{
			matRightChest.pdata[i * matRightChest.width + j] = 0;
		}
	}

	//对右肺下区域点的连通域至为0
	nxvect.clear();//连通域横坐标收纳盒
	nyvect.clear();//连通域纵坐标收纳盒
	indy = ymaxright;
	jndx = xmaxright;
	if ((indy == ymaxright) && (jndx == xmaxright))//上方点16连通域像素置为0,matRightChest.pdata[yminright * matRightChest.width + xminright] == 1
	{
		nxvect.push_back(jndx);
		nyvect.push_back(indy); //将坐标存储到vector向量数组中  
		int runvalue = 0;
		while (nxvect.size() < 16)
		{
			for (int k = 0; k < 8; k++)
			{//坐标
				int nx = nxvect[runvalue] + neibx[k];
				int ny = nyvect[runvalue] + neiby[k];
				if (nx < matRightChest.width && nx >= 0 && ny < matRightChest.height && ny >= 0)//邻域坐标在图像尺寸上下限内
				{
					unsigned short* nebinputimgpos = matRightChest.pdata + ny * matRightChest.width + nx;//输入图像邻域位置
					if (*nebinputimgpos != 0)
					{
						nxvect.push_back(nx);//存储此邻域点的横坐标
						nyvect.push_back(ny);//存储此领域点的纵坐标
						*nebinputimgpos = 0;//原始输入图像此邻域点的值赋为0
					}
				}
			}
			runvalue++;
		}
	}

	//连通域选择
	vector<int> labval;
	vector<int> labind;
	bool TF = 0;
	Matrix<unsigned short> matLabelimg(matRightChest.width, matRightChest.height);
	Conectchose(matRightChest, matLabelimg, labval, labind, TF);//四联通
	if (labind.size() <= 1)
	{
		return 0;
	}

	//通过X坐标累加平均，判断边界是肋缘or横纵隔边界
	vector<int> summean1;
	vector<int> summean2;
	for (int i = 0, step1 = 0; i < matRightChest.height; i++, step1 += matRightChest.width)
	{
		for (int j = 0; j < matRightChest.width; j++)
		{
			if (matLabelimg.pdata[step1 + j] == labind.back())//最大连通域
			{
				summean1.push_back(j);

			}
			else if (matLabelimg.pdata[step1 + j] == labind[labind.size() - 2])//倒数第二大连通域
			{
				summean2.push_back(j);
			}
		}
	}
	int summean1value = accumulate(summean1.begin(), summean1.end(), 0) / summean1.size();
	int summean2value = accumulate(summean2.begin(), summean2.end(), 0) / summean2.size();


	//根据横坐标的平均值进行判定
	if (summean2value > summean1value)//1为横纵隔膜边界
	{

		for (int i = 0, step1 = 0; i < matRightChest.height; i++, step1 += matRightChest.width)
		{
			for (int j = 0; j < matRightChest.width; j++)
			{
				if (matLabelimg.pdata[step1 + j] == labind.back())//平均值小的是横纵隔膜边界
				{
					xvec_rightdiaph.push_back(j);
					yvec_rightdiaph.push_back(i);

				}
				else if (matLabelimg.pdata[step1 + j] == labind[labind.size() - 2])//平均值大的是肋缘
				{
					xvec_rightrib.push_back(j);
					yvec_rightrib.push_back(i);
				}
			}
		}

	}
	else//2为横纵隔膜边界，1为肋缘
	{
		for (int i = 0, step1 = 0; i < matRightChest.height; i++, step1 += matRightChest.width)
		{
			for (int j = 0; j < matRightChest.width; j++)
			{
				if (matLabelimg.pdata[step1 + j] == labind.back())//平均值大的是肋缘
				{
					xvec_rightrib.push_back(j);
					yvec_rightrib.push_back(i);

				}
				else if (matLabelimg.pdata[step1 + j] == labind[labind.size() - 2])//平均值小的是横纵隔膜边界
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

	//模板数组行列遍历，首先取行首地址
	int templatedis = templateerodewidth * templateerodeheight;

	int addwidth = templateerodewidth / 2;
	int addheight = templateerodeheight / 2;
	int widthheight = width * height;

	int tempwid = width + addwidth * 2;
	int tempheight = height + addheight * 2;
	int tempheigthwid = tempwid * tempheight;
	//创建扩大后的图像空间，将输入图像填充进扩大后的图像空间
	Matrix<unsigned short> matTempcalulate(tempwid, tempheight);

	//填充新空间
	//中间部分填充
	for (int i = addheight, step1 = addheight * tempwid, step2 = 0; i < addheight + height; i++, step1 += tempwid, step2 += width)
	{
		for (int j = addwidth, step3 = 0; j < addwidth + width; j++, step3 += 1)
		{
			matTempcalulate.pdata[step1 + j] = matInputimg.pdata[step2 + step3];
		}
	}

	//左右侧边界填充
	for (int i = addheight, step1 = addheight * tempwid, step2 = 0; i < addheight + height; ++i, step1 += tempwid, step2 += width) //遍历新空间
	{
		//左侧边界填充
		for (int j = 0; j < addwidth; ++j)
		{
			matTempcalulate.pdata[step1 + j] = matInputimg.pdata[step2];//左侧边界填充的为与距离最近的数相同。
		}
		//右侧边界填充
		for (int j = width + addwidth; j < tempwid; ++j)
		{
			matTempcalulate.pdata[step1 + j] = matInputimg.pdata[step2 + width - 1];//右侧边界填充的为与距离最近的数相同。
		}
	}

	//上侧边界填充
	for (int i = 0, step1 = 0; i < addheight; i++, step1 += tempwid)//上侧行数遍历
	{
		for (int j = 0; j < tempwid; j++)
		{
			matTempcalulate.pdata[step1 + j] = matTempcalulate.pdata[addheight * tempwid + j];
		}
	}
	//下侧行数遍历
	for (int i = height + addheight; i < tempheight; i++)
	{
		unsigned short* tempcalulaterow = matTempcalulate.pdata + i * tempwid;  //输出图像第i行的首地址         
		unsigned short* botrow = matTempcalulate.pdata + (addheight + height - 1) * tempwid;  //输出图像第i行的首地址    
		for (int j = 0; j < tempwid; j++)
		{
			tempcalulaterow[j] = botrow[j];
		}
	}

	//注意：如果模板中出现的非前景值，默认为“不关心元素”，见冈萨459页
	//模板中的前景计数
	int countnum = 0;
	for (int i = 0; i < templateerodewidth * templateerodeheight; i++)
	{
		if (templatedilate.pdata[i] == 1)
		{
			countnum = countnum + 1;//计数加一
		}
	}

	//遍历每个中间填充像素点。
	for (int j = addheight; j < addheight + height; j++)
	{
		unsigned short* inputimgrow = matTempcalulate.pdata + j * tempwid;//输入图像中心点的行首地址
		unsigned short* outputimgrow = matOutputimg.pdata + (j - addheight) * width;//输出图像中心点的行首地址
		for (int i = addwidth; i < addwidth + width; i++)//每一行的遍历
		{
			unsigned short* inputimgpos = inputimgrow + i;//输入图像中心点的地址
			unsigned short* outputimgpos = outputimgrow + i - addwidth;//输出图像中心点的地址

			int numcount = 0;//计数变量。
			//进行模板遍历
			for (int m = -addheight; m <= addheight; m++)//模板遍历和当前数组匹配遍历
			{
				unsigned short* templateeroderow = templatedilate.pdata + (m + addheight) * templateerodewidth;//模板的行首值              
				unsigned short* temparryrow = inputimgpos + m * tempwid;//输入数组行首值
				for (int n = -addwidth; n <= addwidth; n++)
				{
					unsigned short* templateerodepos = templateeroderow + n + addwidth;//模板的当前数值
					unsigned short* temparrypos = temparryrow + n;//输入图像当前值
					if (*templateerodepos == 1 && *temparrypos == 1) //模板前景和数组前景的此像素一一对应
					{
						numcount = numcount + 1;
					}
				}
			}//结束模板遍历
			if (numcount == countnum)//如果模板遍历中的前景个数与计数值相等,意味着模板中和遍历的区块中都一一对应
			{
				*outputimgpos = 1;//输出值赋值为1
			}
			else//
			{
				*outputimgpos = 0;//输出值赋值为0
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

	//模板数组行列遍历，首先取行首地址
	int templatedis = templatedilatewidth * templatedilateheight;

	int addwidth = templatedilatewidth / 2;
	int addheight = templatedilateheight / 2;
	int widthheight = width * height;

	int tempwid = width + addwidth * 2;
	int tempheight = height + addheight * 2;
	int tempheigthwid = tempwid * tempheight;
	//创建扩大后的图像空间，将输入图像填充进扩大后的图像空间
	Matrix<unsigned short> matTempcalulate(tempwid, tempheight);

	//填充新空间
	//中间部分填充
	for (int i = addheight, step1 = addheight * tempwid, step2 = 0; i < addheight + height; i++, step1 += tempwid, step2 += width)
	{
		for (int j = addwidth, step3 = 0; j < addwidth + width; j++, step3 += 1)
		{
			matTempcalulate.pdata[step1 + j] = matInputimg.pdata[step2 + step3];
		}
	}

	//左右侧边界填充
	for (int i = addheight, step1 = addheight * tempwid, step2 = 0; i < addheight + height; ++i, step1 += tempwid, step2 += width) //遍历新空间
	{
		//左侧边界填充
		for (int j = 0; j < addwidth; ++j)
		{
			matTempcalulate.pdata[step1 + j] = matInputimg.pdata[step2];//左侧边界填充的为与距离最近的数相同。
		}
		//右侧边界填充
		for (int j = width + addwidth; j < tempwid; ++j)
		{
			matTempcalulate.pdata[step1 + j] = matInputimg.pdata[step2 + width - 1];//右侧边界填充的为与距离最近的数相同。
		}
	}

	//上侧边界填充
	for (int i = 0, step1 = 0; i < addheight; i++, step1 += tempwid)//上侧行数遍历
	{
		for (int j = 0; j < tempwid; j++)
		{
			matTempcalulate.pdata[step1 + j] = matTempcalulate.pdata[addheight * tempwid + j];
		}
	}
	//下侧行数遍历
	for (int i = height + addheight; i < tempheight; i++)
	{
		unsigned short* tempcalulaterow = matTempcalulate.pdata + i * tempwid;  //输出图像第i行的首地址         
		unsigned short* botrow = matTempcalulate.pdata + (addheight + height - 1) * tempwid;  //输出图像第i行的首地址    
		for (int j = 0; j < tempwid; j++)
		{
			tempcalulaterow[j] = botrow[j];
		}
	}

	//模板遍历
	for (int i = addheight; i < addheight + height; ++i) //遍历模板滤波         
	{
		unsigned short* inputrow = matTempcalulate.pdata + i * tempwid; //输入图像第i行的首地址               
		unsigned short* outputimgrow = matOutputimg.pdata + (i - addheight) * width;  //输出图像第i行的首地址                  
		for (int j = addwidth; j < addwidth + width; ++j)
		{
			unsigned short* inputpos = inputrow + j; //输入图像第i行第j列坐标中心元素的地址
			unsigned short* outputimgpos = outputimgrow + j - addwidth; //输出图像第i行第j列中心坐标元素的地址
			//在模板大小的局部区域内，循环遍历累加。
			for (int m = -addheight; m <= addheight; m++)
			{
				unsigned short* starinputrow = inputpos + m * tempwid;//输入图像
				unsigned short* templatedilaterow = mattemplatedilate.pdata + (m + addheight) * templatedilatewidth;//模板行首地址
				for (int n = -addwidth; n <= addwidth; n++)
				{
					unsigned short* templatedilatepos = templatedilaterow + n + addwidth;//模板当前值
					unsigned short* starinputpos = starinputrow + n;//原始数组当前模板中的对应值
					//判断只要模板中有一处重叠处为前景值，则此中心点置为1
					if (*templatedilatepos == 1 && *starinputpos == 1)
					{
						*outputimgpos = 1;//模板中心对应的输出图像的此坐标置为1
						break;//跳出当前模板第二层循环
					}
				}
				if (*outputimgpos == 1)//跳出当前模板第一层循环
				{
					break;//进入模板外侧的下一个中心元素的遍历计算和判断。
				}
			}
		}
	}
}

//matLableimg的上下断开
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

	////根据线段的起点，进行搜索
	Search_Conectpoint(matLableimg, startX, startY, Xdiaph_Leftorder, Ydiaph_Leftorder);

	//将前1/3的数据去除
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

	int clasenumber = 0;//连通域表盒
	//定义八邻域数组，从左上角开始，顺时针遍历八邻域
	int neibx[8] = { -1, 0, 1, 1, 1, 0, -1, -1 };//八邻域x方向
	int neiby[8] = { -1, -1, -1, 0, 1, 1, 1, 0 };//八邻域y方向

	int i = startY;
	int j = startX;

	unsigned short* inputimgpos = matConectlabel.pdata + i * width + j;
	int deter = *inputimgpos;

	if (deter != 0)
	{
		clasenumber = clasenumber + 1;//连通域标号加一
		*inputimgpos = 0;//旧图像此点置为0        
		// 
	   //获取此次中心点的横纵坐标
	   //邻域判定
		int runvalue = 0;
		int countnumber = 0; // 邻域计数
		orderX.push_back(j);
		orderY.push_back(i); // 将坐标存储到vector向量数组中             

		while (runvalue <= countnumber)
		{
			for (int k = 0; k < 8; k++)
			{//坐标
				int nx = orderX[runvalue] + neibx[k];
				int ny = orderY[runvalue] + neiby[k];
				if (nx < width && nx >= 0 && ny < height && ny >= 0)//邻域坐标在图像尺寸上下限内
				{
					unsigned short* nebinputimgpos = matConectlabel.pdata + ny * width + nx;//输入图像邻域位置
					if (*nebinputimgpos == 1)
					{
						countnumber = countnumber + 1;
						orderX.push_back(nx);// 存储此邻域点的横坐标
						orderY.push_back(ny);// 存储此领域点的纵坐标
						*nebinputimgpos = 0;//原始输入图像此邻域点的值赋为0
					}
				}
			}
			runvalue = runvalue + 1;//  追赶连通域的总像素数目，直到追上countnumber 
		}

	}

}

void CTRUnet_Detection::GetHeartDiaphPoint_leftlung(vector<int>& Xdiaph_order, vector<int>& Ydiaph_order, int& maxdisind, vector<float>& nvect_ratio, int leftmaxdis_Y, int leftrightparm)
{
	//左肺处理
	int w = Xdiaph_order.size();
	int detapix = 20;//步长

	int numx = w / detapix;//段数
	int startpix = detapix / 2;

	//曲线的平滑处理
	vector<float> nXXmeanvect;
	vector<float> nYYmeanvect;
	Line_Smooth(Xdiaph_order, Ydiaph_order, detapix, nXXmeanvect, nYYmeanvect);

	//斜率计算
	//曲线开头的斜率计算
	Line_Ratio(nXXmeanvect, nYYmeanvect, detapix, nvect_ratio);

	//弓形距离的计算
	vector<double> Vertical_disvector;//曲线上每一点的到首尾向量的垂距
	ARC_Distance(nXXmeanvect, nYYmeanvect, Vertical_disvector, leftrightparm);

	int startind = 0;
	int endind = 0;
	for (int i = 0; i < nYYmeanvect.size(); i++)
	{
		int compr = Ydiaph_order[i];
		if (compr == leftmaxdis_Y)//根据右肺的Y坐标向上提高20个像素的距离进行遍历
		{
			startind = i;
			break;
		}
		else
		{
			startind = Vertical_disvector.size() / 3;//从1/3处开始寻找
		}
	}

	//找到垂距最大的那个点的位置
	maxdisind = max_element(Vertical_disvector.begin() + startind, Vertical_disvector.end())
		- Vertical_disvector.begin();
}

void CTRUnet_Detection::GetHeartDiaphPoint_rightlung(vector<int>& Xdiaph_order, vector<int>& Ydiaph_order, int& maxdisind, vector<float>& nvect_ratio, int leftrightparm)
{
	int w = Xdiaph_order.size();
	int detapix = 20;//步长

	int numx = w / detapix;//段数
	int startpix = detapix / 2;

	//曲线的平滑处理
	vector<float> nXXmeanvect;
	vector<float> nYYmeanvect;
	Line_Smooth(Xdiaph_order, Ydiaph_order, detapix, nXXmeanvect, nYYmeanvect);

	//斜率计算
	//曲线开头的斜率计算
	Line_Ratio(nXXmeanvect, nYYmeanvect, detapix, nvect_ratio);

	//弓形距离的计算
	vector<double> Vertical_disvector;//曲线上每一点的到首尾向量的垂距
	ARC_Distance(nXXmeanvect, nYYmeanvect, Vertical_disvector, leftrightparm);

	//找到垂距最大的那个点的位置
	maxdisind = max_element(Vertical_disvector.begin(), Vertical_disvector.end())
		- Vertical_disvector.begin();
}

void CTRUnet_Detection::Line_Smooth(vector<int>& Xdiaph_order, vector<int>& Ydiaph_order,
	int detapix, vector<float>& nXXmeanvect, vector<float>& nYYmeanvect)
{
	int w = Xdiaph_order.size();

	int numx = w / detapix;//段数
	int startpix = detapix / 2;

	//曲线开头的移动平均计算
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

	//曲线中央的移动平均计算
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

	//曲线末尾的移动平均计算，直到最后一个数据
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

	int numx = w / detapix;//段数
	int startpix = detapix / 2;

	//曲线开头的斜率的计算
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

	//曲线中央数据的斜率计算
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

	//曲线结尾的斜率计算
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
	// 弓形距离
	//首尾之间的基向量
	vector<double> StEndvector;//向量两个元素，第一个是X，第二个是Y
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
		//曲线上每一点到起始点的向量
		Midvector[0] = nXXmeanvect[i] - nXXmeanvect[0];
		Midvector[1] = nYYmeanvect[i] - nYYmeanvect[0];

		//曲线上每一点到起始点的距离
		Dis_Midvector = sqrtl(Midvector[0] * Midvector[0] +
			Midvector[1] * Midvector[1]);

		//曲线上每一点到起始点的向量与首尾向量之间的夹角
		//曲线上每一点到起始点的向量与首尾向量之间的夹角
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

		//曲线上每一点的垂距
		Vertical_disvector.push_back(Dis_Midvector * sintheta);
	}
}

void CTRUnet_Detection::Find_hartdiaphind_smallheart(vector<int>& nXXsmall_vect, vector<int>& nYYsmall_vect,
	vector<float>& nvect_ratio, int& maxdisind, int& hart_Leftind, int& diaph_Leftind, int leftrightparm)
{
	//根据斜率的绝对值最大的那个值，找到心脏位置的有效点
	int detadis = 100;//100
	hart_Leftind = maxdisind;
	if ((maxdisind - detadis) > 0)
	{
		int detapix = 10;
		//曲线拐点结尾处向前10个单位的斜率重新计算
		for (int i = maxdisind - detapix; i < maxdisind; i++)
		{
			float nx1 = nXXsmall_vect[i - detapix];
			float nx2 = nXXsmall_vect[i];

			float ny1 = nYYsmall_vect[i - detapix];
			float ny2 = nYYsmall_vect[i];

			if (abs(ny1 - ny2) < 1e-5)
			{
				nvect_ratio[i] = 10;// 拐点前向前10个单位           
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

		for (int i = 0; i < detadis - 10; i++)//找心脏点
		{
			if (hartchest > abs(nvect_ratio[tempind + i]))//找斜率模值最小的
			{
				if (abs(nvect_ratio[tempind + i]) != 10)
				{
					hartchest = abs(nvect_ratio[tempind + i]);
					hart_Leftind = tempind + i;
				}
			}

			if (leftrightparm == -1)//右侧
			{
				if (nXXsmall_vect[tempind + i] > hart_smallX)//找到最外侧，对于左侧找最左边，右侧找最右边
				{
					hart_smallX = nXXsmall_vect[tempind + i];
					Current_smallX = tempind + i;
				}
			}
			else//左侧
			{
				if (nXXsmall_vect[tempind + i] < hart_smallX)//找到最外侧，对于左侧找最左边，右侧找最右边
				{
					hart_smallX = nXXsmall_vect[tempind + i];
					Current_smallX = tempind + i;
				}
			}
		}

		if (leftrightparm == -1)//右侧
		{
			if (nXXsmall_vect[Current_smallX] > nXXsmall_vect[hart_Leftind])
			{
				hart_Leftind = Current_smallX;
			}
		}
		else//左侧
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

		for (int i = tempind; i >= 0; i--)//找心脏点
		{
			if (hartchest > abs(nvect_ratio[i]))//找斜率模值最小的
			{
				if (abs(nvect_ratio[i]) != 10)
				{
					hartchest = abs(nvect_ratio[i]);
					hart_Leftind = i;
				}
			}

			if (leftrightparm == -1)//右侧
			{
				if (nXXsmall_vect[i] > hart_smallX)//找到最外侧，对于左侧找最左边，右侧找最右边
				{
					hart_smallX = nXXsmall_vect[i];
					Current_smallX = i;
				}
			}
			else//左侧
			{
				if (nXXsmall_vect[i] < hart_smallX)//找到最外侧，对于左侧找最左边，右侧找最右边
				{
					hart_smallX = nXXsmall_vect[i];
					Current_smallX = i;
				}
			}
		}

		if (leftrightparm == -1)//右侧
		{
			if (nXXsmall_vect[Current_smallX] > nXXsmall_vect[hart_Leftind])
			{
				hart_Leftind = Current_smallX;
			}
		}
		else//左侧
		{
			if (nXXsmall_vect[Current_smallX] < nXXsmall_vect[hart_Leftind])
			{
				hart_Leftind = Current_smallX;
			}
		}
	}

	//根据斜率的绝对值最大的那个值，找到膈肌的位置
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
	//	for (int i = maxdisind; i < nvect_ratio.size(); i++)//找到斜率模值最大的
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
	vector<float>& nvect_rightratio, int& maxdisind, int& hart_Rightind, int& diaph_Rightind, int leftrightparm)//leftrightparm左侧为1，右侧为-1
{
	// 根据斜率的绝对值最逼近零的位置从而找到心脏边缘位置 
	int detadis = 50;//50  
	hart_Rightind = maxdisind;
	if ((maxdisind - detadis) > 0)
	{
		int detapix = 10;
		//曲线拐点结尾处向前10个单位的斜率重新计算
		for (int i = maxdisind - detapix; i < maxdisind; i++)
		{
			float nx1 = nXXlarge_vect[i - detapix];
			float nx2 = nXXlarge_vect[i];

			float ny1 = nYYlarge_vect[i - detapix];
			float ny2 = nYYlarge_vect[i];

			if (abs(ny1 - ny2) < 1e-5)
			{
				nvect_rightratio[i] = 10;// 拐点前向前10个单位           
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
		for (int i = 0; i < detadis - 5; i++)//从前detadis个数据点斜率中寻找斜率模值最小，找心脏点
		{
			if (hartchest > abs(nvect_rightratio[tempind + i]))//找到斜率模值最小的
			{
				if (abs(nvect_rightratio[tempind + i]) != 10)
				{
					hartchest = abs(nvect_rightratio[tempind + i]);
					hart_Rightind = tempind + i;
				}
			}

			if (leftrightparm == -1)//右侧
			{
				if (nXXlarge_vect[tempind + i] > hart_RightX)//找到最外侧，对于左侧找最左边，右侧找最右边
				{
					hart_RightX = nXXlarge_vect[tempind + i];
					Current_RightX = tempind + i;
				}
			}
			else//左侧
			{
				if (nXXlarge_vect[tempind + i] < hart_RightX)//找到最外侧，对于左侧找最左边，右侧找最右边
				{
					hart_RightX = nXXlarge_vect[tempind + i];
					Current_RightX = tempind + i;
				}
			}

		}

		if (leftrightparm == -1)//右侧
		{
			if (nXXlarge_vect[Current_RightX] > nXXlarge_vect[hart_Rightind])
			{
				hart_Rightind = Current_RightX;
			}
		}
		else//左侧
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
		for (int i = tempind; i >=0; i--)//从前detadis个数据点斜率中寻找斜率模值最小，找心脏点
		{
			if (hartchest > abs(nvect_rightratio[i]))//找到斜率模值最小的
			{
				if (abs(nvect_rightratio[i]) != 10)
				{
					hartchest = abs(nvect_rightratio[i]);
					hart_Rightind = i;
				}
			}

			if (leftrightparm == -1)//右侧
			{
				if (nXXlarge_vect[i] > hart_RightX)//找到最外侧，对于左侧找最左边，右侧找最右边
				{
					hart_RightX = nXXlarge_vect[ i];
					Current_RightX = i;
				}
			}
			else//左侧
			{
				if (nXXlarge_vect[i] < hart_RightX)//找到最外侧，对于左侧找最左边，右侧找最右边
				{
					hart_RightX = nXXlarge_vect[ i];
					Current_RightX = i;
				}
			}
		}

		if (leftrightparm == -1)//右侧
		{
			if (nXXlarge_vect[Current_RightX] > nXXlarge_vect[hart_Rightind])
			{
				hart_Rightind = Current_RightX;
			}
		}
		else//左侧
		{
			if (nXXlarge_vect[Current_RightX] < nXXlarge_vect[hart_Rightind])
			{
				hart_Rightind = Current_RightX;
			}
		}
	}

	////根据斜率的绝对值最大的那个值，找到膈肌的位置
	//detadis = 30;
	//diaph_Rightind = maxdisind;
	//if ((maxdisind + detadis) < nvect_rightratio.size())
	//{
	//	int tempind = maxdisind;
	//	double diaphchest = abs(nvect_rightratio[tempind]);
	//	for (int i = 5; i < detadis; i++)
	//	{
	//		if (diaphchest < abs(nvect_rightratio[tempind + i]))//找到斜率最大的
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

	//左侧构建
	for (int i = 0; i < Ydiaph_Leftorder.size(); i++)
	{
		matLeftdis.pdata[Ydiaph_Leftorder[i] * matLeftdis.width + Xdiaph_Leftorder[i]] = 1;
	}

	//右侧构建
	for (int i = 0; i < Ydiaph_Rightorder.size(); i++)
	{
		matRightdis.pdata[Ydiaph_Rightorder[i] * matRightdis.width + Xdiaph_Rightorder[i]] = 1;
	}

	//计算左侧纵隔距离中央的距离
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

	//计算右侧纵隔距离中央的距离
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

	//对输入图像进行前景和背景反转
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

	//消除背景孔洞
	Matrix<unsigned short> imgcomple_inv(width, height);
	signum = Eliminateholes(imgcomple, imgcomple_inv, ratiovalue);//消除背景孔洞
	if (signum != 1)
	{
		//MYLog.WriteLog(1,"消除背景孔洞，输入图像没有连通域");
		return 0;
	}

	//再次前景和背景反转
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
	signum = Eliminateholes(imgcomple, imgcomple_inv, ratiovalue);//消除前景孔  
	if (signum != 1)
	{
		//MYLog.WriteLog(1, "消除前景孔洞，输入图像没有连通域");
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
	//labnum的数值代表连通域的大小，索引值代表连通域的标号，标号体现在bwimglabel的连通域图中
	Conectchose(matInputimg, matBWimgblabel, labval, labind, FALSE);
	if (labind.empty())
	{
		return 0;
	}

	//将所有连通域的大小求和
	double connectsum = accumulate(labval.begin(), labval.end(), 0);
	double uptovalue = connectsum * ratiovalue;//比率值乘以连通域总值获取连通域截止值
	int uptoind = -1;
	for (int i = 0; i < labval.size(); i++)
	{
		if (labval[i] > uptovalue)
		{
			uptoind = i; //获取连通域的截止索引 。
			break;
		}
		else
		{
			uptoind = -1;
		}
	}

	//开启孔洞填充or连通域筛选的判断和循环
	if (uptoind != -1)//说明获取到了连通域的截至索引
	{
		for (int i = 0; i < widthheight; i++)
		{
			if (matBWimgblabel.pdata[i] != 0)
			{
				int labelnumber = matBWimgblabel.pdata[i];//图像中的连通域标号
				for (int j = 0; j < uptoind; j++)//遍历小连通域的标号
				{
					if (labelnumber == labind[j])//如果小连通域的标号和图像中的标号相对应则进行赋值为0的操作
					{
						matBWimgblabel.pdata[i] = 0; // 只要满足即可置为赋值为0
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
		if (matBWimgblabel.pdata[i] != 0)//输入不为0的连通域
		{
			matOutputimg.pdata[i] = 1;//纳入到输出数组接口中
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

	//赋值matinputmasklung数组作为连通域处理的输入数组
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
	//labnum的数值代表连通域的大小，索引值代表连通域的标号，标号体现在bwimglabel的连通域图中
	Conectchose(matinputmasklung, matmasklung_copy, labval, labind, FALSE);
	if (labind.empty())
	{
		return 0;
	}

	int sigm = 0;
	int marknum = 0;

	//从上到下，从左到右，获取左侧的肺部区域
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

	//左侧肺野
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

	//从从上到下，从右到左，获取右侧的肺部区域
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

	//右侧肺野
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

//直线链接算法
void CTRUnet_Detection::DDALine11(unsigned short* neblabelimgpos, int startx, int starty, int endx, int endy, int pixelvalue, int width, int height)
{
	double x, y, tmp, xIncre, yIncre;
	int dx = endx - startx;
	int dy = endy - starty;
	tmp = max(abs(dx), abs(dy));
	xIncre = 1.0 * dx / tmp;  // 增量
	yIncre = 1.0 * dy / tmp;
	x = startx;  // 当前需要绘制的点
	y = starty;
	for (int i = 0; i < tmp; i++)
	{
		x += xIncre;
		y += yIncre;
		neblabelimgpos[int(x + 0.5) + int(y + 0.5)*width] = pixelvalue;
	}
}

//心胸比和特征点的映射
void CTRUnet_Detection::MaptoOrg_CTR(int imgwidth, int imgheight, int downsampwidth, int downsampheight, int params[])
{
	//对paramsY坐标映射
	for (int i = 3; i <= 26; i += 2)
	{
		params[i] = round(float(params[i]+1) / downsampheight * imgheight);
	}
	//对paramsX坐标映射
	for (int i = 4; i <= 26; i += 2)
	{
		params[i] = round(float(params[i]+1) / downsampwidth * imgwidth);
	}
}

//横膈边界映射
void CTRUnet_Detection::MaptoOrg_Diaph(int imgwidth, int imgheight, int downsampwidth, int downsampheight, unsigned char* pdiaph_Line_img, unsigned short* pdiaph_Line_imgorg)
{
	//对横膈膜边界进行映射
	vector<int> nxvect;//连通域横坐标收纳盒
	vector<int> nyvect;//连通域纵坐标收纳盒
	int neibx[8] = { -1, 0, 1, 1, 1, 0, -1, -1 }; //八邻域x方向
	int neiby[8] = { -1, -1, -1, 0, 1, 1, 1, 0 }; //八邻域y方向
   
 	//对放大的不规则曲线进行映射                                        
	for (int i = 0; i < imgheight; i++)
	{
		for (int j = 0; j < imgwidth; j++)
		{

			int mapwidth = float(j+1) / float(imgwidth) * (downsampwidth - 1);
			int mapheight = float(i+1) / float(imgheight) * (downsampheight - 1);
			int index = mapheight * downsampwidth + mapwidth;
			if (pdiaph_Line_img[index] != 0)//如果小图不为0
			{
				int orgindex = i * imgwidth + j;
				pdiaph_Line_imgorg[orgindex] = pdiaph_Line_img[index];//将小图在此坐标的值赋值给大图
				pdiaph_Line_img[index] = 0;//小图置0。                                             

				//领域判定
				int runvalue = 0;
				int countnumber = 0; // 邻域计数
				nxvect.push_back(mapwidth);//
				nyvect.push_back(mapheight); // 将小图有效坐标存储到vector向量数组中   

				while (runvalue <= countnumber)
				{

					for (int k = 0; k < 8; k++)
					{
						int nx = nxvect[runvalue] + neibx[k];
						int ny = nyvect[runvalue] + neiby[k];

						//大图输出图像此领域点的坐标位置
						int centerwith = float(nxvect[runvalue]) / (downsampwidth - 1) * float(imgwidth - 1);
						int centerheight = float(nyvect[runvalue]) / (downsampheight - 1) * float(imgheight - 1);


						if (nx < downsampwidth && nx >= 0 && ny < downsampheight && ny >= 0)//邻域坐标在小图像尺寸上下限内
						{
							unsigned char* nebinputimgpos = pdiaph_Line_img + ny * downsampwidth + nx;//输入图像邻域位置

							//输出图像此领域点的坐标位置
							int currwith = float(nx) / (downsampwidth - 1) * float(imgwidth);
							int currheight = float(ny) / (downsampheight - 1) * float(imgheight);

							int currindex = currheight * imgwidth + currwith;
							unsigned short* neblabelimgpos = pdiaph_Line_imgorg + currindex;//大图像输出图像此邻域点的位置
							if (*nebinputimgpos != 0)//  小图的领域为真，则执行填充操作
							{
								countnumber = countnumber + 1;
								nxvect.push_back(nx);//存储小图此邻域点的横坐标
								nyvect.push_back(ny);//存储小图此领域点的纵坐标

								//新图像点的扩大填充起始点
								DDALine11(pdiaph_Line_imgorg, currwith, currheight, centerwith, centerheight, int(*nebinputimgpos), imgwidth, imgheight);//直线划线法，输入两点坐标，绘制直线并赋值
								*nebinputimgpos = 0;//原始输入图像此邻域点的值赋为0
							}
						}
					}
					runvalue = runvalue + 1;//追赶有效点
				}
			}
			nxvect.clear();
			nyvect.clear();
		}
	}
}

//左右肺识别和肺面积映射
void CTRUnet_Detection::MaptoOrg_Areacal(int imgwidth, int imgheight, int downsampwidth, int downsampheight, int params[], unsigned char* pleftrightlung_imgmask, unsigned short* pleftrightlung_imgmaskorg)
{
	//对左右肺部识别区域进行映射
	for (int i = 0; i < imgheight; i++)
	{
		for (int j = 0; j < imgwidth; j++)
		{

			int mapwidth = float(j + 1) / float(imgwidth) * (downsampwidth - 1);
			int mapheight = float(i + 1) / float(imgheight) * (downsampheight - 1);
			pleftrightlung_imgmaskorg[i * imgwidth + j] = pleftrightlung_imgmask[mapheight * downsampwidth + mapwidth];
		}
	}

	//对面积进行映射
	params[27] = params[27] * (float(imgwidth) / downsampwidth)*(float(imgheight) / downsampheight);
	params[28] = params[28] * (float(imgwidth) / downsampwidth)*(float(imgheight) / downsampheight);
}