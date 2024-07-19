#pragma once
#include"Matrix_calculate.h"
#include<map>
#include<numeric>
#include<opencv2/opencv.hpp>


using namespace std;

class CTRUnet_Detection
{
public:

	//心胸比点检测
	/*************************************************
	Function: 	    	// CTR_main
	Description: 		//心胸比和特征点坐标位置检测
	Input:        	// presult_img为输入的肺野图像地址，width和height为输入图像的宽高；
	Output: 		// params为输出坐标的数组，具体含义见接口定义
	Others: 		//
	*******************************************/
    int CTR_main(unsigned char* presult_img, int downsampwidth, int downsampheight,  int params[]);

	//膈肌运动点检测
	/*************************************************
	Function: 	    	// Diaphragm_detect
	Description: 		// 此函数根据左右纵膈边界距离胸椎中线的平均距离判定左右肺野
	Input:        	// presult_img为输入的肺野图像地址，width和height为输入图像的宽高；
	Output: 		// pdiaph_Line_img为输出图像地址，图像中包含膈肌运动点，左侧横膈肌边界像素值值为128，右侧值为255
	Others: 		//
	*******************************************/
	int Diaphragm_detect(unsigned char* presult_img, int downsampwidth, int downsampheight, int params[],
		unsigned short* pdiaph_Line_imgorg); 

	//左右肺识别检测
	/*************************************************
	Function: 	    	// Lung_Areacalculate
	Description: 		// 此函数根据左右纵膈边界距离胸椎中线的平均距离判定左右肺
	Input:        	// presult_img为输入的肺野图像地址，width和height为输入图像的宽高；
	Output: 		// pleftrightlung_imgmask输出识别的左右肺，右肺值1，左肺值2； params[27]左侧肺面积;params[28]右侧肺面积 params[29]值为1左侧为右肺，右侧为左肺；值为2左侧为左肺，右侧为右肺
	Others: 		// 
	*******************************************/
	int Lung_Areacalculate(unsigned char* presult_img, int width, int height, unsigned short* pleftrightlung_imgmaskorg, int params[]);

	/*************************************************
	Function: 	    	// Erase_holl1
	Description: 		// 孔洞填充和消除：对输入图像的背景和前景的小区区域孔洞连通域进行消除
	matInputimg:           输入图像
	ratiovalue：           消除比率
	Others: 		// 返回值为1为函数运行成功，-1为失败
	*************************************************/
	int Erase_holl(Matrix<unsigned short>& matInputimg, double ratiovalue);

	/*************************************************
	Function: 	    	// Eliminateholes
	Description: 		// 孔洞（连通域）填充优化函数，去除杂散孔洞连通域，低于比率下的连通域消除
	Input:        	// matInputimg:输入图像，ratiovalue为消除的比率
	Output: 		// matOutputimg为输出图像
	Others: 		// 此函数的输入inputimg数组会在函数执行结束后全置零,返回值为1为函数运行成功，-1为失败
	*************************************************/
	int Eliminateholes(Matrix<unsigned short>& matInputimg, Matrix<unsigned short>& matOutputimg, double ratiovalue);

	/*************************************************
	Function: 	    	// Conectchose
	Description: 		// 连通域选择
	Input:        	// matInputimg：输入图像的首地址
	Output: 		// matLabelimg：输出连通域标记图 连通域安装从小到大的顺序存入向量中labval（连通域大小）和labind（连通域标号）；TF假为四联通真为八连通
	Others: 		// 完成连通域检测后，输入数组matInputimg会置0（数组中所有的1会置为0）
	*******************************************/
	void Conectchose(Matrix<unsigned short>& matInputimg, Matrix<unsigned short>& matLabelimg, vector<int>& labval, vector<int>& labind, bool TF);

	/*************************************************
	Function: 	    	// Erodeimg
	Description: 		// 腐蚀运算
	Input:        	// inputimg：输入图像的首地址；width和height分别为图像的宽高；templateerode：模板首地址；
	templateerodewidth和templateerodeheight为模板宽高
	Output: 		// outputimg：腐蚀后的图像
	Others: 		// 边界采用复制延拓方式
	*************************************************/
	void Erodeimg(Matrix<unsigned short>& matInputimg, Matrix<unsigned short>& templatedilate,
		Matrix<unsigned short>& matOutputimg);

	/*************************************************
	Function: 	    	// Dilateimg
	Description: 		// 膨胀运算
	Input:        	// inputimg：输入图像的首地址；width和height分别为图像的宽高；templateerode：模板首地址；
	templateerodewidth和templateerodeheight为模板宽高
	Output: 		// outputimg：膨胀后的图像
	Others: 		// 边界采用复制延拓方式
	*************************************************/
	void Dilateimg(Matrix<unsigned short>& matInputimg, Matrix<unsigned short>& templatedilate,
		Matrix<unsigned short>& matOutputimg);  //

	/*************************************************
	Function: 	    	// Lungmask_clean
	Description: 		//肺野区域的连通域处理
	Input:        	// presult_img为输入的肺野图像地址，width和height为输入图像的宽高；
	Output: 		// params为输出坐标的数组，具体含义见接口定义
	Others: 		//
	*******************************************/
	void Lungmask_clean(cv::Mat& result_img);

private:

	//心胸比和特征点的原始图像映射
	/*************************************************
	Function: 	    	// MaptoOrg_CTR
	Description: 		// 心胸比特征点的坐标映射到原始图像中的坐标位置
	Input:        	// params坐标矩阵，imgwidth和imgheight为原始图的宽高；downsampwidth和downsampheight为下采样图像宽高
	Output: 		// params映射到对应定义的接口位置
	Others: 		// 无
	*******************************************/
	void MaptoOrg_CTR(int imgwidth, int imgheight, int downsampwidth, int downsampheight, int params[]);

	//横膈肌边界曲线的原始图像映射
	/*************************************************
	Function: 	    	// MaptoOrg_Diaph
	Description: 		// 横膈边界曲线映射到原始图
	Input:        	// pdiaph_Line_img为下采样图中的横膈边界曲线，左侧曲线像素值为128，右侧曲线像素值为255；imgwidth和imgheight为原始图的宽高；downsampwidth和downsampheight为下采样图像宽高
	Output: 		// pdiaph_Line_imgorg为映射到原始图像的横膈边界图，左侧曲线值为128，右侧曲线值为255
	Others: 		// 采用了连通域检测 + DDA直线法
	*******************************************/
	void MaptoOrg_Diaph(int imgwidth, int imgheight, int downsampwidth, int downsampheight, unsigned char* pdiaph_Line_img, unsigned short* pdiaph_Line_imgorg);

	//左右肺野识别的原始图像映射
	/*************************************************
	Function: 	    	// MaptoOrg_Areacal
	Description: 		// 肺野面积映射，将下采样的小面积映射到原图大小
	Input:        	// pleftrightlung_imgmask输入的下采样左右肺野图像；imgwidth和imgheight为原始图的宽高；downsampwidth和downsampheight为下采样图像宽高
	Output: 		// pleftrightlung_imgmaskorg：为原始图像的左右肺野识别；params[27]params[28]左右侧原始图像肺野面积的计算
	Others: 		// 面积的计算通过根据宽高放大的倍数计算而来
	*******************************************/
	void MaptoOrg_Areacal(int imgwidth, int imgheight, int downsampwidth, int downsampheight, int params[], unsigned char* pleftrightlung_imgmask, unsigned short* pleftrightlung_imgmaskorg);//

	/*************************************************
	Function: 	    	// LeftRightSegimg
	Description: 		// 此函数实现两个功能，一根据肺野面积将肺野图分割为左肺野图和右肺野图；2,输出左侧肺野边界和右侧肺野边界
	Input:        	// matmasklung为输入的肺野掩模图像也可视为输出图像
	Output: 		// matLeftChest为输出的左侧肺野图像，matRightChest为输出的右侧肺野图像（只含有边界轮廓）xvec_left等四个向量为左右侧肺野轮廓的边界坐标
	Others: 		// 此函数matmasklung输入数组可以作为使用面积大小来判别左右肺的输出数组，面积大的为右肺置为1，面积小的为左肺置为2
	*******************************************/
	int LeftRightSegimg(Matrix<unsigned short>& matmasklung, Matrix<unsigned short>& matLeftChest, Matrix<unsigned short>& matRightChest,
		vector<int>& xvec_left, vector<int>& yvec_left, vector<int>& xvec_right, vector<int>& yvec_right);

	/*************************************************
	Function: 	    	// LeftRightsideimg
	Description: 		// 此函数将肺野图分割为左侧肺野图和右侧肺野图
	Input:        	// matmasklung为输入的肺野图像；
	Output: 		// matleftlungmask_copy为输出的左侧肺野图像，matrightlungmask_copy为输出的右侧肺野图像
	Others: 		// 只区分左侧和右侧肺野图像
	*******************************************/
	int LeftRightsideimg(Matrix<unsigned short>& matmasklung, Matrix<unsigned short>& matleftlungmask_copy, Matrix<unsigned short>& matrightlungmask_copy);

	/*************************************************
	Function: 	    	// DDALine11
	Description: 		// 直线划线法
	Input:        	// neblabelimgpos为输入需要划线的图像；width和height为图像的宽高；startx和starty为起始坐标，endx和endy为终止坐标。pixelvalue为划线的像素值
	Output: 		// neblabelimgpos：输出两点之间的线段，线段中的像素值为pixelvalue
	Others: 		// 无
	*******************************************/
	void DDALine11(unsigned short* neblabelimgpos, int startx, int starty, int endx, int endy, int pixelvalue, int width, int height);

	/*************************************************
	Function: 	    	// RibDiaphSeg_left
	Description: 		// 左侧肺野肋缘和膈膜的分割
	Input:        	// matLeftChest：输入左侧肺野图像的首地址；xvec_left和yvec_left为左侧肺野边界的坐标；
	Output: 		// xvec_leftrib和yvec_leftrib分别是右侧肋缘的坐标；xvec_leftdiaph和yvec_leftdiaph分别为右侧横纵膈膜边界坐标
	Others: 		// 肺野边界的顶部和底部使用连通域进行了断开
	*******************************************/
	int RibDiaphSeg_left(Matrix<unsigned short>& matLeftChest,
		vector<int>& xvec_left, vector<int>& yvec_left,
		vector<int>& xvec_leftrib, vector<int>& yvec_leftrib, vector<int>& xvec_leftdiaph, vector<int>& yvec_leftdiaph);

	/*************************************************
	Function: 	    	// RibDiaphSeg_right
	Description: 		// 右侧肺野肋缘和膈膜的分割
	Input:        	// matRightChest：输入右侧肺野图像的首地址；xvec_right和yvec_right为右侧肺野边界的坐标；
	Output: 		// xvec_rightrib和yvec_rightrib分别是右侧肋缘的坐标；xvec_rightdiaph和yvec_rightdiaph分别为右侧横纵膈膜边界坐标
	Others: 		// 肺野边界的顶部和底部使用连通域进行了断开
	*******************************************/
	int RibDiaphSeg_right(Matrix<unsigned short>& matRightChest,
		vector<int>& xvec_right, vector<int>& yvec_right,
		vector<int>& xvec_rightrib, vector<int>& yvec_rightrib, vector<int>& xvec_rightdiaph, vector<int>& yvec_rightdiaph);

	/*************************************************
	Function: 	    	// GetorderDiaph。
	Description: 		// 获取连通域的运算运算。
	Input:        	// matLableimg：输入图像的首地址；Xdiaph_Leftorder为你后天出生的角色。
	xvec_leftdiaph和yvec_leftdiaph为分别为左右左右两侧，Xdiaph_Leftorder和Ydiaph_Leftorder为册立腰部的椎骨。
	Output: 		// outputimg：膨胀后的图像。
	Others: 		// 边界采用复制延拓方式。
	*************************************************/
	void GetorderDiaph(Matrix<unsigned short>& matLableimg, vector<int>& xvec_leftdiaph, vector<int>& yvec_leftdiaph,
		vector<int>& Xdiaph_Leftorder, vector<int>& Ydiaph_Leftorder);//  

	/*************************************************
	Function: 	    	// Search_Conectpoint
	Description: 		// 连通域的排列函数，给出连通域的起始，从起始点开始搜索连通域的坐标
	Input:        	// matConectlabel：输入图像连通域图像；startY和startX为分别为起始点的Y和X坐标
	Output: 		// orderY和orderX分别为从起始到终止点的连通域的Y和X坐标
	Others: 		// 此函数从新排列了连通域的坐标顺序
	*************************************************/
	void Search_Conectpoint(Matrix<unsigned short>& matConectlabel, int startX, int startY,
		vector<int>& orderX, vector<int>& orderY);

	/*************************************************
	 Function: 	    	// GetHeartDiaphPoint_leftlung和GetHeartDiaphPoint_rightlung
	 Description: 		// 此函数用于计算和获取横纵隔边界的拐点位置，并获取曲线的斜率
	 Input:        	// Ydiaph_order和Xdiaph_order分别为横纵隔边界坐标，maxdisind为拐点的位置
	 Output: 		// nvect_ratio为横纵隔膜边界的斜率
	 Others: 		// 此函数从新排列了连通域的坐标顺序
	 *************************************************/
	void GetHeartDiaphPoint_leftlung(vector<int>& Xdiaph_order, vector<int>& Ydiaph_order,
		int& maxdisind, vector<float>& nvect_ratio, int leftmaxdis_Y, int leftrightparm);

	void GetHeartDiaphPoint_rightlung(vector<int>& Xdiaph_order, vector<int>& Ydiaph_order,
		int& maxdisind, vector<float>& nvect_ratio, int leftrightparm);
	/*************************************************
	Function: 	    	// Line_Smooth
	Description: 		// 此函数对有序曲线进行平滑
	Input:        	// Xdiaph_order和Ydiaph_order分别为输入曲线的坐标（从起始到终止）；detapix：为平滑间距
	Output: 		// nXXmeanvect和nYYmeanvect分别为经过平均滤波平滑后的连通域的X和Y坐标
	Others: 		// 此函数对有序曲线进行了平滑
	*************************************************/
	void Line_Smooth(vector<int>& Xdiaph_order, vector<int>& Ydiaph_order, int detapix,
		vector<float>& nXXmeanvect, vector<float>& nYYmeanvect);

	/*************************************************
	Function: 	    	// Line_Ratio
	Description: 		// 计算曲线上每一点的斜率
	Input:        	// nXXmeanvect和nYYmeanvect分别为输入曲线的坐标（从起始到终止）；detapix：计算斜率的间距
	Output: 		// nvect_ratio：为曲线上每一点的斜率
	Others: 		// 此函数计算曲线斜率
	*************************************************/
	void Line_Ratio(vector<float>& nXXmeanvect, vector<float>& nYYmeanvect, int detapix, vector<float>& nvect_ratio);

	/*************************************************
	Function: 	    	// ARC_Distance
	Description: 		// 计算的弓形距离
	Input:        	// nXXmeanvect和nYYmeanvect分别为输入曲线的坐标（从起始到终止）
	Output: 		// Vertical_disvector：为曲线上每一点的弓形距离
	Others: 		// 此函数计算曲线的弓形距离
	*************************************************/
	void ARC_Distance(vector<float>& nXXmeanvect, vector<float>& nYYmeanvect, vector<double>& Vertical_disvector, int posnegvalue);

	/*************************************************
	Function: 	    	// Find_hartdiaphind_smallheart
	Description: 		// 人体右心脏肺野的有效点检测包括右横膈肌点和右心脏边缘点的检测
	Input:        	// nXXsmall_vect和nYYsmall_vect分别为输入右横隔边界有序曲线的坐标（从起始到终止）；nvect_ratio：曲线的斜率；maxdisind为拐点位置
	Output: 		// hart_Leftind和diaph_Leftind分别为右侧心脏有效点的位置和横隔肌边界有效点的位置
	Others: 		// 用于获取右肺有效点的位置信息
	*************************************************/
	void Find_hartdiaphind_smallheart(vector<int>& nXXsmall_vect, vector<int>& nYYsmall_vect, vector<float>& nvect_ratio,
		int& maxdisind, int& hart_Leftind, int& diaph_Leftind, int leftrightparm);

	/*************************************************
	Function: 	    	// Find_hartdiaphind_largeheart
	Description: 		// 人体左心脏肺野的有效点检测包括左横膈肌点和左心脏边缘点的检测
	Input:        	// nXXlarge_vect和nYYlarge_vect分别为输入左横隔边界有序曲线的坐标（从起始到终止）；nvect_ratio：曲线的斜率;maxdisind为拐点位置
	Output: 		// hart_Leftind和diaph_Leftind分别为左侧心脏有效点的位置和横隔肌边界有效点的位置
	Others: 		// 用于获取左肺有效点的位置信息
	*************************************************/
	void Find_hartdiaphind_largeheart(vector<int>& nXXlarge_vect, vector<int>& nYYlarge_vect, vector<float>& nvect_ratio,
		int& maxdisind, int& hart_Rightind, int& diaph_Rightind, int leftrightparm);

	/*************************************************
	 Function: 	    	// LeftRight_dist
	 Description: 		// 计算中间区域纵膈距离胸椎中心线的平均距离，通过此后续来判定左右肺
	 Input:        	// xribdataleft与Xdiaph_Leftorder等左右肋缘和横纵隔膜边界坐标点，Midseg_X为中心线位置
	 Output: 		// 输出左侧LeftdisX和右侧RightdisX距离
	 Others: 		// 此函数计算曲线的弓形距离
	 *************************************************/
	void LeftRight_dist(int& LeftdisX, int& RightdisX, int Midseg_X, Matrix<unsigned short>& matLeftdis, Matrix<unsigned short>& matRightdis,
		vector<int>& xribdataleft, vector<int>& yribdataleft, vector<int>& xribdataright, vector<int>& yribdataright,
		vector<int>& Xdiaph_Leftorder, vector<int>& Ydiaph_Leftorder, vector<int>& Xdiaph_Rightorder, vector<int>& Ydiaph_Rightorder);

};




