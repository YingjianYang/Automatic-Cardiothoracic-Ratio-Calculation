#pragma once
#include"Matrix_calculate.h"
#include<map>
#include<numeric>
#include<opencv2/opencv.hpp>


using namespace std;

class CTRUnet_Detection
{
public:

	//���رȵ���
	/*************************************************
	Function: 	    	// CTR_main
	Description: 		//���رȺ�����������λ�ü��
	Input:        	// presult_imgΪ����ķ�Ұͼ���ַ��width��heightΪ����ͼ��Ŀ�ߣ�
	Output: 		// paramsΪ�����������飬���庬����ӿڶ���
	Others: 		//
	*******************************************/
    int CTR_main(unsigned char* presult_img, int downsampwidth, int downsampheight,  int params[]);

	//�����˶�����
	/*************************************************
	Function: 	    	// Diaphragm_detect
	Description: 		// �˺����������������߽������׵���ߵ�ƽ�������ж����ҷ�Ұ
	Input:        	// presult_imgΪ����ķ�Ұͼ���ַ��width��heightΪ����ͼ��Ŀ�ߣ�
	Output: 		// pdiaph_Line_imgΪ���ͼ���ַ��ͼ���а��������˶��㣬���������߽�����ֵֵΪ128���Ҳ�ֵΪ255
	Others: 		//
	*******************************************/
	int Diaphragm_detect(unsigned char* presult_img, int downsampwidth, int downsampheight, int params[],
		unsigned short* pdiaph_Line_imgorg); 

	//���ҷ�ʶ����
	/*************************************************
	Function: 	    	// Lung_Areacalculate
	Description: 		// �˺����������������߽������׵���ߵ�ƽ�������ж����ҷ�
	Input:        	// presult_imgΪ����ķ�Ұͼ���ַ��width��heightΪ����ͼ��Ŀ�ߣ�
	Output: 		// pleftrightlung_imgmask���ʶ������ҷΣ��ҷ�ֵ1�����ֵ2�� params[27]�������;params[28]�Ҳ����� params[29]ֵΪ1���Ϊ�ҷΣ��Ҳ�Ϊ��Σ�ֵΪ2���Ϊ��Σ��Ҳ�Ϊ�ҷ�
	Others: 		// 
	*******************************************/
	int Lung_Areacalculate(unsigned char* presult_img, int width, int height, unsigned short* pleftrightlung_imgmaskorg, int params[]);

	/*************************************************
	Function: 	    	// Erase_holl1
	Description: 		// �׶�����������������ͼ��ı�����ǰ����С������׶���ͨ���������
	matInputimg:           ����ͼ��
	ratiovalue��           ��������
	Others: 		// ����ֵΪ1Ϊ�������гɹ���-1Ϊʧ��
	*************************************************/
	int Erase_holl(Matrix<unsigned short>& matInputimg, double ratiovalue);

	/*************************************************
	Function: 	    	// Eliminateholes
	Description: 		// �׶�����ͨ������Ż�������ȥ����ɢ�׶���ͨ�򣬵��ڱ����µ���ͨ������
	Input:        	// matInputimg:����ͼ��ratiovalueΪ�����ı���
	Output: 		// matOutputimgΪ���ͼ��
	Others: 		// �˺���������inputimg������ں���ִ�н�����ȫ����,����ֵΪ1Ϊ�������гɹ���-1Ϊʧ��
	*************************************************/
	int Eliminateholes(Matrix<unsigned short>& matInputimg, Matrix<unsigned short>& matOutputimg, double ratiovalue);

	/*************************************************
	Function: 	    	// Conectchose
	Description: 		// ��ͨ��ѡ��
	Input:        	// matInputimg������ͼ����׵�ַ
	Output: 		// matLabelimg�������ͨ����ͼ ��ͨ��װ��С�����˳�����������labval����ͨ���С����labind����ͨ���ţ���TF��Ϊ����ͨ��Ϊ����ͨ
	Others: 		// �����ͨ�������������matInputimg����0�����������е�1����Ϊ0��
	*******************************************/
	void Conectchose(Matrix<unsigned short>& matInputimg, Matrix<unsigned short>& matLabelimg, vector<int>& labval, vector<int>& labind, bool TF);

	/*************************************************
	Function: 	    	// Erodeimg
	Description: 		// ��ʴ����
	Input:        	// inputimg������ͼ����׵�ַ��width��height�ֱ�Ϊͼ��Ŀ�ߣ�templateerode��ģ���׵�ַ��
	templateerodewidth��templateerodeheightΪģ����
	Output: 		// outputimg����ʴ���ͼ��
	Others: 		// �߽���ø������ط�ʽ
	*************************************************/
	void Erodeimg(Matrix<unsigned short>& matInputimg, Matrix<unsigned short>& templatedilate,
		Matrix<unsigned short>& matOutputimg);

	/*************************************************
	Function: 	    	// Dilateimg
	Description: 		// ��������
	Input:        	// inputimg������ͼ����׵�ַ��width��height�ֱ�Ϊͼ��Ŀ�ߣ�templateerode��ģ���׵�ַ��
	templateerodewidth��templateerodeheightΪģ����
	Output: 		// outputimg�����ͺ��ͼ��
	Others: 		// �߽���ø������ط�ʽ
	*************************************************/
	void Dilateimg(Matrix<unsigned short>& matInputimg, Matrix<unsigned short>& templatedilate,
		Matrix<unsigned short>& matOutputimg);  //

	/*************************************************
	Function: 	    	// Lungmask_clean
	Description: 		//��Ұ�������ͨ����
	Input:        	// presult_imgΪ����ķ�Ұͼ���ַ��width��heightΪ����ͼ��Ŀ�ߣ�
	Output: 		// paramsΪ�����������飬���庬����ӿڶ���
	Others: 		//
	*******************************************/
	void Lungmask_clean(cv::Mat& result_img);

private:

	//���رȺ��������ԭʼͼ��ӳ��
	/*************************************************
	Function: 	    	// MaptoOrg_CTR
	Description: 		// ���ر������������ӳ�䵽ԭʼͼ���е�����λ��
	Input:        	// params�������imgwidth��imgheightΪԭʼͼ�Ŀ�ߣ�downsampwidth��downsampheightΪ�²���ͼ����
	Output: 		// paramsӳ�䵽��Ӧ����Ľӿ�λ��
	Others: 		// ��
	*******************************************/
	void MaptoOrg_CTR(int imgwidth, int imgheight, int downsampwidth, int downsampheight, int params[]);

	//�������߽����ߵ�ԭʼͼ��ӳ��
	/*************************************************
	Function: 	    	// MaptoOrg_Diaph
	Description: 		// �����߽�����ӳ�䵽ԭʼͼ
	Input:        	// pdiaph_Line_imgΪ�²���ͼ�еĺ����߽����ߣ������������ֵΪ128���Ҳ���������ֵΪ255��imgwidth��imgheightΪԭʼͼ�Ŀ�ߣ�downsampwidth��downsampheightΪ�²���ͼ����
	Output: 		// pdiaph_Line_imgorgΪӳ�䵽ԭʼͼ��ĺ����߽�ͼ���������ֵΪ128���Ҳ�����ֵΪ255
	Others: 		// ��������ͨ���� + DDAֱ�߷�
	*******************************************/
	void MaptoOrg_Diaph(int imgwidth, int imgheight, int downsampwidth, int downsampheight, unsigned char* pdiaph_Line_img, unsigned short* pdiaph_Line_imgorg);

	//���ҷ�Ұʶ���ԭʼͼ��ӳ��
	/*************************************************
	Function: 	    	// MaptoOrg_Areacal
	Description: 		// ��Ұ���ӳ�䣬���²�����С���ӳ�䵽ԭͼ��С
	Input:        	// pleftrightlung_imgmask������²������ҷ�Ұͼ��imgwidth��imgheightΪԭʼͼ�Ŀ�ߣ�downsampwidth��downsampheightΪ�²���ͼ����
	Output: 		// pleftrightlung_imgmaskorg��Ϊԭʼͼ������ҷ�Ұʶ��params[27]params[28]���Ҳ�ԭʼͼ���Ұ����ļ���
	Others: 		// ����ļ���ͨ�����ݿ�߷Ŵ�ı����������
	*******************************************/
	void MaptoOrg_Areacal(int imgwidth, int imgheight, int downsampwidth, int downsampheight, int params[], unsigned char* pleftrightlung_imgmask, unsigned short* pleftrightlung_imgmaskorg);//

	/*************************************************
	Function: 	    	// LeftRightSegimg
	Description: 		// �˺���ʵ���������ܣ�һ���ݷ�Ұ�������Ұͼ�ָ�Ϊ���Ұͼ���ҷ�Ұͼ��2,�������Ұ�߽���Ҳ��Ұ�߽�
	Input:        	// matmasklungΪ����ķ�Ұ��ģͼ��Ҳ����Ϊ���ͼ��
	Output: 		// matLeftChestΪ���������Ұͼ��matRightChestΪ������Ҳ��Ұͼ��ֻ���б߽�������xvec_left���ĸ�����Ϊ���Ҳ��Ұ�����ı߽�����
	Others: 		// �˺���matmasklung�������������Ϊʹ�������С���б����ҷε�������飬������Ϊ�ҷ���Ϊ1�����С��Ϊ�����Ϊ2
	*******************************************/
	int LeftRightSegimg(Matrix<unsigned short>& matmasklung, Matrix<unsigned short>& matLeftChest, Matrix<unsigned short>& matRightChest,
		vector<int>& xvec_left, vector<int>& yvec_left, vector<int>& xvec_right, vector<int>& yvec_right);

	/*************************************************
	Function: 	    	// LeftRightsideimg
	Description: 		// �˺�������Ұͼ�ָ�Ϊ����Ұͼ���Ҳ��Ұͼ
	Input:        	// matmasklungΪ����ķ�Ұͼ��
	Output: 		// matleftlungmask_copyΪ���������Ұͼ��matrightlungmask_copyΪ������Ҳ��Ұͼ��
	Others: 		// ֻ���������Ҳ��Ұͼ��
	*******************************************/
	int LeftRightsideimg(Matrix<unsigned short>& matmasklung, Matrix<unsigned short>& matleftlungmask_copy, Matrix<unsigned short>& matrightlungmask_copy);

	/*************************************************
	Function: 	    	// DDALine11
	Description: 		// ֱ�߻��߷�
	Input:        	// neblabelimgposΪ������Ҫ���ߵ�ͼ��width��heightΪͼ��Ŀ�ߣ�startx��startyΪ��ʼ���꣬endx��endyΪ��ֹ���ꡣpixelvalueΪ���ߵ�����ֵ
	Output: 		// neblabelimgpos���������֮����߶Σ��߶��е�����ֵΪpixelvalue
	Others: 		// ��
	*******************************************/
	void DDALine11(unsigned short* neblabelimgpos, int startx, int starty, int endx, int endy, int pixelvalue, int width, int height);

	/*************************************************
	Function: 	    	// RibDiaphSeg_left
	Description: 		// ����Ұ��Ե����Ĥ�ķָ�
	Input:        	// matLeftChest����������Ұͼ����׵�ַ��xvec_left��yvec_leftΪ����Ұ�߽�����ꣻ
	Output: 		// xvec_leftrib��yvec_leftrib�ֱ����Ҳ���Ե�����ꣻxvec_leftdiaph��yvec_leftdiaph�ֱ�Ϊ�Ҳ������Ĥ�߽�����
	Others: 		// ��Ұ�߽�Ķ����͵ײ�ʹ����ͨ������˶Ͽ�
	*******************************************/
	int RibDiaphSeg_left(Matrix<unsigned short>& matLeftChest,
		vector<int>& xvec_left, vector<int>& yvec_left,
		vector<int>& xvec_leftrib, vector<int>& yvec_leftrib, vector<int>& xvec_leftdiaph, vector<int>& yvec_leftdiaph);

	/*************************************************
	Function: 	    	// RibDiaphSeg_right
	Description: 		// �Ҳ��Ұ��Ե����Ĥ�ķָ�
	Input:        	// matRightChest�������Ҳ��Ұͼ����׵�ַ��xvec_right��yvec_rightΪ�Ҳ��Ұ�߽�����ꣻ
	Output: 		// xvec_rightrib��yvec_rightrib�ֱ����Ҳ���Ե�����ꣻxvec_rightdiaph��yvec_rightdiaph�ֱ�Ϊ�Ҳ������Ĥ�߽�����
	Others: 		// ��Ұ�߽�Ķ����͵ײ�ʹ����ͨ������˶Ͽ�
	*******************************************/
	int RibDiaphSeg_right(Matrix<unsigned short>& matRightChest,
		vector<int>& xvec_right, vector<int>& yvec_right,
		vector<int>& xvec_rightrib, vector<int>& yvec_rightrib, vector<int>& xvec_rightdiaph, vector<int>& yvec_rightdiaph);

	/*************************************************
	Function: 	    	// GetorderDiaph��
	Description: 		// ��ȡ��ͨ����������㡣
	Input:        	// matLableimg������ͼ����׵�ַ��Xdiaph_LeftorderΪ���������Ľ�ɫ��
	xvec_leftdiaph��yvec_leftdiaphΪ�ֱ�Ϊ�����������࣬Xdiaph_Leftorder��Ydiaph_LeftorderΪ����������׵�ǡ�
	Output: 		// outputimg�����ͺ��ͼ��
	Others: 		// �߽���ø������ط�ʽ��
	*************************************************/
	void GetorderDiaph(Matrix<unsigned short>& matLableimg, vector<int>& xvec_leftdiaph, vector<int>& yvec_leftdiaph,
		vector<int>& Xdiaph_Leftorder, vector<int>& Ydiaph_Leftorder);//  

	/*************************************************
	Function: 	    	// Search_Conectpoint
	Description: 		// ��ͨ������к�����������ͨ�����ʼ������ʼ�㿪ʼ������ͨ�������
	Input:        	// matConectlabel������ͼ����ͨ��ͼ��startY��startXΪ�ֱ�Ϊ��ʼ���Y��X����
	Output: 		// orderY��orderX�ֱ�Ϊ����ʼ����ֹ�����ͨ���Y��X����
	Others: 		// �˺���������������ͨ�������˳��
	*************************************************/
	void Search_Conectpoint(Matrix<unsigned short>& matConectlabel, int startX, int startY,
		vector<int>& orderX, vector<int>& orderY);

	/*************************************************
	 Function: 	    	// GetHeartDiaphPoint_leftlung��GetHeartDiaphPoint_rightlung
	 Description: 		// �˺������ڼ���ͻ�ȡ���ݸ��߽�Ĺյ�λ�ã�����ȡ���ߵ�б��
	 Input:        	// Ydiaph_order��Xdiaph_order�ֱ�Ϊ���ݸ��߽����꣬maxdisindΪ�յ��λ��
	 Output: 		// nvect_ratioΪ���ݸ�Ĥ�߽��б��
	 Others: 		// �˺���������������ͨ�������˳��
	 *************************************************/
	void GetHeartDiaphPoint_leftlung(vector<int>& Xdiaph_order, vector<int>& Ydiaph_order,
		int& maxdisind, vector<float>& nvect_ratio, int leftmaxdis_Y, int leftrightparm);

	void GetHeartDiaphPoint_rightlung(vector<int>& Xdiaph_order, vector<int>& Ydiaph_order,
		int& maxdisind, vector<float>& nvect_ratio, int leftrightparm);
	/*************************************************
	Function: 	    	// Line_Smooth
	Description: 		// �˺������������߽���ƽ��
	Input:        	// Xdiaph_order��Ydiaph_order�ֱ�Ϊ�������ߵ����꣨����ʼ����ֹ����detapix��Ϊƽ�����
	Output: 		// nXXmeanvect��nYYmeanvect�ֱ�Ϊ����ƽ���˲�ƽ�������ͨ���X��Y����
	Others: 		// �˺������������߽�����ƽ��
	*************************************************/
	void Line_Smooth(vector<int>& Xdiaph_order, vector<int>& Ydiaph_order, int detapix,
		vector<float>& nXXmeanvect, vector<float>& nYYmeanvect);

	/*************************************************
	Function: 	    	// Line_Ratio
	Description: 		// ����������ÿһ���б��
	Input:        	// nXXmeanvect��nYYmeanvect�ֱ�Ϊ�������ߵ����꣨����ʼ����ֹ����detapix������б�ʵļ��
	Output: 		// nvect_ratio��Ϊ������ÿһ���б��
	Others: 		// �˺�����������б��
	*************************************************/
	void Line_Ratio(vector<float>& nXXmeanvect, vector<float>& nYYmeanvect, int detapix, vector<float>& nvect_ratio);

	/*************************************************
	Function: 	    	// ARC_Distance
	Description: 		// ����Ĺ��ξ���
	Input:        	// nXXmeanvect��nYYmeanvect�ֱ�Ϊ�������ߵ����꣨����ʼ����ֹ��
	Output: 		// Vertical_disvector��Ϊ������ÿһ��Ĺ��ξ���
	Others: 		// �˺����������ߵĹ��ξ���
	*************************************************/
	void ARC_Distance(vector<float>& nXXmeanvect, vector<float>& nYYmeanvect, vector<double>& Vertical_disvector, int posnegvalue);

	/*************************************************
	Function: 	    	// Find_hartdiaphind_smallheart
	Description: 		// �����������Ұ����Ч��������Һ���������������Ե��ļ��
	Input:        	// nXXsmall_vect��nYYsmall_vect�ֱ�Ϊ�����Һ���߽��������ߵ����꣨����ʼ����ֹ����nvect_ratio�����ߵ�б�ʣ�maxdisindΪ�յ�λ��
	Output: 		// hart_Leftind��diaph_Leftind�ֱ�Ϊ�Ҳ�������Ч���λ�úͺ�����߽���Ч���λ��
	Others: 		// ���ڻ�ȡ�ҷ���Ч���λ����Ϣ
	*************************************************/
	void Find_hartdiaphind_smallheart(vector<int>& nXXsmall_vect, vector<int>& nYYsmall_vect, vector<float>& nvect_ratio,
		int& maxdisind, int& hart_Leftind, int& diaph_Leftind, int leftrightparm);

	/*************************************************
	Function: 	    	// Find_hartdiaphind_largeheart
	Description: 		// �����������Ұ����Ч������������������������Ե��ļ��
	Input:        	// nXXlarge_vect��nYYlarge_vect�ֱ�Ϊ���������߽��������ߵ����꣨����ʼ����ֹ����nvect_ratio�����ߵ�б��;maxdisindΪ�յ�λ��
	Output: 		// hart_Leftind��diaph_Leftind�ֱ�Ϊ���������Ч���λ�úͺ�����߽���Ч���λ��
	Others: 		// ���ڻ�ȡ�����Ч���λ����Ϣ
	*************************************************/
	void Find_hartdiaphind_largeheart(vector<int>& nXXlarge_vect, vector<int>& nYYlarge_vect, vector<float>& nvect_ratio,
		int& maxdisind, int& hart_Rightind, int& diaph_Rightind, int leftrightparm);

	/*************************************************
	 Function: 	    	// LeftRight_dist
	 Description: 		// �����м���������������׵�����ߵ�ƽ�����룬ͨ���˺������ж����ҷ�
	 Input:        	// xribdataleft��Xdiaph_Leftorder��������Ե�ͺ��ݸ�Ĥ�߽�����㣬Midseg_XΪ������λ��
	 Output: 		// ������LeftdisX���Ҳ�RightdisX����
	 Others: 		// �˺����������ߵĹ��ξ���
	 *************************************************/
	void LeftRight_dist(int& LeftdisX, int& RightdisX, int Midseg_X, Matrix<unsigned short>& matLeftdis, Matrix<unsigned short>& matRightdis,
		vector<int>& xribdataleft, vector<int>& yribdataleft, vector<int>& xribdataright, vector<int>& yribdataright,
		vector<int>& Xdiaph_Leftorder, vector<int>& Ydiaph_Leftorder, vector<int>& Xdiaph_Rightorder, vector<int>& Ydiaph_Rightorder);

};




