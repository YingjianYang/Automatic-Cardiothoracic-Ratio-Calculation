#include "Img_Process.h"


void Img_Process::Proc_main(unsigned short* pArrimg, int imgwidth, int imgheight, unsigned char* pmap255)
{

    int widthheight = imgwidth * imgheight;
    double* pinputimg = new(std::nothrow) double[widthheight]();
    for (int i = 0; i < widthheight; i++)
    {
        pinputimg[i] = (double)pArrimg[i];
    }

	//高斯滤波处理
	double* matGauresult=new double[imgwidth*imgheight](); //定义高斯滤波的mat
	int gaswid = 7;//高斯模板宽度，模板宽度必须为奇数
	double tho = 1;//高斯函数方差值
	double* matGastemplate= new double[gaswid*gaswid]();//构建高斯模板函数空间
	Gaussiantemplatefunc(matGastemplate,gaswid, tho);//高斯模板的获取
	//高斯滤波的模板运算计算
	Templatefilter(pinputimg, imgwidth,imgheight,matGastemplate, gaswid, gaswid, matGauresult);

    double* pSublogimag = new(std::nothrow) double[widthheight]();
    double* pL3 = new(std::nothrow) double[widthheight]();
    double* pLogimg = new(std::nothrow) double[widthheight]();

	//对数变换  
    AdaptIogImage(matGauresult, pSublogimag, pL3, pLogimg, imgwidth, imgheight);//对数变换
	//直方图均衡化
    Hist_Equaliation(pLogimg, pLogimg, 4096, imgwidth, imgheight);
	//对比度增强归一化
	Imadjust_func(pLogimg, 0.05, 0.95, pLogimg, imgwidth, imgheight);//
	//映射到255
    Mapto255(pLogimg, pmap255, imgwidth, imgheight);  //  

    //反白
    for (int i = 0; i < widthheight; i++)
    {
        pmap255[i] = 255 - pmap255[i];
    }
	delete[] matGauresult;
	matGauresult = nullptr;
	delete[] matGastemplate;
	matGastemplate = nullptr;
	delete[] pSublogimag;
	pSublogimag = nullptr;
	delete[] pL3;
	pL3 = nullptr;
	delete[] pLogimg;
	pLogimg = nullptr;
	delete[] pinputimg;
	pinputimg = nullptr;
}


void Img_Process::Normalize_func(double* pInputimg, double* pOutputimg, int width, int height)
{
    //归一化操作
    int sampwidheigh = width * height;
    //求最大值和最小值
    double maxL3 = 0;
    maxL3 = pInputimg[0];
    double minL3 = 0;
    minL3 = pInputimg[0];

    //求数组最大值
    for (int i = 1; i < sampwidheigh; ++i)
    {
        if (pInputimg[i] > maxL3)
        {
            maxL3 = pInputimg[i];
        }
    }
    //求数组最小值
    for (int i = 1; i < sampwidheigh; ++i)
    {
        if (pInputimg[i] < minL3)
        {
            minL3 = pInputimg[i];
        }
    }

    //归一化操作
    if (minL3 >= maxL3)
    {
        for (int m = 0; m < sampwidheigh; m++)
        {
            pOutputimg[m] = pInputimg[m];
        }
    }
    else
    {
        double deta = 1 / (maxL3 - minL3);
        for (int n = 0; n < sampwidheigh; n++)
        {
            pOutputimg[n] = (pInputimg[n] - minL3) * deta;
        }
    }
}

void Img_Process::AdaptIogImage(double* pDownsample, double* pSublogimag, double* pL3, double* pLogimg,int width ,int height)
{
    int imglength = width * height;
    double bias = 0.850;
    double L1 = 0;
    double L2 = 0;

    double biasb = log(bias) / log(0.5);//计算自适偏置参数
    for (int i = 0; i < imglength; i++)
    {
        if (pDownsample[i] <= 1)
        {
            pSublogimag[i] = 0;
        }
        else
        {
            pSublogimag[i] = log(pDownsample[i]);
        }
    }
    //求对数变换的最大值
    double maxlog = *max_element(pSublogimag, pSublogimag + imglength);//找最大值

    double maxlogdivis = 1. / double(maxlog);

    for (int j = 0; j < imglength; j++)
    {
        L1 = log(pDownsample[j] + 1) / log10(maxlog + 1);
        L2 = log(2 + (8 * (pow((pDownsample[j] * maxlogdivis), biasb))));
        pL3[j] = L1 / L2;
    }

    ////对数归一化处理
    //double minvalue = *min_element(pL3, pL3 + imglength);//找最小值
    ////对数归一化处理
    //double maxnvalue = *max_element(pL3, pL3 + imglength);//找最大值
    ////求数组中除了零之外的最小值
    //double tempdl = pL3[0];
    //for (int i = 1; i < imglength; ++i)
    //{
    //    double temp = pL3[i];
    //    if ((temp > minvalue) && (temp< tempdl))
    //    {
    //        tempdl = temp;
    //    }
    //}
    //if (maxnvalue> tempdl)
    //{
    //    double deta = 1 / (maxnvalue - tempdl);
    //    for (int i = 0; i < imglength; i++)
    //    {
    //        if (pL3[i] < tempdl)
    //        {
    //            pL3[i] = tempdl;
    //        }
    //        pLogimg[i] = (pL3[i] - tempdl) * deta;
    //    }
    //}

    Normalize_func(pL3,pLogimg,width,height);

}

void Img_Process::Mapto255(double* pInputimg, unsigned char* pOutputimg, int width, int height)
{
    int imglength = width * height;
    double minvalue = *min_element(pInputimg, pInputimg + imglength);
    double maxvalue = *max_element(pInputimg, pInputimg + imglength);

    for (int i = 0; i < imglength; i++)
    {
       double midtempvalue = (pInputimg[i] - minvalue) / (maxvalue - minvalue);
       double resultdl = 255 * midtempvalue;
       pOutputimg[i] = unsigned char(resultdl + 0.5);
    }

}

void Img_Process::Imadjust_func(double* pInputimg, double downlimit, double uplimit, double* pOutputimg,int width, int height)//对比度调节
{
    int imgwidheigh = width*height;

    double minvalue = pInputimg[0];
    double maxvalue = pInputimg[0];
    //求数组最大值和最小值
    for (int i = 1; i < imgwidheigh; i++)
    {
        if (pInputimg[i] > maxvalue)
        {
            maxvalue = pInputimg[i];//获取数组最大值.
        }
        else if (pInputimg[i] < minvalue)
        {
            minvalue = pInputimg[i];//获取数组最小值
        }
    }
    if (maxvalue > minvalue)
    {
        double basedata = uplimit - downlimit;
        double detavalue = maxvalue - minvalue;
        for (int i = 0; i != imgwidheigh; i++)
        {
            double tempvalue = (pInputimg[i] - minvalue) / detavalue;//归一化计算
            //imadjust对比度增强判定
            if (tempvalue < downlimit)
            {
                pOutputimg[i] = 1e-9;
            }
            else if (tempvalue > uplimit)
            {
                pOutputimg[i] = 1;
            }
            else
            {
                pOutputimg[i] = (tempvalue - downlimit) / basedata;
            }
        }
    }
}

/*图像下采样，下采样倍率为ratio, 方式为取ratio*ratio区域中像素均值作为此区域的代表像素*/
void Img_Process::DownSample(double* pGauresult, double* pdownsample, int ratio,int width,int height,
    int sample_wid,int sample_heigh)
{
    int ratiosqu = ratio * ratio;
    double divratio = 1.0 / double(ratiosqu);

    int ratiowidth = width * ratio;
    for (int i = 0, step1 = 0, step2 = 0; i < sample_heigh; i++, step1 += sample_wid, step2 += ratiowidth)
    {
        double* gausrow = pGauresult + step2;//定义原始第row行
        double* samprow = pdownsample + step1;//定义采样底row行

        for (int j = 0; j < sample_wid; j++)
        {
            double* gauspoistion = gausrow + j * ratio;//定义原始第row行，第j*ratio列 
            double sumtemp = 0;//区块的累加变量

            /*此for循环用于下采样，将radia区域所围的数据求和取平均*/
            for (int m = 0; m < ratio; m++)
            {
                for (int n = 0; n < ratio; n++)
                {
                    sumtemp = sumtemp + gauspoistion[m*width + n];
                }
            }
            samprow[j] = double(sumtemp) * divratio;
        }
    }
}


void Img_Process::Hist_Equaliation(double* pInputimg, double* pOutputimg, int levelnumber,int width,int height)
{
    int widthheight = width * height;
    double maxorgvalue = *max_element(pInputimg, pInputimg + widthheight);
    double minorgvalue = *min_element(pInputimg, pInputimg + widthheight);
    //将原始数据映射到0至255，申请设置256映射空间
    int* imgmaplevel = new int[widthheight]();

    //遍历每个像素，计算映射值
    if (maxorgvalue > minorgvalue)//只有最大值大于最小值才执行
    {
        //imgmaplevelnumber中最大值为levelnumber，最小值为0，共levelnumber个(浮点型数组)
        for (int i = 0; i < widthheight; i++)//遍历每个像素，映射到0至levelnumber-1之间
        {
            imgmaplevel[i] = int((levelnumber - 1) * (pInputimg[i] - minorgvalue) / (maxorgvalue - minorgvalue) + 0.5f);//映射的最大值为levelnumber-1
        }

        //申请设置levelnumber个计数数组空间用于像素值的计数
        double* imh = new double[levelnumber]();

        //遍历每个像素，对每个像素进行判定，置入计数数组的累加。
        for (int i = 0; i < widthheight; i++)
        {
            //valueindex在0至levelnumber-1这levelnumber个数之间,加0.5是为了整型强转浮点型
            int valueindex = imgmaplevel[i];
            if (valueindex <= 0)
            {
                valueindex = 0;
                imh[valueindex] = imh[valueindex] + 1;//累加器索引值为像素值，累加数据。
            }
            else if (valueindex >= levelnumber - 1)
            {
                valueindex = levelnumber - 1;
                imh[valueindex] = imh[valueindex] + 1;//累加器索引值为像素值，累加数据1。
            }
            else
            {
                imh[valueindex] = imh[valueindex] + 1;//累加器索引值为像素值，累加数据1。
            }
        }

        //概率计算，每个像素值的概率，概率空间的申请设置
        double* imhprob = new double[levelnumber]();
        for (int i = 0; i < levelnumber; i++)
        {
            imhprob[i] = imh[i] / (double)widthheight;//概率计算，每个像素数值对应一个概率值
        }

        double* equals = new double[levelnumber]();//映射到新的值空间
        for (int i = 0; i < levelnumber; i++)
        {
            double sumimhprob = 0;
            for (int j = 0; j <= i; j++)
            {
                sumimhprob = sumimhprob + imhprob[j];
            }
            equals[i] = (levelnumber - 1) * sumimhprob;
        }

        for (int i = 0; i < widthheight; i++)
        {
            int vlaue = imgmaplevel[i];//vlaue的最大值为levelnumber-1，最小值为0
            pOutputimg[i] = equals[vlaue] / (levelnumber - 1);//归一化，最大值为levelnumber-1，对应为1
        }

        delete[] imh;
        imh = nullptr;
        delete[] imhprob;
        imhprob = nullptr;
        delete[] equals;
        equals = nullptr;
    }
    else
    {
        ;
    }
    delete[] imgmaplevel;
    imgmaplevel = nullptr;
}



void Img_Process::Gaussiantemplatefunc(double* matGastemplate,int wide, double tho)
{
	int gaswidup = wide / 2;//向下取整,作为上限
	int gaswiddown = -gaswidup;//取负号作为下限

	//计算临时值
	double elementvalue1 = 1.0 / (2.0 * tho * tho * 3.141592653);
	double elementvalue2 = 1.0 / (2.0 * tho * tho);

	for (int i = gaswiddown; i <= gaswidup; i++)
	{
		double* gaustemplrow = matGastemplate  + (i + gaswidup) * wide;
		for (int j = gaswiddown; j <= gaswidup; j++)
		{
			double* gaustemplrcol = gaustemplrow + j + gaswidup;
			*(gaustemplrcol) = elementvalue1 * exp(-(i * i + j * j) * elementvalue2);//高斯公式计算
		}
	}

	double sumtemp = 0;
	sumtemp = accumulate(matGastemplate , matGastemplate  + wide* wide, 0.0f);
	for (int i = 0; i < wide * wide; i++)
	{
		matGastemplate [i] = matGastemplate [i] / sumtemp;//归一化，确保生成的模板数组总和为1
	}
}


void Img_Process::Templatefilter(double* matInputimg,int width,int height, double* matTemplate,int matTemplatewidth,int matTemplateheight, double* matOutputimg)
{
	int addwidth = matTemplatewidth / 2;
	int addheight = matTemplateheight / 2;
	int widthheight = width * height;
	int tempwid = width + addwidth * 2;
	int tempheight = height + addheight * 2;
	int tempheigthwid = tempwid * tempheight;

	double* addtempimg = new double[tempwid*tempheight]();//创建扩大后的图像空间，将输入图像填充金扩大后的图像空间

	//填充新空间
	//中间部分填充
	for (int i = addheight, step1 = 0; i < addheight + height; i++, step1 += width)
	{
		double* tempcalulaterow = addtempimg  + i * tempwid;  //输出图像第i行的首地址               
		double* inputrow = matInputimg  + step1;  //输出图像第i行的首地址    
		for (int j = addwidth, step2 = 0; j < addwidth + width; j++, step2 += 1)
		{
			tempcalulaterow[j] = inputrow[step2];
		}
	}

	//左右侧边界填充
	for (int i = addheight, step1 = 0; i < addheight + height; ++i, step1 += width) //遍历新空间
	{
		double* addtempimgrow = addtempimg  + i * tempwid;  //输出图像第i行的首地址   
		double leftvalue = matInputimg [step1];
		double rightvalue = matInputimg [step1 - 1 + width];
		//左侧边界填充
		for (int j = 0; j < addwidth; ++j)
		{
			addtempimgrow[j] = leftvalue;//左侧边界填充的为与距离最近的数相同。
		}
		//右侧边界填充
		for (int k = width + addwidth; k < tempwid; ++k)
		{
			addtempimgrow[k] = rightvalue;//右侧边界填充的为与距离最近的数相同。
		}
	}

	//上侧边界填充
	for (int i = 0; i < addheight; i++)//上侧行数遍历
	{
		double* addtempimgrow = addtempimg  + i * tempwid;  //输出图像第i行的首地址  
		double* addheightimgrow = addtempimg  + addheight * tempwid;  //输出图像第addheight行的首地址 

		for (int k = 0; k < tempwid; k++)
		{
			addtempimgrow[k] = addheightimgrow[k];
		}
	}
	//下侧行数填充
	for (int i = height + addheight; i < tempheight; i++)
	{
		double* addtempimgrow = addtempimg  + i * tempwid;  //输出图像第i行的首地址 
		double* addheightimgrow = addtempimg  + (height + addheight - 1) * tempwid;  //输出图像第i行的首地址 

		for (int k = 0; k < tempwid; k++)
		{
			addtempimgrow[k] = addheightimgrow[k];
		}
	}

	//模板遍历
	for (int i = addheight, step1 = addheight * tempwid, step2 = 0; i < addheight + height; ++i, step1 += tempwid, step2 += width) //遍历模板滤波
	{
		double* addtempimgrow = addtempimg  + step1; //扩展图像第i行的首地址
		double* outputimgrow = matOutputimg  + step2;  //输出图像第i行的首地址          
		for (int j = addwidth, k = 0; j < addwidth + width; ++j, k += 1)
		{
			double sumtempvalue = 0;
			double* addtempimgpos = addtempimgrow + j; //扩展图像第i行第j列坐标中心元素的地址

			//在模板大小的局部区域内，循环遍历累加
			for (int m = -addheight; m <= addheight; m++)
			{
				double* starinputpos = addtempimgpos + m * tempwid;
				double* matTemplaterow = matTemplate  + (m + addheight) * matTemplatewidth;
				for (int n = -addwidth; n <= addwidth; n++)
				{
					sumtempvalue = sumtempvalue + starinputpos[n] * matTemplaterow[n + addwidth];
				}
			}
			//获取累加平均值，由于模板加权值为1，此操作无需平均分
			outputimgrow[k] = sumtempvalue;//输出图像第i行第j列中心坐标元素的地址
		}
	}
	delete[] addtempimg;
	addtempimg = nullptr;
}

//获取特定格式的文件名  
void Img_Process::GetAllFormatFiles(string path, vector<string>& files, string format)
{
	//文件句柄    
	long long  hFile = 0;
	//文件信息    
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*" + format).c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
				{
					files.push_back(p.assign(path).append("\\").append(fileinfo.name));
					GetAllFormatFiles(p.assign(path).append("\\").append(fileinfo.name), files, format);
				}
			}
			else
			{
				files.push_back(p.assign(fileinfo.name));  //将文件路径保存，也可以只保存文件名:  p.assign(path).append("\\").append(fileinfo.name)  
			}
		} while (_findnext(hFile, &fileinfo) == 0);

		/*if ((hFile = _findfirst("G:\\Yangyingjian\\MatlabToC\\LUNG_ventilation_data\\original\\*.dcm", &fileinfo)) == -1L)
			printf("没有找到匹配的项目\n");
		else
		{
			printf("%s\n", fileinfo.name);
			while (_findnext(hFile, &fileinfo) == 0)
				printf("%s\n", fileinfo.name);
			_findclose(hFile);
		}*/

		_findclose(hFile);
	}
}

void Img_Process::Conectchose(Matrix<unsigned short>& matInputimg, Matrix<unsigned short>& matLabelimg, vector<int>& labval, vector<int>& labind, bool TF)
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
				if (deter !=0 )//只要像素值为1，即纳入考虑范畴
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
								if (*nebinputimgpos !=0)
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
				if (deter != 0)//只要像素值为1，即纳入考虑范畴
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
								if (*nebinputimgpos != 0)
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

void Img_Process::Isinverimg(unsigned short* pArrimg, unsigned char* pmap255, int imgwidth, int imgheight)
{
    //图像左上角
    double leftupsummeanvalue = 0;
    int leftupsumnumber = 1;
    for (int i = 100, step1 = 100 * imgwidth; i < 200; i++, step1 += imgwidth)//图像高
    {
        for (int j = 100; j < 200; j++)
        {
            leftupsummeanvalue += pArrimg[step1 + j];
            leftupsumnumber++;
        }
    }

    //图像右上角
    int rightupsummeanvalue = 0;
    int rightupsumnumber = 1;
    for (int i = 100, step1 = 100 * imgwidth; i < 200; i++, step1 += imgwidth)//图像高
    {
        for (int j = imgwidth - 100; j < imgwidth; j++)
        {
            rightupsummeanvalue += pArrimg[step1 + j];
            rightupsumnumber++;
        }
    }

    //图像正中央
    double midupsummeanvalue = 0;
    int midupsumnumber = 1;
    for (int i = imgheight / 2, step1 = 100 * imgwidth; i < imgheight / 2 + 200; i++)//图像高
    {
        for (int j = imgwidth / 2 - 100; j < imgwidth / 2 + 100; j++)
        {
            midupsummeanvalue += pArrimg[step1 + j];
            midupsumnumber++;
        }
    }
    midupsummeanvalue = midupsummeanvalue / midupsumnumber;//中央平均值	
    double meanupleft = (leftupsummeanvalue + rightupsummeanvalue) / (leftupsumnumber + rightupsumnumber);//边角平均值

    double* pmapinput = new double[imgwidth * imgheight]();
    for (int i = 0; i < imgwidth * imgheight; i++)
    {
        pmapinput[i] = pArrimg[i];
    }

    //图像反白判断处理
    if (meanupleft > midupsummeanvalue)//如果图像的中央灰度值低，则是原始图像
    {
        //映射到255
        Mapto255(pmapinput, pmap255, imgwidth, imgheight);
        //反白
        for (int i = 0; i < imgwidth * imgheight; i++)
        {
            pmap255[i] = 255 - pmap255[i];
        }
    }
    else//否则则是反白图像，直接映射255
    {
        //只有映射算法
        Mapto255(pmapinput, pmap255, imgwidth, imgheight);
    }

    delete[] pmapinput;
    pmapinput = nullptr;
}
