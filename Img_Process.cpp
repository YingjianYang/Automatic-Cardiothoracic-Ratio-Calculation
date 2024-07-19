#include "Img_Process.h"


void Img_Process::Proc_main(unsigned short* pArrimg, int imgwidth, int imgheight, unsigned char* pmap255)
{

    int widthheight = imgwidth * imgheight;
    double* pinputimg = new(std::nothrow) double[widthheight]();
    for (int i = 0; i < widthheight; i++)
    {
        pinputimg[i] = (double)pArrimg[i];
    }

	//��˹�˲�����
	double* matGauresult=new double[imgwidth*imgheight](); //�����˹�˲���mat
	int gaswid = 7;//��˹ģ���ȣ�ģ���ȱ���Ϊ����
	double tho = 1;//��˹��������ֵ
	double* matGastemplate= new double[gaswid*gaswid]();//������˹ģ�庯���ռ�
	Gaussiantemplatefunc(matGastemplate,gaswid, tho);//��˹ģ��Ļ�ȡ
	//��˹�˲���ģ���������
	Templatefilter(pinputimg, imgwidth,imgheight,matGastemplate, gaswid, gaswid, matGauresult);

    double* pSublogimag = new(std::nothrow) double[widthheight]();
    double* pL3 = new(std::nothrow) double[widthheight]();
    double* pLogimg = new(std::nothrow) double[widthheight]();

	//�����任  
    AdaptIogImage(matGauresult, pSublogimag, pL3, pLogimg, imgwidth, imgheight);//�����任
	//ֱ��ͼ���⻯
    Hist_Equaliation(pLogimg, pLogimg, 4096, imgwidth, imgheight);
	//�Աȶ���ǿ��һ��
	Imadjust_func(pLogimg, 0.05, 0.95, pLogimg, imgwidth, imgheight);//
	//ӳ�䵽255
    Mapto255(pLogimg, pmap255, imgwidth, imgheight);  //  

    //����
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
    //��һ������
    int sampwidheigh = width * height;
    //�����ֵ����Сֵ
    double maxL3 = 0;
    maxL3 = pInputimg[0];
    double minL3 = 0;
    minL3 = pInputimg[0];

    //���������ֵ
    for (int i = 1; i < sampwidheigh; ++i)
    {
        if (pInputimg[i] > maxL3)
        {
            maxL3 = pInputimg[i];
        }
    }
    //��������Сֵ
    for (int i = 1; i < sampwidheigh; ++i)
    {
        if (pInputimg[i] < minL3)
        {
            minL3 = pInputimg[i];
        }
    }

    //��һ������
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

    double biasb = log(bias) / log(0.5);//��������ƫ�ò���
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
    //������任�����ֵ
    double maxlog = *max_element(pSublogimag, pSublogimag + imglength);//�����ֵ

    double maxlogdivis = 1. / double(maxlog);

    for (int j = 0; j < imglength; j++)
    {
        L1 = log(pDownsample[j] + 1) / log10(maxlog + 1);
        L2 = log(2 + (8 * (pow((pDownsample[j] * maxlogdivis), biasb))));
        pL3[j] = L1 / L2;
    }

    ////������һ������
    //double minvalue = *min_element(pL3, pL3 + imglength);//����Сֵ
    ////������һ������
    //double maxnvalue = *max_element(pL3, pL3 + imglength);//�����ֵ
    ////�������г�����֮�����Сֵ
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

void Img_Process::Imadjust_func(double* pInputimg, double downlimit, double uplimit, double* pOutputimg,int width, int height)//�Աȶȵ���
{
    int imgwidheigh = width*height;

    double minvalue = pInputimg[0];
    double maxvalue = pInputimg[0];
    //���������ֵ����Сֵ
    for (int i = 1; i < imgwidheigh; i++)
    {
        if (pInputimg[i] > maxvalue)
        {
            maxvalue = pInputimg[i];//��ȡ�������ֵ.
        }
        else if (pInputimg[i] < minvalue)
        {
            minvalue = pInputimg[i];//��ȡ������Сֵ
        }
    }
    if (maxvalue > minvalue)
    {
        double basedata = uplimit - downlimit;
        double detavalue = maxvalue - minvalue;
        for (int i = 0; i != imgwidheigh; i++)
        {
            double tempvalue = (pInputimg[i] - minvalue) / detavalue;//��һ������
            //imadjust�Աȶ���ǿ�ж�
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

/*ͼ���²������²�������Ϊratio, ��ʽΪȡratio*ratio���������ؾ�ֵ��Ϊ������Ĵ�������*/
void Img_Process::DownSample(double* pGauresult, double* pdownsample, int ratio,int width,int height,
    int sample_wid,int sample_heigh)
{
    int ratiosqu = ratio * ratio;
    double divratio = 1.0 / double(ratiosqu);

    int ratiowidth = width * ratio;
    for (int i = 0, step1 = 0, step2 = 0; i < sample_heigh; i++, step1 += sample_wid, step2 += ratiowidth)
    {
        double* gausrow = pGauresult + step2;//����ԭʼ��row��
        double* samprow = pdownsample + step1;//���������row��

        for (int j = 0; j < sample_wid; j++)
        {
            double* gauspoistion = gausrow + j * ratio;//����ԭʼ��row�У���j*ratio�� 
            double sumtemp = 0;//������ۼӱ���

            /*��forѭ�������²�������radia������Χ���������ȡƽ��*/
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
    //��ԭʼ����ӳ�䵽0��255����������256ӳ��ռ�
    int* imgmaplevel = new int[widthheight]();

    //����ÿ�����أ�����ӳ��ֵ
    if (maxorgvalue > minorgvalue)//ֻ�����ֵ������Сֵ��ִ��
    {
        //imgmaplevelnumber�����ֵΪlevelnumber����СֵΪ0����levelnumber��(����������)
        for (int i = 0; i < widthheight; i++)//����ÿ�����أ�ӳ�䵽0��levelnumber-1֮��
        {
            imgmaplevel[i] = int((levelnumber - 1) * (pInputimg[i] - minorgvalue) / (maxorgvalue - minorgvalue) + 0.5f);//ӳ������ֵΪlevelnumber-1
        }

        //��������levelnumber����������ռ���������ֵ�ļ���
        double* imh = new double[levelnumber]();

        //����ÿ�����أ���ÿ�����ؽ����ж����������������ۼӡ�
        for (int i = 0; i < widthheight; i++)
        {
            //valueindex��0��levelnumber-1��levelnumber����֮��,��0.5��Ϊ������ǿת������
            int valueindex = imgmaplevel[i];
            if (valueindex <= 0)
            {
                valueindex = 0;
                imh[valueindex] = imh[valueindex] + 1;//�ۼ�������ֵΪ����ֵ���ۼ����ݡ�
            }
            else if (valueindex >= levelnumber - 1)
            {
                valueindex = levelnumber - 1;
                imh[valueindex] = imh[valueindex] + 1;//�ۼ�������ֵΪ����ֵ���ۼ�����1��
            }
            else
            {
                imh[valueindex] = imh[valueindex] + 1;//�ۼ�������ֵΪ����ֵ���ۼ�����1��
            }
        }

        //���ʼ��㣬ÿ������ֵ�ĸ��ʣ����ʿռ����������
        double* imhprob = new double[levelnumber]();
        for (int i = 0; i < levelnumber; i++)
        {
            imhprob[i] = imh[i] / (double)widthheight;//���ʼ��㣬ÿ��������ֵ��Ӧһ������ֵ
        }

        double* equals = new double[levelnumber]();//ӳ�䵽�µ�ֵ�ռ�
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
            int vlaue = imgmaplevel[i];//vlaue�����ֵΪlevelnumber-1����СֵΪ0
            pOutputimg[i] = equals[vlaue] / (levelnumber - 1);//��һ�������ֵΪlevelnumber-1����ӦΪ1
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
	int gaswidup = wide / 2;//����ȡ��,��Ϊ����
	int gaswiddown = -gaswidup;//ȡ������Ϊ����

	//������ʱֵ
	double elementvalue1 = 1.0 / (2.0 * tho * tho * 3.141592653);
	double elementvalue2 = 1.0 / (2.0 * tho * tho);

	for (int i = gaswiddown; i <= gaswidup; i++)
	{
		double* gaustemplrow = matGastemplate  + (i + gaswidup) * wide;
		for (int j = gaswiddown; j <= gaswidup; j++)
		{
			double* gaustemplrcol = gaustemplrow + j + gaswidup;
			*(gaustemplrcol) = elementvalue1 * exp(-(i * i + j * j) * elementvalue2);//��˹��ʽ����
		}
	}

	double sumtemp = 0;
	sumtemp = accumulate(matGastemplate , matGastemplate  + wide* wide, 0.0f);
	for (int i = 0; i < wide * wide; i++)
	{
		matGastemplate [i] = matGastemplate [i] / sumtemp;//��һ����ȷ�����ɵ�ģ�������ܺ�Ϊ1
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

	double* addtempimg = new double[tempwid*tempheight]();//����������ͼ��ռ䣬������ͼ������������ͼ��ռ�

	//����¿ռ�
	//�м䲿�����
	for (int i = addheight, step1 = 0; i < addheight + height; i++, step1 += width)
	{
		double* tempcalulaterow = addtempimg  + i * tempwid;  //���ͼ���i�е��׵�ַ               
		double* inputrow = matInputimg  + step1;  //���ͼ���i�е��׵�ַ    
		for (int j = addwidth, step2 = 0; j < addwidth + width; j++, step2 += 1)
		{
			tempcalulaterow[j] = inputrow[step2];
		}
	}

	//���Ҳ�߽����
	for (int i = addheight, step1 = 0; i < addheight + height; ++i, step1 += width) //�����¿ռ�
	{
		double* addtempimgrow = addtempimg  + i * tempwid;  //���ͼ���i�е��׵�ַ   
		double leftvalue = matInputimg [step1];
		double rightvalue = matInputimg [step1 - 1 + width];
		//���߽����
		for (int j = 0; j < addwidth; ++j)
		{
			addtempimgrow[j] = leftvalue;//���߽�����Ϊ��������������ͬ��
		}
		//�Ҳ�߽����
		for (int k = width + addwidth; k < tempwid; ++k)
		{
			addtempimgrow[k] = rightvalue;//�Ҳ�߽�����Ϊ��������������ͬ��
		}
	}

	//�ϲ�߽����
	for (int i = 0; i < addheight; i++)//�ϲ���������
	{
		double* addtempimgrow = addtempimg  + i * tempwid;  //���ͼ���i�е��׵�ַ  
		double* addheightimgrow = addtempimg  + addheight * tempwid;  //���ͼ���addheight�е��׵�ַ 

		for (int k = 0; k < tempwid; k++)
		{
			addtempimgrow[k] = addheightimgrow[k];
		}
	}
	//�²��������
	for (int i = height + addheight; i < tempheight; i++)
	{
		double* addtempimgrow = addtempimg  + i * tempwid;  //���ͼ���i�е��׵�ַ 
		double* addheightimgrow = addtempimg  + (height + addheight - 1) * tempwid;  //���ͼ���i�е��׵�ַ 

		for (int k = 0; k < tempwid; k++)
		{
			addtempimgrow[k] = addheightimgrow[k];
		}
	}

	//ģ�����
	for (int i = addheight, step1 = addheight * tempwid, step2 = 0; i < addheight + height; ++i, step1 += tempwid, step2 += width) //����ģ���˲�
	{
		double* addtempimgrow = addtempimg  + step1; //��չͼ���i�е��׵�ַ
		double* outputimgrow = matOutputimg  + step2;  //���ͼ���i�е��׵�ַ          
		for (int j = addwidth, k = 0; j < addwidth + width; ++j, k += 1)
		{
			double sumtempvalue = 0;
			double* addtempimgpos = addtempimgrow + j; //��չͼ���i�е�j����������Ԫ�صĵ�ַ

			//��ģ���С�ľֲ������ڣ�ѭ�������ۼ�
			for (int m = -addheight; m <= addheight; m++)
			{
				double* starinputpos = addtempimgpos + m * tempwid;
				double* matTemplaterow = matTemplate  + (m + addheight) * matTemplatewidth;
				for (int n = -addwidth; n <= addwidth; n++)
				{
					sumtempvalue = sumtempvalue + starinputpos[n] * matTemplaterow[n + addwidth];
				}
			}
			//��ȡ�ۼ�ƽ��ֵ������ģ���ȨֵΪ1���˲�������ƽ����
			outputimgrow[k] = sumtempvalue;//���ͼ���i�е�j����������Ԫ�صĵ�ַ
		}
	}
	delete[] addtempimg;
	addtempimg = nullptr;
}

//��ȡ�ض���ʽ���ļ���  
void Img_Process::GetAllFormatFiles(string path, vector<string>& files, string format)
{
	//�ļ����    
	long long  hFile = 0;
	//�ļ���Ϣ    
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
				files.push_back(p.assign(fileinfo.name));  //���ļ�·�����棬Ҳ����ֻ�����ļ���:  p.assign(path).append("\\").append(fileinfo.name)  
			}
		} while (_findnext(hFile, &fileinfo) == 0);

		/*if ((hFile = _findfirst("G:\\Yangyingjian\\MatlabToC\\LUNG_ventilation_data\\original\\*.dcm", &fileinfo)) == -1L)
			printf("û���ҵ�ƥ�����Ŀ\n");
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
				if (deter !=0 )//ֻҪ����ֵΪ1�������뿼�Ƿ���
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
								if (*nebinputimgpos !=0)
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
				if (deter != 0)//ֻҪ����ֵΪ1�������뿼�Ƿ���
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
								if (*nebinputimgpos != 0)
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

void Img_Process::Isinverimg(unsigned short* pArrimg, unsigned char* pmap255, int imgwidth, int imgheight)
{
    //ͼ�����Ͻ�
    double leftupsummeanvalue = 0;
    int leftupsumnumber = 1;
    for (int i = 100, step1 = 100 * imgwidth; i < 200; i++, step1 += imgwidth)//ͼ���
    {
        for (int j = 100; j < 200; j++)
        {
            leftupsummeanvalue += pArrimg[step1 + j];
            leftupsumnumber++;
        }
    }

    //ͼ�����Ͻ�
    int rightupsummeanvalue = 0;
    int rightupsumnumber = 1;
    for (int i = 100, step1 = 100 * imgwidth; i < 200; i++, step1 += imgwidth)//ͼ���
    {
        for (int j = imgwidth - 100; j < imgwidth; j++)
        {
            rightupsummeanvalue += pArrimg[step1 + j];
            rightupsumnumber++;
        }
    }

    //ͼ��������
    double midupsummeanvalue = 0;
    int midupsumnumber = 1;
    for (int i = imgheight / 2, step1 = 100 * imgwidth; i < imgheight / 2 + 200; i++)//ͼ���
    {
        for (int j = imgwidth / 2 - 100; j < imgwidth / 2 + 100; j++)
        {
            midupsummeanvalue += pArrimg[step1 + j];
            midupsumnumber++;
        }
    }
    midupsummeanvalue = midupsummeanvalue / midupsumnumber;//����ƽ��ֵ	
    double meanupleft = (leftupsummeanvalue + rightupsummeanvalue) / (leftupsumnumber + rightupsumnumber);//�߽�ƽ��ֵ

    double* pmapinput = new double[imgwidth * imgheight]();
    for (int i = 0; i < imgwidth * imgheight; i++)
    {
        pmapinput[i] = pArrimg[i];
    }

    //ͼ�񷴰��жϴ���
    if (meanupleft > midupsummeanvalue)//���ͼ�������Ҷ�ֵ�ͣ�����ԭʼͼ��
    {
        //ӳ�䵽255
        Mapto255(pmapinput, pmap255, imgwidth, imgheight);
        //����
        for (int i = 0; i < imgwidth * imgheight; i++)
        {
            pmap255[i] = 255 - pmap255[i];
        }
    }
    else//�������Ƿ���ͼ��ֱ��ӳ��255
    {
        //ֻ��ӳ���㷨
        Mapto255(pmapinput, pmap255, imgwidth, imgheight);
    }

    delete[] pmapinput;
    pmapinput = nullptr;
}
