#include<iostream>
#include<opencv2/opencv.hpp>
#include<Windows.h>
#include<fstream>
#include<io.h>
#include<direct.h>
using namespace std;
using namespace cv;

bool comparePoint(Point2f a, Point2f b);//1
void API_ShiftLength(Mat& inputImg, float& shiftX, float& shiftY);//1
void get_light_info(Mat& inputImg, int nowPattern, int nowImgNum, float* inParam, Mat& dstImg, float* outValue, vector<vector<Point>>& outline, int& badNum);//1
void get_defect_info(Mat& inputImg, int nowPattern, float* inParam, float& shiftX, float& shiftY, Mat& dstImg, float* outValue, vector<vector<Point>>& outline, int& badNum);//1
void API_fillImage(Mat& inputImg, int thresh, int fillType, Mat& dstImg, float& shiftX, float& shiftY);//1
void get_defect_last_roi(Mat& inputImg, Mat& maskImg,  int thresh, int fillType, Mat& dstImg, float& shiftX, float& shiftY);//1
void get_pixel_level(vector<Mat>& src, int& imgArray, float& max);//1
void get_defect_biggest_mask(vector<Mat>& src, float& shiftX, float& shiftY, Mat& maskImg);//1
void get_defect_first_roi(vector<Mat>& src, Mat& maskImg, vector<Mat>& defImg, vector<int>& badSite);//1
double Entropy(Mat& img, string name)//信息熵
{
	int width = img.cols;
	int height = img.rows;
	//开辟内存
	double temp[256] = { 0.0 };

	// 计算每个像素的累积值
	for (int m = 0; m < height; m++)
	{// 有效访问行列的方式
		const uchar* p = img.ptr<uchar>(m);
		for (int n = 0; n < width; n++)
		{
			int i = p[n];
			temp[i] = temp[i] + 1;
		}
	}

	// 计算每个像素的概率
	for (int i = 0; i < 256; i++)
	{
		temp[i] = temp[i] / (width * height);
	}

	double result = 0;
	// 计算图像信息熵
	for (int i = 0; i < 256; i++)
	{
		if (temp[i] == 0.0)
			result = result;
		else
			result = result - temp[i] * (log(temp[i]) / log(2.0));
	}

	cout << name << "的信息熵为：" << result << endl;
	return result;
}



/*** SMD（灰度方差）函数***/
double smd(cv::Mat& image)
{
	assert(image.empty());

	cv::Mat gray_img, smd_image_x, smd_image_y, G;
	if (image.channels() == 3) {
		cv::cvtColor(image, image, COLOR_BGR2GRAY);
	}
	cv::Mat kernel_x(3, 3, CV_32F, cv::Scalar(0));
	kernel_x.at<float>(1, 2) = -1.0;
	kernel_x.at<float>(1, 1) = 1.0;
	cv::Mat kernel_y(3, 3, CV_32F, cv::Scalar(0));
	kernel_y.at<float>(0, 1) = -1.0;
	kernel_y.at<float>(1, 1) = 1.0;
	cv::filter2D(image, smd_image_x, image.depth(), kernel_x);
	cv::filter2D(image, smd_image_y, image.depth(), kernel_y);

	smd_image_x = cv::abs(smd_image_x);
	smd_image_y = cv::abs(smd_image_y);
	G = smd_image_x + smd_image_y;

	return cv::mean(G)[0];
}

/*** SMD2 （灰度方差乘积）函数 ***/
double smd2(cv::Mat& image)
{
	assert(image.empty());

	cv::Mat gray_img, smd_image_x, smd_image_y, G;
	if (image.channels() == 3) {
		cv::cvtColor(image, image, COLOR_BGR2GRAY);
	}

	cv::Mat kernel_x(3, 3, CV_32F, cv::Scalar(0));
	kernel_x.at<float>(1, 2) = -1.0;
	kernel_x.at<float>(1, 1) = 1.0;
	cv::Mat kernel_y(3, 3, CV_32F, cv::Scalar(0));
	kernel_y.at<float>(1, 1) = 1.0;
	kernel_y.at<float>(2, 1) = -1.0;
	cv::filter2D(image, smd_image_x, image.depth(), kernel_x);
	cv::filter2D(image, smd_image_y, image.depth(), kernel_y);

	smd_image_x = cv::abs(smd_image_x);
	smd_image_y = cv::abs(smd_image_y);
	cv::multiply(smd_image_x, smd_image_y, G);

	return cv::mean(G)[0];
}

/*** 能量梯度函数 ***/
double energy_gradient(cv::Mat& image)
{
	Mat gray_img, smd_image_x, smd_image_y, G;
	if (image.channels() == 3) {
		cvtColor(image, image, COLOR_BGR2GRAY);
	}

	Mat kernel_x(3, 3, CV_32F, Scalar(0));
	kernel_x.at<float>(1, 2) = -1.0;
	kernel_x.at<float>(1, 1) = 1.0;
	Mat kernel_y(3, 3, CV_32F, cv::Scalar(0));
	kernel_y.at<float>(1, 1) = 1.0;
	kernel_y.at<float>(2, 1) = -1.0;
	filter2D(image, smd_image_x, image.depth(), kernel_x);
	filter2D(image, smd_image_y, image.depth(), kernel_y);

	multiply(smd_image_x, smd_image_x, smd_image_x);
	multiply(smd_image_y, smd_image_y, smd_image_y);
	G = smd_image_x + smd_image_y;

	return mean(G)[0];
}

/*** 拉普拉斯函数 ***/
double laplacian(Mat& image)
{
	Mat gray_img, lap_image;
	if (image.channels() == 3) {
		cvtColor(image, image, COLOR_BGR2GRAY);
	}
	Laplacian(image, lap_image, CV_32FC1);
	lap_image = abs(lap_image);

	return mean(lap_image)[0];
}


int main()
{
	for (int i = 16; i < 17; i++)
	{
		string num = to_string(i);
		string lightAddress = "E:\\AFI复判demo\\亮点\\" + num;
		string address = "E:\\AFI复判demo\\TestImg\\5-26\\16-3号密集点0\\" + num + "\\Mura\\";
		
		vector<Mat> src;
		vector<Mat> outImg;
		int thresh = 70;
		int nowPattern = 0;
		int nowImgNum = 0;
		float inParam[60] = { 0 };
		float outValue[60] = { 0 };
		float backup[60] = { 0 };
		vector<vector<Point>> outline;
		vector<string> fn;
		glob(lightAddress, fn);
		if (fn.empty())
		{
			cout << i << " no address.." << endl;
			continue;
		}
		for (int i = 0; i < fn.size(); i++)
		{
			Mat srcImg;
			srcImg = imread(fn[i]);
			if (srcImg.empty())
			{
				cout << " no image.." << endl;
				return -2;
			}
			src.push_back(srcImg);
		}
		int nowTime = GetTickCount64();
		if (src.size() != 5)
		{
			continue;
		}
		/********************API**********************/

		string prefix = "../Log/ReAFI/";
		if (_access(prefix.c_str(), 0) == -1)
		{
			_mkdir(prefix.c_str());
		}
		SYSTEMTIME sys;
		GetLocalTime(&sys);
		string timePath = to_string(sys.wYear) + '.' + to_string(sys.wMonth) + '.' + to_string(sys.wDay);
		string LogPath = prefix + timePath + ".log";
		fstream fs;
		fs.open(LogPath, ios::app | ios::out);
		string strSime = to_string(sys.wHour) + ':' + to_string(sys.wMinute) + ':' + to_string(sys.wSecond);
		fs << strSime;
		int timeSpend = GetTickCount64();

		int badNum = 0;//坏点数量
		float shiftX = 0.0;//像素x方向位移量
		float shiftY = 0.0;//像素y方向位移量
		int imgArray = 0;//亮点所在位置
		vector<int> badSite;
		float max = 0;//组图中最大平均值
		outImg.assign(src.begin(), src.end());
		for (int i = 0; i < src.size(); i++)
		{
			if (src[i].channels() == 3)
			{
				cvtColor(src[i], src[i], COLOR_BGR2GRAY);
			}
		}
		get_pixel_level(src, imgArray, max);//寻找像素在哪一层
		
		if (max < 10)
		{
			/***********亮点检测***********/
			get_light_info(src[imgArray], imgArray, nowImgNum, inParam, outImg[imgArray], outValue, outline, badNum);
			for (int i = 0; i < (20 - strSime.size()); i++)
			{
				fs << '-';
			}
			timeSpend = GetTickCount64() - timeSpend;
			string strTime = to_string(timeSpend) + "ms";
			fs << strTime;
			for (int i = 0; i < (15 - strTime.size()); i++)
			{
				fs << '-';
			}
			fs << " 亮点 " << endl;
		}
		else
		{
			Mat maskImg, img2;
			vector<Mat> defImg;
			Mat defPixelImg1, defPixelImg2;
			vector<vector<Point>> contours;
			API_ShiftLength(src[imgArray], shiftX, shiftY);//获取像素间隔
			get_defect_biggest_mask(src, shiftX, shiftY, maskImg);//获得初步的异物mask
			get_defect_first_roi(src,maskImg, defImg, badSite);//大致的异物mask
			for (int i = 0; i < defImg.size(); i++)
			{
				int a = badSite[i];
				get_defect_last_roi(src[a], defImg[i], 50, 1, defPixelImg1, shiftX, shiftY);//精确的异物mask
				get_defect_info(defPixelImg1, a, inParam, shiftX, shiftY, outImg[a], outValue, outline, badNum);
			}
			for (int i = 0; i < (20 - strSime.size()); i++)
			{
				fs << '-';
			}
			timeSpend = GetTickCount64() - timeSpend;
			string strTime = to_string(timeSpend) + "ms";
			fs << strTime;
			for (int i = 0; i < (15 - strTime.size()); i++)
			{
				fs << '-';
			}
			fs << badNum << " 个异物" << endl;
			cout << strTime << endl;
		}
	}
	
	return 0;
}

//获得初步不良mask
void get_defect_first_roi(vector<Mat>& src,Mat& maskImg, vector<Mat>& defImg,  vector<int>& badSite)
{
	vector<vector<Point>> contours;
	findContours(maskImg,contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	vector<float>defArea;
	for (int i = 0; i < contours.size(); i++)
	{
		float area = contourArea((Mat)contours[i]);
		if (area > 10)
		{
			defArea.push_back(area);
			Mat tempImg = Mat::zeros(maskImg.size(), maskImg.type());
			drawContours(tempImg, contours, i, Scalar(255), -1);
			defImg.push_back(tempImg);
			Mat copyImg0, copyImg1, copyImg2, copyImg3, copyImg4;
			src[0].copyTo(copyImg0, tempImg);
			src[1].copyTo(copyImg1, tempImg);
			src[2].copyTo(copyImg2, tempImg);
			src[3].copyTo(copyImg3, tempImg);
			src[4].copyTo(copyImg4, tempImg);
			vector<Mat> allCopy;
			allCopy.push_back(copyImg0);
			allCopy.push_back(copyImg1);
			allCopy.push_back(copyImg2);
			allCopy.push_back(copyImg3);
			allCopy.push_back(copyImg4);
			float getBlackPixel[5] = { 0 };
			int sum = 0;
			float getMinPixel[5] = { 0 };
			for (int m = 0; m < allCopy.size(); m++)
			{
				Mat tempImg = allCopy[m];
				int blackPixel = 0;
				for (int i = 0; i < tempImg.rows; i++)
				{
					uchar* p = tempImg.ptr<uchar>(i);
					for (int j = 0; j < tempImg.cols; j++)
					{
						if (p[j] < 8 && p[j] >0)
						{
							blackPixel++;
						}
					}
				}
				getBlackPixel[m] = blackPixel;//统计ROI的亮暗程度
			}
			int site = max_element(getBlackPixel, getBlackPixel + 5) - getBlackPixel;//黑点最多的即为最暗，判定其为异物所在层

			badSite.push_back(site);//确定暗点或者异物在哪一层
			for (int i = 0; i < 5; i++)
			{
				sum += getBlackPixel[i];
			}
			float blackScale = 0.001;
			if (sum != 0)
			{
				blackScale = (float)getBlackPixel[site] / sum;
			}
			if (getBlackPixel[badSite[i]] < 40 && blackScale < 0.7)
			{
				badSite.erase(badSite.end() - 1);
				getMinPixel[0] = mean(allCopy[0])[0];
				getMinPixel[1] = mean(allCopy[1])[0];
				getMinPixel[2] = mean(allCopy[2])[0];
				getMinPixel[3] = mean(allCopy[3])[0];
				getMinPixel[4] = mean(allCopy[4])[0];
				badSite.push_back(min_element(getMinPixel, getMinPixel + 5) - getMinPixel);//确定暗点或者异物在哪一层
			}

		}
	}
	if (!badSite.empty())
	{
		int maxAreaSite = max_element(defArea.begin(), defArea.end()) - defArea.begin();
		int maxfloorSite = *max_element(badSite.begin(), badSite.end());
		int minfloorSite = *min_element(badSite.begin(), badSite.end());
		int sub = maxfloorSite - minfloorSite;
		if (sub < 2 && sub != 0)
		{
			int badSize = badSite.size();
			int badFloor = badSite[maxAreaSite];
			badSite.clear();
			for (int i = 0; i < badSize; i++)
			{
				badSite.push_back(badFloor);
			}
		}
	}
}


//填充图像
void API_fillImage(Mat& inputImg, int thresh, int fillType, Mat& dstImg, float& shiftX, float& shiftY)
{
	Mat defPixImg1, defThreshImg1;
	Mat drawImg;
	dstImg = Mat::zeros(inputImg.size(), CV_8U);
	drawImg = Mat::zeros(inputImg.size(), CV_8U);
	Mat srcImg = inputImg.clone();
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	int Width = inputImg.cols;
	int Height = inputImg.rows;
	int shiftLengthX = round(shiftX * 2);
	int shiftLengthY = round(shiftY * 2);
	threshold(inputImg, srcImg, 10, 0, 3);

	//将图上下扩充
	Mat bigMaskImg1 = Mat::zeros(Size(Width + abs(shiftLengthX) * 2, Height + abs(shiftLengthY) * 2), CV_8U);
	for (size_t i = 0; i < Height; i++)
	{
		uchar* p = srcImg.ptr<uchar>(i);
		uchar* u = bigMaskImg1.ptr<uchar>(i);
		for (size_t j = 0; j < Width; j++)
		{
			int a = j + shiftLengthX + shiftLengthY * Width;
			if ((i + shiftLengthY) > Height)
			{
				a = j - shiftLengthX - shiftLengthY * Width;
			}
			int  b = j - shiftLengthX - shiftLengthY * Width;
			if (i < shiftLengthY)
			{
				b = j + shiftLengthX + shiftLengthY * Width;
			}
			if (p[j] > thresh)
			{
				if ((u[j] - p[j] > 10 || (abs(p[j] - p[a]) > 20 && abs(p[j] - p[b]) > 20)) && fillType == 1)
				{
					continue;
				}
				u[j] = p[j];
				u[(j + shiftLengthX) + shiftLengthY * bigMaskImg1.cols] = p[j];
				u[(j + shiftLengthX * 2) + shiftLengthY * bigMaskImg1.cols * 2] = p[j];

			}
		}
	}
	//mask与原图相减
	Mat maskImg1 = bigMaskImg1(Rect(0, 0, Width, Height)).clone();
	morphologyEx(maskImg1, maskImg1, MORPH_ERODE, kernel);
	defPixImg1 = maskImg1 - srcImg;
	threshold(defPixImg1, defThreshImg1, 20, 255, 0);//最低20，再低影响暗点的判断
	morphologyEx(defThreshImg1, defThreshImg1, MORPH_CLOSE, kernel);

	//原图选出坏点
	vector<vector<Point>> contours1;
	vector<vector<Point>> contours2;
	vector<Point2f> ALLPixCorePoint;

	findContours(defThreshImg1, contours1, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	for (size_t i = 0; i < contours1.size(); i++)
	{
		Moments m = moments((Mat)contours1[i]);
		Point2f	mPoint;
		mPoint.x = m.m10 / m.m00;
		mPoint.y = m.m01 / m.m00;
		double Area = contourArea(Mat(contours1[i]));
		Point2f circleCenter;
		if (Area > 20)
		{
			drawContours(drawImg, contours1, i, Scalar(255), -1);
			ALLPixCorePoint.push_back(mPoint);
		}
	}

	//距离近的异物点合并
	for (int i = 0; i < ALLPixCorePoint.size(); i++)
	{
		for (int j = i + 1; j < ALLPixCorePoint.size(); j++)
		{
			double a = sqrt(pow((ALLPixCorePoint[i].x - ALLPixCorePoint[j].x), 2) + pow((ALLPixCorePoint[i].y - ALLPixCorePoint[j].y), 2));
			if (a < shiftY)
			{
				line(drawImg, ALLPixCorePoint[i], ALLPixCorePoint[j], Scalar(255));
			}
		}
	}

	findContours(drawImg, contours2, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	for (int i = 0; i < contours2.size(); i++)
	{
		Moments m = moments((Mat)contours2[i]);
		Point2f	mPoint;
		mPoint.x = m.m10 / m.m00;
		mPoint.y = m.m01 / m.m00;
		if (m.m00 > 5 && mPoint.x < Width / 10 * 9 && mPoint.x > Width / 50 * 3 && mPoint.y < Height / 10 * 9 && mPoint.y > Height / 50 * 3)
		{
			vector<vector<Point>> hullContours(contours2.size());
			convexHull(contours2[i], hullContours[i]);
			drawContours(dstImg, hullContours, i, Scalar(255), -1, 8);
		}
	}
	waitKey(1);
}

//获取异物mask模版
void get_defect_last_roi(Mat& inputImg, Mat& maskImg, int thresh, int fillType, Mat& dstImg, float& shiftX, float& shiftY)
{
	Mat defPixImg1, defThreshImg1;
	Mat drawImg;
	dstImg = Mat::zeros(inputImg.size(), CV_8U);
	drawImg = Mat::zeros(inputImg.size(), CV_8U);
	Mat srcImg = inputImg.clone();
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	int Width = inputImg.cols;
	int Height = inputImg.rows;
	int shiftLengthX = round(shiftX * 2);
	int shiftLengthY = round(shiftY * 2);
	threshold(inputImg, srcImg, 20, 0, 3);

	vector<vector<Point>> contours1;
	findContours(maskImg, contours1, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	for (int i = 0; i < contours1.size(); i++)
	{
		Point2f minCircle;
		float minD;
		minEnclosingCircle((Mat)contours1[i], minCircle, minD);
		int rectX = minCircle.x - minD * 4;
		int rectY = minCircle.y - minD * 4;
		int rectLength = minD * 8;
		if (rectX < 0)
		{
			rectX = abs(shiftLengthX);
		}
		if (rectY < 0)
		{
			rectY = shiftLengthY;
		}
		if (rectLength > Height)
		{
			rectLength = 1;
		}
		if (rectX > (Width - rectLength))
		{
			rectX = Width - rectLength;
		}
		if (rectY > (Height - rectLength))
		{
			rectY = Height - rectLength;
		}
		Rect rect = Rect(rectX, rectY, rectLength, rectLength);
		Mat rectImg = srcImg(rect);
		int rect2X = rect.tl().x - shiftLengthX;
		int rect2Y = rect.tl().y - shiftLengthY;
		if (rect2X < 0)
		{
			rect2X = 0;
		}
		if (rect2Y < 0)
		{
			rect2Y = 0;
		}
		Rect rect2 = Rect(rect2X, rect2Y, rect.width, rect.height);;
		Mat tempImg = inputImg(rect2);
		Mat emptyImg = Mat::zeros(rect.height, rect.width, CV_8U);
		emptyImg = tempImg - rectImg;
		threshold(emptyImg, emptyImg, 30, 255, 0);
		emptyImg.copyTo(drawImg(rect));

	}

	//原图选出坏点
	vector<vector<Point>> contours2;
	vector<Point2f> ALLPixCorePoint;
	findContours(drawImg, contours2, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	for (size_t i = 0; i < contours2.size(); i++)
	{
		Moments m = moments((Mat)contours2[i]);
		Point2f	mPoint;
		mPoint.x = m.m10 / m.m00;
		mPoint.y = m.m01 / m.m00;
		double Area = contourArea(Mat(contours2[i]));
		if (Area > 20)
		{
			drawContours(drawImg, contours2, i, Scalar(255), -1);
			ALLPixCorePoint.push_back(mPoint);
		}
	}

	//临近异物合并
	for (int i = 0; i < ALLPixCorePoint.size(); i++)
	{
		for (int j = i + 1; j < ALLPixCorePoint.size(); j++)
		{
			double a = sqrt(pow((ALLPixCorePoint[i].x - ALLPixCorePoint[j].x), 2) + pow((ALLPixCorePoint[i].y - ALLPixCorePoint[j].y), 2));
			if (a < shiftY)
			{
				line(drawImg, ALLPixCorePoint[i], ALLPixCorePoint[j], Scalar(255));
			}
		}
	}
	vector<vector<Point>> contours3;
	findContours(drawImg, contours3, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	for (int i = 0; i < contours3.size(); i++)
	{
		Moments m = moments((Mat)contours3[i]);
		Point2f	mPoint;
		mPoint.x = m.m10 / m.m00;
		mPoint.y = m.m01 / m.m00;
		if (m.m00 > 20 && mPoint.x < Width / 10 * 9 && mPoint.x > Width / 10 && mPoint.y < Height / 10 * 9 && mPoint.y > Height / 10)
		{ 
			vector<vector<Point>> hullContours(contours3.size());
			convexHull(contours3[i], hullContours[i]);
			drawContours(dstImg, hullContours, i, Scalar(255), -1, 8);
			waitKey(1);
		}
	}
}

//获取不良最大mask
void get_defect_biggest_mask(vector<Mat>& src,float& shiftX,float& shiftY, Mat& maskImg)
{
	Mat img0, img1,img2, img3, img4;
	API_fillImage(src[0], 50, 0, img0, shiftX, shiftY);
	API_fillImage(src[1], 50, 1, img1, shiftX, shiftY);
	API_fillImage(src[2], 50, 1, img2, shiftX, shiftY);
	API_fillImage(src[3], 50, 1, img3, shiftX, shiftY);
	API_fillImage(src[4], 50, 1, img4, shiftX, shiftY);

	vector<Mat>img;
	img.push_back(img0);
	img.push_back(img1);
	img.push_back(img2);
	img.push_back(img3);
	img.push_back(img4);
	float getMaxPixel[5] = { 0 };
	getMaxPixel[0] = mean(img0)[0];
	getMaxPixel[1] = mean(img1)[0];
	getMaxPixel[2] = mean(img2)[0];
	getMaxPixel[3] = mean(img3)[0];
	getMaxPixel[4] = mean(img4)[0];
	int maxSite = max_element(getMaxPixel, getMaxPixel + 5) - getMaxPixel;//获取最大mask
	maskImg = img[maxSite].clone();
}

//异物
void get_defect_info(Mat& inputImg, int nowPattern, float* inParam, float& shiftX, float& shiftY, Mat& dstImg, float* outValue, vector<vector<Point>>& outline, int& badNum)
{
	int addPixel = 0;
	if (nowPattern > 1)
	{
		addPixel = 4;
	}
	vector<vector<Point>> contours;
	findContours(inputImg, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	for (int i = 0; i < contours.size(); i++)
	{
		double Area = contourArea((Mat)contours[i]);
		RotatedRect rect = minAreaRect(contours[i]);
		Point2f pt[4];
		rect.points(pt);
		float Width = 0;
		float Height = 0;//控制宽比高更长（长宽比）
		if (rect.size.width > rect.size.height)
		{
			Width = rect.size.width;
			Height = rect.size.height;
		}
		else
		{
			Width = rect.size.height;
			Height = rect.size.width;
		}
		if (Height > 7)
		{
			for (int i = 0; i < 4; i++)
			{
				line(dstImg, pt[i], pt[(i + 1) % 4], Scalar(255, 255, 255), 1);
			}
			string pixelSizeW = "Width: " + to_string(Width + addPixel);
			string pixelSizeH = "Height: " + to_string(Height + addPixel);
			string pixelNum = to_string(badNum + 1) + ":";
			putText(dstImg, pixelNum, Point(pt[1].x - 75, pt[1].y - 60), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 1);
			putText(dstImg, pixelSizeW, Point(pt[1].x - 50, pt[1].y - 35), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 1);
			putText(dstImg, pixelSizeH, Point(pt[1].x - 50, pt[1].y - 10), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 1);
			outline.push_back(contours[i]);
			outValue[0 + badNum * 10] = nowPattern;
			outValue[5 + badNum * 10] = rect.center.x;
			outValue[6 + badNum * 10] = rect.center.y;
			outValue[7 + badNum * 10] = Width + addPixel;
			outValue[8 + badNum * 10] = Height + addPixel;
			badNum++;
		}
	}
	waitKey(1);
}

//暗点
void getDarkSpot(Mat& inputImg, int nowPattern, int nowImgNum, float* inParam, Mat& dstImg, float* outValue, int& badNum, float& shiftX, float& shiftY)
{
	Mat defPixelImg;
	int fillType = 0;
	API_fillImage(inputImg, 60, fillType, defPixelImg, shiftX, shiftY);

	//获取坏点最小包围矩形和最小外接圆
	vector<vector<Point>> contours2;
	int pixelNum = 0;
	findContours(defPixelImg, contours2, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	for (int i = 0; i < contours2.size(); i++)
	{
		double Area = contourArea((Mat)contours2[i]);
		RotatedRect rect = minAreaRect(contours2[i]);
		Point2f pt[4];
		rect.points(pt);
		float Width = 0;
		float Height = 0;//控制宽比高更长（长宽比）
		if (rect.size.width > rect.size.height)
		{
			Width = rect.size.width;
			Height = rect.size.height;
		}
		else
		{
			Width = rect.size.height;
			Height = rect.size.width;
		}
		if (Height > 5)
		{
			for (int i = 0; i < 4; i++)
			{
				line(dstImg, pt[i], pt[(i + 1) % 4], Scalar(255, 255, 255), 1);
			}
			string pixelSizeW = "Width: " + to_string(Width * 1.725);
			string pixelSizeH = "Height: " + to_string(Height * 1.725);
			putText(dstImg, pixelSizeW, Point(rect.center.x - 100, rect.center.y - 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1);
			putText(dstImg, pixelSizeH, Point(rect.center.x - 50, rect.center.y - 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1);
			outValue[0 + badNum * 10] = nowPattern;
			outValue[5 + badNum * 10] = rect.center.x;
			outValue[6 + badNum * 10] = rect.center.y;
			outValue[7 + badNum * 10] = Width * 1.725;
			outValue[8 + badNum * 10] = Height * 1.725;
			badNum++;
		}
	}
	waitKey(1);
}

//亮点
void get_light_info(Mat& inputImg, int nowPattern, int nowImgNum, float* inParam, Mat& dstImg, float* outValue, vector<vector<Point>>& outline, int& badNum)
{
	Mat srcImg = inputImg.clone();
	Mat drawImg = Mat::zeros(inputImg.size(), CV_8U);
	Mat threshImg;
	threshold(srcImg, threshImg, 10, 255, 0);
	vector<vector<Point>> contours;
	vector<vector<Point>> contours2;
	//寻找最大轮廓
	findContours(threshImg, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	float maxArea = 0;
	int maxPattern = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		double area = contourArea((Mat)contours[i]);
		if (area > maxArea)
		{
			maxArea = area;
			maxPattern = i;
		}
	}
	if (maxArea > 0)
	{
		drawContours(drawImg, contours, maxPattern, Scalar(255), -1, 8);
	}

	//亮点信息输出
	findContours(drawImg, contours2, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	for (int j = 0; j < contours2.size(); j++)
	{
		double Area = contourArea((Mat)contours2[j]);

		RotatedRect rect = minAreaRect(contours2[j]);
		Point2f pt[4];
		rect.points(pt);
		float Width = 0;
		float Height = 0;//控制宽比高长度（长宽比）
		if (rect.size.width > rect.size.height)
		{
			Width = rect.size.width;
			Height = rect.size.height;
		}
		else
		{
			Width = rect.size.height;
			Height = rect.size.width;
		}
		for (int i = 0; i < 4; i++)
		{
			line(dstImg, pt[i], pt[(i + 1) % 4], Scalar(255, 255, 255), 1);
		}

		if (Height > 5)
		{
			string pixelSizeW = "Width: " + to_string(Width);
			string pixelSizeH = "Height: " + to_string(Height);
			putText(dstImg, pixelSizeW, Point(rect.center.x - 100, rect.center.y - 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1);
			putText(dstImg, pixelSizeH, Point(rect.center.x - 50, rect.center.y - 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1);
			outline.push_back(contours2[j]);
			outValue[0 + badNum * 10] = nowPattern;
			outValue[5 + badNum * 10] = rect.center.x;
			outValue[6 + badNum * 10] = rect.center.y;
			outValue[7 + badNum * 10] = Width;
			outValue[8 + badNum * 10] = Height;
			badNum++;
		}
	}
}

//寻找像素层
void get_pixel_level(vector<Mat>& src, int& imgArray, float& max)
{
	float maxPixel = 0;
	for (int m = 0; m < src.size(); m++)
	{
		float a = mean(src[m])[0];
		Mat tempImg = src[m].clone();
		int subPixel = 0;
		int numPixel = 0;
		for (int i = 0; i < tempImg.rows; i++)
		{
			uchar* p = tempImg.ptr<uchar>(i);
			for (int j = 0; j < tempImg.cols; j++)
			{
				if (p[j] > 35 && p[j - 1] > 0 && p[j + 1] > 0)
				{
					subPixel += p[j];
					numPixel++;
				}
			}
		}
		if (numPixel == 0)
		{
			continue;
		}
		else
		{
			float meanPixel = subPixel / numPixel;
			if (meanPixel > maxPixel && numPixel > 25)
			{
				maxPixel = meanPixel;
				max = a;
				imgArray = m;
			}
		}
		
	}
	waitKey(1);
}

//获取像素偏移量
void API_ShiftLength(Mat& inputImg, float& shiftX, float& shiftY)
{
	Mat srcImg = inputImg.clone();
	int Width = srcImg.cols;
	int Height = srcImg.rows;
	Mat threshImg;
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	vector<vector<Point>> contours;
	vector<Point2f> pixelCenter;
	vector<vector<Point2f>> sortPixCenter;

	if (inputImg.channels() == 3)
	{
		cvtColor(inputImg, inputImg, COLOR_BGR2GRAY);
	}
	threshold(inputImg, threshImg, 250, 255, 0);//将亮的像素分割出来
	vector<Mat> allRoiImg;
	allRoiImg.push_back(threshImg(Rect(Width / 10, Height / 10, Width / 5, Height / 5)));//边角
	allRoiImg.push_back(threshImg(Rect(Width / 10 * 7, Height / 10, Width / 5, Height / 5)));
	allRoiImg.push_back(threshImg(Rect(Width / 10 * 7, Height / 10 * 7, Width / 5, Height / 5)));
	allRoiImg.push_back(threshImg(Rect(Width / 10, Height / 10 * 7, Width / 5, Height / 5)));

	allRoiImg.push_back(threshImg(Rect(Width / 4, Height / 4, Width / 5, Height / 5)));//中间
	allRoiImg.push_back(threshImg(Rect(Width / 4, Height / 2, Width / 5, Height / 5)));
	allRoiImg.push_back(threshImg(Rect(Width / 2, Height / 4, Width / 5, Height / 5)));
	allRoiImg.push_back(threshImg(Rect(Width / 2, Height / 2, Width / 5, Height / 5)));
	allRoiImg.push_back(threshImg(Rect(Width / 8 * 3, Height / 8 * 3, Width / 5, Height / 5)));

	Entropy(allRoiImg[0], "0");
	Entropy(allRoiImg[1], "1");
	Entropy(allRoiImg[2], "2");
	Entropy(allRoiImg[3], "3");
	Entropy(allRoiImg[4], "4");
	Entropy(allRoiImg[5], "5");
	Entropy(allRoiImg[6], "6");
	Entropy(allRoiImg[7], "7");
	Entropy(allRoiImg[8], "8");

	float meanPixel[9] = { 0 };
	meanPixel[0] = mean(allRoiImg[0])[0];
	meanPixel[1] = mean(allRoiImg[1])[0];
	meanPixel[2] = mean(allRoiImg[2])[0];
	meanPixel[3] = mean(allRoiImg[3])[0];
	meanPixel[4] = mean(allRoiImg[4])[0];
	meanPixel[5] = mean(allRoiImg[5])[0];
	meanPixel[6] = mean(allRoiImg[6])[0];
	meanPixel[7] = mean(allRoiImg[7])[0];
	meanPixel[8] = mean(allRoiImg[8])[0];

	int maxSite = max_element(meanPixel, meanPixel + 9) - meanPixel;

	Mat roiImg = allRoiImg[maxSite];

	morphologyEx(roiImg, roiImg, MORPH_OPEN, kernel);

	findContours(roiImg, contours, RETR_LIST, CHAIN_APPROX_NONE);//获取所有亮像素的质心坐标
	for (size_t i = 0; i < contours.size(); i++)
	{
		if (contourArea((Mat)contours[i]) > 35)
		{
			float pixelArea = contourArea(Mat(contours[i]));
			Moments mm = moments(Mat(contours[i]));
			Point2f pixCt;
			pixCt.x = mm.m10 / mm.m00;
			pixCt.y = mm.m01 / mm.m00;
			if (pixCt.x > 5 || pixCt.x < (roiImg.cols - 5) || pixCt.y > 5 || pixCt.y < (roiImg.rows - 5))
			{
				pixelCenter.push_back(pixCt);
			}

		}
	}

	vector<Point2f> pixCenterClone;
	vector<Point2f> tempPoint;
	pixCenterClone.assign(pixelCenter.begin(), pixelCenter.end());//对质心坐标进行排序
	do
	{
		for (int j = pixCenterClone.size() - 1; j > -1; j--)
		{
			if (abs(pixCenterClone[0].x - pixCenterClone[j].x) < 10)
			{
				tempPoint.push_back(pixCenterClone[j]);
				pixCenterClone.erase(pixCenterClone.begin() + j);
			}
		}
		sort(tempPoint.begin(), tempPoint.end(), comparePoint);
		sortPixCenter.push_back(tempPoint);
		tempPoint.clear();
	} while (pixCenterClone.size() > 5);


	float pixNum = 0;//获取每行质心个数的平均值，,目的是为了排除坏点部分的行质心坐标
	for (int i = 0; i < sortPixCenter.size(); i++)
	{
		pixNum += sortPixCenter[i].size();
	}
	pixNum = pixNum / sortPixCenter.size();


	double meanDistX = 0.0;
	int Xtime = 0;
	for (int i = 0; i < sortPixCenter.size(); i++)//获取X方向像素间距
	{
		double distPixelX = 0.0;
		if (!(sortPixCenter[i].size() < pixNum))
		{
			for (int j = sortPixCenter[i].size() - 1; j > 0; j--)
			{
				distPixelX += (sortPixCenter[i][j].x - sortPixCenter[i][(int)j - 1].x);
			}
			distPixelX = distPixelX / (sortPixCenter[i].size() - 1);
			meanDistX += distPixelX;
			Xtime++;
		}
	}
	shiftX = meanDistX / Xtime;

	double meanDistY = 0.0;
	int Ytime = 0;
	for (int i = 0; i < sortPixCenter.size(); i++)//获取Y方向像素间距
	{
		double distPixelY = 0.0;
		if (!(sortPixCenter[i].size() < pixNum))
		{
			for (int j = sortPixCenter[i].size() - 1; j > 0; j--)
			{
				distPixelY += (sortPixCenter[i][j].y - sortPixCenter[i][(int)j - 1].y);
			}
			distPixelY = distPixelY / (sortPixCenter[i].size() - 1);
			meanDistY += distPixelY;
			Ytime++;
		}
	}
	shiftY = meanDistY / Ytime;
}

//比较排列坐标
bool comparePoint(Point2f a, Point2f b)
{
	return a.y < b.y;
}