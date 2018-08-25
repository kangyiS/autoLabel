/**********************************************************
TLD���Դ���
2016-03-10
��������Ի���win8.1 64λ�����뻷��vs2013��OpenCV3.0.0���������ͨ��
������Ҫ�������еķ�ʽ���ã�����ͨ��VS�����������������Ҳ��������д��bat�ļ����С�
�����в�����test2.exe -p parameters.yml
ע�⣺debug��release�ٶ�������vs�������б�ֱ��������,��õĽ������release����exe�ļ�����������bat����ȥ���У�����û����
**********************************************************/
#include <opencv2/opencv.hpp>
#include "tld_utils.h"
#include <iostream>
#include <sstream>
#include "TLD.h"
#include <stdio.h>
#include "tinyxml2.h"

#include <string>  
#include "direct.h"  
#include <sys/stat.h> 

//#define CREATE_XML 1

using namespace cv;
using namespace std;
using namespace tinyxml2;
//Global variables
Rect box;
bool drawing_box = false;
bool gotBB = false;
bool tl = true;
bool rep = false;
bool fromfile=false;
string video;
bool XMLisReady = false;
long int filename_int = 0;
string labelXml_path, labelImg_path;
int createXML = 0;
int webcam_id = 0;

void readBB(char* file){
  ifstream bb_file (file);
  string line;
  getline(bb_file,line);
  istringstream linestream(line);
  string x1,y1,x2,y2;
  getline (linestream,x1, ',');
  getline (linestream,y1, ',');
  getline (linestream,x2, ',');
  getline (linestream,y2, ',');
  int x = atoi(x1.c_str());// = (int)file["bb_x"];
  int y = atoi(y1.c_str());// = (int)file["bb_y"];
  int w = atoi(x2.c_str())-x;// = (int)file["bb_w"];
  int h = atoi(y2.c_str())-y;// = (int)file["bb_h"];
  box = Rect(x,y,w,h);
}
//bounding box mouse callback
void mouseHandler(int event, int x, int y, int flags, void *param){
  switch( event ){
  case CV_EVENT_MOUSEMOVE:
    if (drawing_box){
        box.width = x-box.x;
        box.height = y-box.y;
    }
    break;
  case CV_EVENT_LBUTTONDOWN:
    drawing_box = true;
    box = Rect( x, y, 0, 0 );
    break;
  case CV_EVENT_LBUTTONUP:
    drawing_box = false;
    if( box.width < 0 ){
        box.x += box.width;
        box.width *= -1;
    }
    if( box.height < 0 ){
        box.y += box.height;
        box.height *= -1;
    }
    gotBB = true;
    break;
  }
}

void print_help(char** argv){
  printf("use:\n     %s -p /path/parameters.yml\n",argv[0]);
  printf("-s    source video\n-b        bounding box file\n-tl  track and learn\n-r     repeat\n");
}

void read_options(int argc, char** argv,VideoCapture& capture,FileStorage &fs){
  for (int i=0;i<argc;i++){
      if (strcmp(argv[i],"-b")==0){
          if (argc>i){
              readBB(argv[i+1]);
              gotBB = true;
          }
          else
            print_help(argv);
      }
      if (strcmp(argv[i],"-s")==0){
          if (argc>i){
              video = string(argv[i+1]);
              capture.open(video);
              fromfile = true;
          }
          else
            print_help(argv);

      }
      if (strcmp(argv[i],"-p")==0){
          if (argc>i){
              fs.open(argv[i+1], FileStorage::READ);
          }
          else
            print_help(argv);
      }
      if (strcmp(argv[i],"-no_tl")==0){
          tl = false;
      }
      if (strcmp(argv[i],"-r")==0){
          rep = true;
      }
  }
}
void createXMLandImg(Mat img, BoundingBox bb)
{
	XMLDocument doc;
	string str_tmp;

	string folder_str = "jpgSource2";
	const char* folder_char;

	//static long int filename_int = 1;
	const char* filename_char;

	string path_str = "/home/ubuntu/labelImg-master/myself/";
	const char* path_char;

	//int img_width = 1, img_height = 2, img_depth = 3;

	int segmented = 0;

	string name_str = "person";
	const char* name_char;

	int truncated = 0, difficult = 0;

	//int bb_xmin = 0, bb_ymin = 0, bb_xmax = 0, bb_ymax = 0;
	// ������Ԫ��<annotation>  
	XMLElement* root = doc.NewElement("annotation");
	doc.InsertEndChild(root);

	// ������Ԫ��<folder>  
	XMLElement* folderElement = doc.NewElement("folder");
	folder_char = folder_str.data();
	folderElement->SetText(folder_char);
	root->InsertEndChild(folderElement);

	// ������Ԫ��<filename>  
	XMLElement* filenameElement = doc.NewElement("filename");
	str_tmp = std::to_string(filename_int) + ".jpg";
	filename_char = str_tmp.data();
	filenameElement->SetText(filename_char);
	root->InsertEndChild(filenameElement);

	// ������Ԫ��<path>  
	XMLElement* pathElement = doc.NewElement("path");
	str_tmp = path_str + folder_str + "/" + str_tmp;
	path_char = str_tmp.data();
	pathElement->SetText(path_char);
	root->InsertEndChild(pathElement);

	// ������Ԫ��<size>  
	XMLElement* sizeElement = doc.NewElement("size");
	root->InsertEndChild(sizeElement);

	// ������Ԫ��<width>  
	XMLElement* widthElement = doc.NewElement("width");
	widthElement->SetText(img.cols);
	sizeElement->InsertEndChild(widthElement);

	// ������Ԫ��<height>  
	XMLElement* heightElement = doc.NewElement("height");
	heightElement->SetText(img.rows);
	sizeElement->InsertEndChild(heightElement);

	// ������Ԫ��<depth>  
	XMLElement* depthElement = doc.NewElement("depth");
	depthElement->SetText(img.channels());
	sizeElement->InsertEndChild(depthElement);

	// ������Ԫ��<segmented>  
	XMLElement* segmentedElement = doc.NewElement("segmented");
	segmentedElement->SetText(segmented);
	root->InsertEndChild(segmentedElement);

	// ������Ԫ��<object>  
	XMLElement* objectElement = doc.NewElement("object");
	root->InsertEndChild(objectElement);

	// ������Ԫ��<name>  
	XMLElement* nameElement = doc.NewElement("name");
	name_char = name_str.data();
	nameElement->SetText(name_char);
	objectElement->InsertEndChild(nameElement);

	// ������Ԫ��<truncated>  
	XMLElement* truncatedElement = doc.NewElement("truncated");
	truncatedElement->SetText(truncated);
	objectElement->InsertEndChild(truncatedElement);

	// ������Ԫ��<difficult>  
	XMLElement* difficultElement = doc.NewElement("difficult");
	difficultElement->SetText(difficult);
	objectElement->InsertEndChild(difficultElement);

	// ������Ԫ��<bndbox>  
	XMLElement* bndboxElement = doc.NewElement("bndbox");
	objectElement->InsertEndChild(bndboxElement);

	// ��������Ԫ��<xmin>  
	XMLElement* xminElement = doc.NewElement("xmin");
	xminElement->SetText(bb.x);
	bndboxElement->InsertEndChild(xminElement);

	// ��������Ԫ��<ymin>  
	XMLElement* yminElement = doc.NewElement("ymin");
	yminElement->SetText(bb.y);
	bndboxElement->InsertEndChild(yminElement);

	// ��������Ԫ��<xmax>  
	XMLElement* xmaxElement = doc.NewElement("xmax");
	if ((bb.x+bb.width)>=img.cols)
		xmaxElement->SetText(img.cols-1);
	else
		xmaxElement->SetText(bb.x+bb.width);
	bndboxElement->InsertEndChild(xmaxElement);

	// ��������Ԫ��<ymax>  
	XMLElement* ymaxElement = doc.NewElement("ymax");
	if ((bb.y + bb.height)>=img.rows)
		ymaxElement->SetText(img.rows-1);
	else
		ymaxElement->SetText(bb.y + bb.height);
	bndboxElement->InsertEndChild(ymaxElement);

	// ���XML���ļ� 
	str_tmp = labelXml_path + std::to_string(filename_int) + ".xml";
	doc.SaveFile(str_tmp.data());

	//���ԭʼͼƬ
	str_tmp = labelImg_path + std::to_string(filename_int) + ".jpg";
	imwrite(str_tmp, img);

	filename_int++;
}
//���ַ���str�У���old_value�滻Ϊnew_value
string& replace_all(string& str, const string& old_value, const string& new_value)
{
	string::size_type  pos(0);
	while (true)
	{
		if ((pos = str.find(old_value, pos)) != string::npos)
		{
			str.replace(pos, old_value.length(), new_value);
		}
		else
		{
			break;
		}
	}

	return  str;
}
//��strSrcFilePath·������
//����Ŀ�����"D:/1/2/3/"
//            "D:/1/2/3"
//            "D:\\1\\2\\3"
//            "D:\\1\\2\\3\\"
//            ".\\..\\1\\2\\3"�ȶ�����ʽ
bool CreateFolder(string strSrcFilePath)
{
	string strFilePath = replace_all(strSrcFilePath, "/", "\\");
	string::size_type rFirstPos = strFilePath.rfind("\\");
	if (strFilePath.size() != (rFirstPos + 1))   /* ���ת�����·��������\\����ʱ����ĩβ���\\������ĸ�ʽͳһΪD:\\1\\2\\3\\ */
	{
		//������һ���Ƿ�Ϊ�ļ���  
		string strTemp = strFilePath.substr(rFirstPos, strFilePath.size());
		if (string::npos != strTemp.find("."))
		{
			//���һ�������ļ�����  
			strFilePath = strFilePath.substr(0, rFirstPos + 1);
		}
		else
		{
			//���һ�����ļ�������  
			strFilePath += "\\";
		}
	}
	else
	{
		strFilePath += "\\";
	}

	string::size_type startPos(0);
	string::size_type endPos(0);

	while (true)
	{
		if ((endPos = strFilePath.find("\\", startPos)) != string::npos)
		{
			string strFolderPath = strFilePath.substr(0, endPos);
			startPos = endPos + string::size_type(1);

			if (strFolderPath.rfind(":") == (strFolderPath.size() - 1))
			{
				//����ֻ��ϵͳ�̵�·����������磺D:  
				continue;
			}

			struct _stat fileStat = { 0 };
			if (_stat(strFolderPath.c_str(), &fileStat) == 0)
			{
				//�ļ����ڣ��ж���ΪĿ¼�����ļ�  
				if (!(fileStat.st_mode & _S_IFDIR))
				{
					//�����ļ��У��򴴽�ʧ���˳�����  
					return false;
				}
			}
			else
			{
				//�ļ��в����ڣ�����д���  
				if (-1 == _mkdir(strFolderPath.c_str()))
				{
					return false;
				}
			}

			continue;
		}

		break;
	}
	return true;
}
int main(int argc, char * argv[]){
/*  VideoCapture capture;
  //capture.open(0);
  FileStorage fs("parameters.yml",FileStorage::READ);
  if (!fs.isOpened())
  {
	  cout << "fs file failed to open!" << endl;
	  return 1;
  }
  //TLD framework
  TLD tld;
  //Read parameters file
  tld.read(fs.getFirstTopLevelNode());
  CreateFolder(labelXml_path);
  CreateFolder(labelImg_path);
  capture.open(webcam_id);
  //Read options
  read_options(argc,argv,capture,fs);
  //Init camera
  if (!capture.isOpened())
  {
	cout << "capture device failed to open!" << endl;
    return 1;
  }
  //Register mouse callback to draw the bounding box
  namedWindow("TLD",CV_WINDOW_AUTOSIZE);
  setMouseCallback( "TLD", mouseHandler, NULL );
  
  Mat frame,frame_raw;
  Mat last_gray;
  Mat first;
  if (fromfile){
      capture >> frame;
      cvtColor(frame, last_gray, COLOR_RGB2GRAY);
      frame.copyTo(first);
  }else{
      capture.set(CV_CAP_PROP_FRAME_WIDTH,512);//320
      capture.set(CV_CAP_PROP_FRAME_HEIGHT,512);//240
  }

  ///Initialization
GETBOUNDINGBOX:
  while(!gotBB)
  {
    if (!fromfile){
      capture >> frame;
    }
    else
      first.copyTo(frame);
    cvtColor(frame, last_gray, COLOR_RGB2GRAY);
    drawBox(frame,box);
    imshow("TLD", frame);
    if (waitKey(33) == 'q')
	    return 0;
  }
  if (min(box.width,box.height)<(int)fs.getFirstTopLevelNode()["min_win"]){
      cout << "Bounding box too small, try again." << endl;
      gotBB = false;
      goto GETBOUNDINGBOX;
  }
  //Remove callback
  setMouseCallback( "TLD", NULL, NULL );
  printf("Initial Bounding Box = x:%d y:%d h:%d w:%d\n",box.x,box.y,box.width,box.height);
  //Output file
  FILE  *bb_file = fopen("bounding_boxes.txt","w");
  //TLD initialization
  tld.init(last_gray,box,bb_file);

  ///Run-time
  Mat current_gray;
  BoundingBox pbox;
  vector<Point2f> pts1;
  vector<Point2f> pts2;
  bool status=true;
  int frames = 1;
  int detections = 1;
REPEAT:
  while(capture.read(frame)){
    //get frame
    cvtColor(frame, current_gray, COLOR_RGB2GRAY);
    //Process Frame
	tld.processFrame(last_gray, current_gray, pts1, pts2, pbox, status, tl, bb_file);
    //Draw Points
    if (status){
      //drawPoints(frame,pts1);
      //drawPoints(frame,pts2,Scalar(0,255,0));
      drawBox(frame,pbox,Scalar(255,0,0));
	  if (createXML == 1)
		  createXMLandImg(frame, pbox);
      detections++;
    }
    //Display
    imshow("TLD", frame);
    //swap points and images
    swap(last_gray,current_gray);
    pts1.clear();
    pts2.clear();
    frames++;
    printf("Detection rate: %d/%d\n",detections,frames);
    if (cvWaitKey(33) == 'q')
      break;
  }
  if (rep){
    rep = false;
    tl = false;
    fclose(bb_file);
    bb_file = fopen("final_detector.txt","w");
    //capture.set(CV_CAP_PROP_POS_AVI_RATIO,0);
    capture.release();
    capture.open(video);
    goto REPEAT;
  }
  fclose(bb_file);
  return 0;*/
//������ͷ
VideoCapture captrue(0);
//��Ƶд�����
VideoWriter write;
//д����Ƶ�ļ���
string outFlie = "F://fcq.avi";
//���֡�Ŀ��
int w = static_cast<int>(captrue.get(CV_CAP_PROP_FRAME_WIDTH));
int h = static_cast<int>(captrue.get(CV_CAP_PROP_FRAME_HEIGHT));
Size S(w, h);
//���֡��
double r = captrue.get(CV_CAP_PROP_FPS);
//����Ƶ�ļ���׼��д��
write.open(outFlie, -1, r, S, true);
//��ʧ��
if (!captrue.isOpened())
{
	return 1;
}
bool stop = false;
Mat frame;
captrue.set(CV_CAP_PROP_FRAME_WIDTH, 320);
captrue.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
//ѭ��
while (!stop)
{
	//��ȡ֡
	if (!captrue.read(frame))
		break;
	imshow("Video", frame);
	//д���ļ�
	write.write(frame);
	if (waitKey(10) > 0)
	{
		stop = true;
	}
}
//�ͷŶ���
captrue.release();
write.release();
cvDestroyWindow("Video");
return 0;
}
