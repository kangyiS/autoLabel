/*
 * TLD.cpp
 *
 *  Created on: Jun 9, 2011
 *      Author: alantrrs
 */

#include "TLD.h"
#include <stdio.h>
using namespace cv;
using namespace std;

extern long int filename_int;
extern string labelXml_path, labelImg_path;
extern int webcam_id, createXML;

TLD::TLD()
{
}
TLD::TLD(const FileNode& file){
  read(file);
}

void TLD::read(const FileNode& file){
  ///Bounding Box Parameters
  min_win = (int)file["min_win"];
  ///Genarator Parameters
  //initial parameters for positive examples
  patch_size = (int)file["patch_size"];
  num_closest_init = (int)file["num_closest_init"];
  num_warps_init = (int)file["num_warps_init"];
  noise_init = (int)file["noise_init"];
  angle_init = (float)file["angle_init"];
  shift_init = (float)file["shift_init"];
  scale_init = (float)file["scale_init"];
  //update parameters for positive examples
  num_closest_update = (int)file["num_closest_update"];
  num_warps_update = (int)file["num_warps_update"];
  noise_update = (int)file["noise_update"];
  angle_update = (float)file["angle_update"];
  shift_update = (float)file["shift_update"];
  scale_update = (float)file["scale_update"];
  //parameters for negative examples
  bad_overlap = (float)file["overlap"];
  bad_patches = (int)file["num_patches"];
  filename_int = (int)file["filename_int"];
  labelXml_path = (string)file["labelXml_path"];
  labelImg_path = (string)file["labelImg_path"];
  webcam_id = (int)file["webcam_id"];
  createXML = (int)file["createXML"];
  classifier.read(file);
}

void TLD::init(const Mat& frame1,const Rect& box,FILE* bb_file){
  //bb_file = fopen("bounding_boxes.txt","w");
  //Get Bounding Boxes
	//构造bounding box，采用扫描窗口的策略，扫描窗口步长为10%，尺度缩放系数为1.2
	//计算每一个窗口（grid）与鼠标划定区域（目标区域）的位置重叠度（两个box的交集与并集的比例）
	//最终把所有的bounding box都放在grid容器里面，以后就使用grid了
	//把鼠标圈出来的box缩放后的尺寸（共21个）放在scales容器里面
    buildGrid(frame1,box);
    printf("Created %d bounding boxes\n",(int)grid.size());
  ///Preparation
  //allocation
	//frame1是摄像头捕捉的灰度图像
  iisum.create(frame1.rows+1,frame1.cols+1,CV_32F);
  iisqsum.create(frame1.rows+1,frame1.cols+1,CV_64F);
  dconf.reserve(100);
  dbb.reserve(100);
  bbox_step =7;
  //tmp.conf.reserve(grid.size());
  tmp.conf = vector<float>(grid.size());
  tmp.patt = vector<vector<int> >(grid.size(),vector<int>(10,0));
  //tmp.patt.reserve(grid.size());
  dt.bb.reserve(grid.size());
  good_boxes.reserve(grid.size());
  bad_boxes.reserve(grid.size());
  pEx.create(patch_size,patch_size,CV_64F);
  //Init Generator
  generator = PatchGenerator (0,0,noise_init,true,1-scale_init,1+scale_init,-angle_init*CV_PI/180,angle_init*CV_PI/180,-angle_init*CV_PI/180,angle_init*CV_PI/180);
  //把重叠度大于0.6的都放到good_boxes里
  //把重叠度小于0.2的都放到bad_boxes里
  //把good_boxes里的重叠度最高的前十个box合成为一个大的box，并用getBBHull()方法来计算边框
  //这里所说的重叠度，是指在图像坐标系下的位置的重叠度，不是图像特征的重叠度
  getOverlappingBoxes(box,num_closest_init);
  printf("Found %d good boxes, %d bad boxes\n",(int)good_boxes.size(),(int)bad_boxes.size());
  printf("Best Box: %d %d %d %d\n",best_box.x,best_box.y,best_box.width,best_box.height);
  printf("Bounding box hull: %d %d %d %d\n",bbhull.x,bbhull.y,bbhull.width,bbhull.height);
  //Correct Bounding Box
  lastbox=best_box;
  lastconf=1;
  lastvalid=true;
  //Print
  fprintf(bb_file,"%d,%d,%d,%d,%f\n",lastbox.x,lastbox.y,lastbox.br().x,lastbox.br().y,lastconf);
  //Prepare Classifier
  //准备分类器，初始化features容器，并给每种尺度变换，随机分配两个像素点，用于以后的图像比较
  //初始化posteriors，pCounter和nCounter容器
  classifier.prepare(scales);
  ///Generate Data
  // Generate positive data
  //把good_boxes里面重叠度最高的前10个样本当成前景，存入正样本（pX）容器中，并计算每个正样本的特征
  generatePositiveData(frame1,num_warps_init);
  // Set variance threshold
  Scalar stdev, mean;
  //获得best_box的均值和标准差
  meanStdDev(frame1(best_box),mean,stdev);
  //计算frame1的积分图像，积分图像保存在iisum里，平方积分图像保存在iisqsum里
  //利用积分图像，可以计算在某象素的上－右方的或者旋转的矩形区域中进行求和、求均值以及标准方差的计算，  
  //并且保证运算的复杂度为O(1)。    
  integral(frame1,iisum,iisqsum);
  var = (float)pow(stdev.val[0],2)*0.5f; //getVar(best_box,iisum,iisqsum);
  cout << "variance: " << var << endl;
  //check variance
  double vr =  getVar(best_box,iisum,iisqsum)*0.5;
  cout << "check variance: " << vr << endl;
  // Generate negative data
  //把方差大于0.25倍的best_box的方差的grid当成背景，存入负样本（nX）容器中，并计算每个负样本的特征
  //从bad_boxes里面取出一定数量的grid，并归一化为15*15的patch，存入nEx容器里面
  generateNegativeData(frame1);
  //Split Negative Ferns into Training and Testing sets (they are already shuffled)
  int half = (int)(nX.size()*0.5f);
  //将负样本的后一半作为测试集
  nXT.assign(nX.begin()+half,nX.end());
  //将负样本的前一半作为训练集                  这倒像是随机来的，不需要按照特征、方差、重叠度什么的分一下训练集和测试集吗？？？？？？？？？？
  nX.resize(half);
  ///Split Negative NN Examples into Training and Testing sets
  half = (int)(nEx.size()*0.5f);
  nExT.assign(nEx.begin()+half,nEx.end());
  nEx.resize(half);
  //Merge Negative Data with Positive Data and shuffle it
  //将正样本和负样本打乱顺序，然后合并，保存在ferns_data容器里
  vector<pair<vector<int>,int> > ferns_data(nX.size()+pX.size());
  vector<int> idx = index_shuffle(0,(int)(ferns_data.size()));
  int a=0;
  for (int i=0;i<pX.size();i++){
      ferns_data[idx[a]] = pX[i];
      a++;
  }
  for (int i=0;i<nX.size();i++){
      ferns_data[idx[a]] = nX[i];
      a++;
  }
  //Data already have been shuffled, just putting it in the same vector
  vector<cv::Mat> nn_data(nEx.size()+1);
  nn_data[0] = pEx;
  for (int i=0;i<nEx.size();i++){
      nn_data[i+1]= nEx[i];
  }
  ///Training
  classifier.trainF(ferns_data,2); //bootstrap = 2
  classifier.trainNN(nn_data);
  ///Threshold Evaluation on testing sets
  classifier.evaluateTh(nXT,nExT);
}

/* Generate Positive data
 * Inputs:
 * - good_boxes (bbP)
 * - best_box (bbP0)
 * - frame (im0)
 * Outputs:
 * - Positive fern features (pX)
 * - Positive NN examples (pEx)
 */
void TLD::generatePositiveData(const Mat& frame, int num_warps){
  Scalar mean;
  Scalar stdev;
  //把best_box图像归一化为均值为零的15*15的patch，存在pEx里面
  //计算pEx每个通道的像素的均值（不是零，是原始的均值）和标准差
  getPattern(frame(best_box),pEx,mean,stdev);
  //Get Fern features on warped patches
  Mat img;
  Mat warped;
  GaussianBlur(frame,img,Size(9,9),1.5);//高斯滤波，把frame滤波，结果存在img里面
  //在img里面截取bbhull信息（bbhull是包含了大小和位置的矩形框），存在warped里面
  warped = img(bbhull);
  RNG& rng = theRNG();
  //计算bbhull矩形框中心的坐标
  Point2f pt(bbhull.x+(bbhull.width-1)*0.5f,bbhull.y+(bbhull.height-1)*0.5f);
  //定义一个fern容器，初始化容器大小为classifier.getNumStructs()
  //fern是蕨类
  vector<int> fern(classifier.getNumStructs());
  pX.clear();
  Mat patch;
  //pX是正样本数量，下面在给pX重新分配空间
  if (pX.capacity()<num_warps*good_boxes.size())
    pX.reserve(num_warps*good_boxes.size());
  int idx;
  for (int i=0;i<num_warps;i++){
     if (i>0)
       generator(frame,pt,warped,bbhull.size(),rng);
       for (int b=0;b<good_boxes.size();b++){
         idx=good_boxes[b];//good_boxes里面保存的是grid的索引，现在good_boxes只有十个了，取了重叠度最高的前十个
		 patch = img(grid[idx]);//把img里面的grid[idx]这个区域的图像抠出来
		 //将patch的第grid[idx].sidx种尺度变换的特征保存在fern容器里面，每种尺度变换有10*13个特征
         classifier.getFeatures(patch,grid[idx].sidx,fern);
         pX.push_back(make_pair(fern,1));//保存正样本特征
     }
  }
  printf("Positive examples generated: ferns:%d NN:1\n",(int)pX.size());
}

void TLD::getPattern(const Mat& img, Mat& pattern,Scalar& mean,Scalar& stdev){
  //Output: resized Zero-Mean patch
	//把img转换成15*15的pattern
  resize(img,pattern,Size(patch_size,patch_size));
  //计算pattern每个通道的像素的均值和标准差
  meanStdDev(pattern,mean,stdev);
  pattern.convertTo(pattern,CV_32F);
  //把pattern的均值归零
  pattern = pattern-mean.val[0];
}

void TLD::generateNegativeData(const Mat& frame){
/* Inputs:
 * - Image
 * - bad_boxes (Boxes far from the bounding box)
 * - variance (pEx variance)
 * Outputs
 * - Negative fern features (nX)
 * - Negative NN examples (nEx)
 */
	//打乱bad_boxes容器的顺序，为什么啊……
  random_shuffle(bad_boxes.begin(),bad_boxes.end());//Random shuffle bad_boxes indexes
  int idx;
  //Get Fern Features of the boxes with big variance (calculated using integral images)
  int a=0;
  //int num = std::min((int)bad_boxes.size(),(int)bad_patches*100); //limits the size of bad_boxes to try
  printf("negative data generation started.\n");
  vector<int> fern(classifier.getNumStructs());
  nX.reserve(bad_boxes.size());
  Mat patch;
  for (int j=0;j<bad_boxes.size();j++){
      idx = bad_boxes[j];
	  //var是best_box的方差的50%
	  //把方差小于0.25倍best_box的给剔除掉，剩下的计算特征，放在负样本（nX）容器里
          if (getVar(grid[idx],iisum,iisqsum)<var*0.5f)
            continue;
      patch =  frame(grid[idx]);
	  classifier.getFeatures(patch,grid[idx].sidx,fern);
      nX.push_back(make_pair(fern,0));//存入负样本
      a++;
  }
  printf("Negative examples generated: ferns: %d ",a);
  //random_shuffle(bad_boxes.begin(),bad_boxes.begin()+bad_patches);//Randomly selects 'bad_patches' and get the patterns for NN;
  Scalar dum1, dum2;
  //bad_patches是从参数文件里面读出来的
  nEx=vector<Mat>((unsigned __int64)bad_patches);
  for (int i=0;i<bad_patches;i++){
      idx=bad_boxes[i];
	  patch = frame(grid[idx]);
	  //这里的nEx是许多数据的集合，是一个容器，而前面的pEx就是用best_box计算出来的，就是一个Mat类型的图像
      getPattern(patch,nEx[i],dum1,dum2);
  }
  printf("NN: %d\n",(int)nEx.size());
}

double TLD::getVar(const BoundingBox& box,const Mat& sum,const Mat& sqsum){
  double brs = sum.at<int>(box.y+box.height,box.x+box.width);
  double bls = sum.at<int>(box.y+box.height,box.x);
  double trs = sum.at<int>(box.y,box.x+box.width);
  double tls = sum.at<int>(box.y,box.x);
  double brsq = sqsum.at<double>(box.y+box.height,box.x+box.width);
  double blsq = sqsum.at<double>(box.y+box.height,box.x);
  double trsq = sqsum.at<double>(box.y,box.x+box.width);
  double tlsq = sqsum.at<double>(box.y,box.x);
  double mean = (brs+tls-trs-bls)/((double)box.area());
  double sqmean = (brsq+tlsq-trsq-blsq)/((double)box.area());
  return sqmean-mean*mean;
}

void TLD::processFrame(const cv::Mat& img1,const cv::Mat& img2,vector<Point2f>& points1,vector<Point2f>& points2,BoundingBox& bbnext,bool& lastboxfound, bool tl, FILE* bb_file){
  vector<BoundingBox> cbb;
  vector<float> cconf;
  int confident_detections=0;
  int didx; //detection index
  ///Track
  if(lastboxfound && tl){
	  //利用LK光流法进行跟踪并预测当前帧中，img2的位置
      track(img1,img2,points1,points2);
  }
  else{
      tracked = false;
  }
  ///Detect
  //依次经过三级分类器（方差分类器，集合分类器，最邻近分类器）
  //img2是当前摄像头采集的灰度图像
  detect(img2);
  ///Integration
  if (tracked){
      bbnext=tbb;//tbb是根据上一帧的bounding box计算出来的当前帧的bounding box
      lastconf=tconf;
      lastvalid=tvalid;
      printf("Tracked\n");
      if(detected) {                                               //   if Detected
		  //如果检测到了，就计算聚类
          clusterConf(dbb,dconf,cbb,cconf);                       //   cluster detections
          printf("Found %d clusters\n",(int)cbb.size());
          for (int i=0;i<cbb.size();i++){
			  //如果通过追踪得到的bounding box和通过聚类器得到的bounding box有不到0.5的重叠度，
			  //并且聚类器的保守相似度更高，就记录下来
              if (bbOverlap(tbb,cbb[i])<0.5 && cconf[i]>tconf){  //  Get index of a clusters that is far from tracker and are more confident than the tracker
                  confident_detections++;
                  didx=i; //detection index
              }
          }
		  //如果满足了上述条件，就初始化追踪器，并且相信通过聚类器得到的bounding box
          if (confident_detections==1){                                //if there is ONE such a cluster, re-initialize the tracker
              printf("Found a better match..reinitializing tracking\n");
              bbnext=cbb[didx];
              lastconf=cconf[didx];
              lastvalid=false;
          }
          else {
              printf("%d confident cluster was found\n",confident_detections);
              int cx=0,cy=0,cw=0,ch=0;
              int close_detections=0;
              for (int i=0;i<dbb.size();i++){
				  //如果通过track得到的bounding box和通过detect得到的bounding box有大于0.7的相似度
				  //就将它们进行加权平均，计算得到一个合理的bounding box
                  if(bbOverlap(tbb,dbb[i])>0.7){                     // Get mean of close detections
                      cx += dbb[i].x;
                      cy +=dbb[i].y;
                      cw += dbb[i].width;
                      ch += dbb[i].height;
                      close_detections++;
                      printf("weighted detection: %d %d %d %d\n",dbb[i].x,dbb[i].y,dbb[i].width,dbb[i].height);
                  }
              }
              if (close_detections>0){
                  bbnext.x = cvRound((float)(10*tbb.x+cx)/(float)(10+close_detections));   // weighted average trackers trajectory with the close detections
                  bbnext.y = cvRound((float)(10*tbb.y+cy)/(float)(10+close_detections));
                  bbnext.width = cvRound((float)(10*tbb.width+cw)/(float)(10+close_detections));
                  bbnext.height =  cvRound((float)(10*tbb.height+ch)/(float)(10+close_detections));
                  printf("Tracker bb: %d %d %d %d\n",tbb.x,tbb.y,tbb.width,tbb.height);
                  printf("Average bb: %d %d %d %d\n",bbnext.x,bbnext.y,bbnext.width,bbnext.height);
                  printf("Weighting %d close detection(s) with tracker..\n",close_detections);
              }
              else{
                printf("%d close detections were found\n",close_detections);

              }
          }
      }
  }
  else{                                       //   If NOT tracking
      printf("Not tracking..\n");
      lastboxfound = false;
      lastvalid = false;
	  //如果没追踪成功，但是检测成功了，那就用检测得到的bounding box
      if(detected){                           //  and detector is defined
		  //dbb是经过三级分类器后留下来的grid，dconf是对应的保守相似度
		  //将相似度大于0.5的grid组合成一个更大的bounding box
		  //经过这个函数，能组合出好多bounding box，最后保存在cbb容器里面，cconf是对应的保守相似度
          clusterConf(dbb,dconf,cbb,cconf);   //  cluster detections
          printf("Found %d clusters\n",(int)cbb.size());
          if (cconf.size()==1){
              bbnext=cbb[0];
              lastconf=cconf[0];
              printf("Confident detection..reinitializing tracker\n");
              lastboxfound = true;
          }
      }
  }
  //计算得到了当前最合理的bounding box
  lastbox=bbnext;
  if (lastboxfound)
    fprintf(bb_file,"%d,%d,%d,%d,%f\n",lastbox.x,lastbox.y,lastbox.br().x,lastbox.br().y,lastconf);
  else
    fprintf(bb_file,"NaN,NaN,NaN,NaN,NaN\n");
  //img2是当前摄像头采集的灰度图像
  if (lastvalid && tl)
    learn(img2);
}


void TLD::track(const Mat& img1, const Mat& img2,vector<Point2f>& points1,vector<Point2f>& points2){
  /*Inputs:
   * -current frame(img2), last frame(img1), last Bbox(bbox_f[0]).
   *Outputs:
   *- Confidence(tconf), Predicted bounding box(tbb),Validity(tvalid), points2 (for display purposes only)
   */
  //Generate points
	//在lastbox中均匀取100个点，存在points1容器里面
  bbPoints(points1,lastbox);
  if (points1.size()<1){
      printf("BB= %d %d %d %d, Points not generated\n",lastbox.x,lastbox.y,lastbox.width,lastbox.height);
      tvalid=false;
      tracked=false;
      return;
  }
  vector<Point2f> points = points1;
  //Frame-to-frame tracking with forward-backward error cheking
  //利用LK光流法进行跟踪，并将points和points2里面的特征点进行匹配与筛选
  tracked = tracker.trackf2f(img1,img2,points,points2);
  if (tracked){
      //Bounding box prediction
	  //根据前后两帧的特征点和前一帧的图像，计算当前bounding box的位置，存在tbb里面
      bbPredict(points,points2,lastbox,tbb);
	  //如果FB_error的中值大于10个像素点（经验值）或者计算的bounding box超过了图像的边界，则认为跟踪失败
      if (tracker.getFB()>10 || tbb.x>img2.cols ||  tbb.y>img2.rows || tbb.br().x < 1 || tbb.br().y <1){
          tvalid =false; //too unstable prediction or bounding box out of image
          tracked = false;
          printf("Too unstable predictions FB error=%f\n",tracker.getFB());
          return;
      }
      //Estimate Confidence and Validity
      Mat pattern;
      Scalar mean, stdev;
      BoundingBox bb;
      bb.x = max(tbb.x,0);
      bb.y = max(tbb.y,0);
      bb.width = min(min(img2.cols-tbb.x,tbb.width),min(tbb.width,tbb.br().x));
      bb.height = min(min(img2.rows-tbb.y,tbb.height),min(tbb.height,tbb.br().y));
	  //归一化img2(bb)对应的patch的size（放缩至patch_size = 15*15），存入pattern
      getPattern(img2(bb),pattern,mean,stdev);
      vector<int> isin;
      float dummy;
	  //计算图像片pattern到在线模型M的保守相似度 
      classifier.NNConf(pattern,isin,dummy,tconf); //Conservative Similarity
      tvalid = lastvalid;
	  //保守相似度大于阈值，则认为跟踪有效
      if (tconf>classifier.thr_nn_valid){
          tvalid =true;
      }
  }
  else
    printf("No points tracked\n");

}
/*
*在bb图像里，均匀取了10*10=100个点，存在了points容器里面
*/
void TLD::bbPoints(vector<cv::Point2f>& points,const BoundingBox& bb){
  int max_pts=10;
  int margin_h=0;
  int margin_v=0;
  int stepx = (int)ceil((double)(bb.width-2*margin_h)/max_pts);
  int stepy = (int)ceil((double)(bb.height - 2 * margin_v) / max_pts);
  for (int y=bb.y+margin_v;y<bb.y+bb.height-margin_v;y+=stepy){
      for (int x=bb.x+margin_h;x<bb.x+bb.width-margin_h;x+=stepx){
          points.push_back(Point2f((float)x,(float)y));
      }
  }
}

void TLD::bbPredict(const vector<cv::Point2f>& points1,const vector<cv::Point2f>& points2,
                    const BoundingBox& bb1,BoundingBox& bb2)    {
  int npoints = (int)points1.size();
  vector<float> xoff(npoints);
  vector<float> yoff(npoints);
  printf("tracked points : %d\n",npoints);
  //计算两帧之间匹配点的位移
  for (int i=0;i<npoints;i++){
      xoff[i]=points2[i].x-points1[i].x;
      yoff[i]=points2[i].y-points1[i].y;
  }
  float dx = median(xoff);//计算位移的中值
  float dy = median(yoff);
  float s;
  if (npoints>1){
      vector<float> d;
      d.reserve(npoints*(npoints-1)/2);//等差数列求和1+2+3+...+(npoints-1)
      for (int i=0;i<npoints;i++){
          for (int j=i+1;j<npoints;j++){
			  //计算当前特征点每两个点之间的距离和上一帧特征点每两个点之间的距离的比值
              d.push_back((float)(norm(points2[i]-points2[j])/norm(points1[i]-points1[j])));
          }
      }
      s = median(d);//计算这个比值的中值
  }
  else {
      s = 1.0;
  }
  float s1 = 0.5f*(s-1)*bb1.width;
  float s2 = 0.5f*(s-1)*bb1.height;
  printf("s= %f s1= %f s2= %f \n",s,s1,s2);
  //计算当前bounding box的位置
  bb2.x = (int)round( bb1.x + dx -s1);
  bb2.y = (int)round( bb1.y + dy -s2);
  bb2.width = (int)round(bb1.width*s);
  bb2.height = (int)round(bb1.height*s);
  printf("predicted bb: %d %d %d %d\n",bb2.x,bb2.y,bb2.br().x,bb2.br().y);
}

void TLD::detect(const cv::Mat& frame){
  //cleaning
  dbb.clear();
  dconf.clear();
  dt.bb.clear();
  double t = (double)getTickCount();
  Mat img(frame.rows,frame.cols,CV_8U);
  //计算积分图
  integral(frame,iisum,iisqsum);
  //将frame进行高斯滤波，存在img里面
  GaussianBlur(frame,img,Size(9,9),1.5);
  int numtrees = classifier.getNumStructs();
  float fern_th = classifier.getFernTh();
  vector <int> ferns(10);
  float conf;
  int a=0;
  Mat patch;
  for (int i=0;i<grid.size();i++){//FIXME: BottleNeck
	  //var是best_box的方差的一半，首先进入的是方差分类器
	  //计算grid[i]的方差，利用上面求的iisum和iisqsum可以减少计算时间
	  //这是第一个分类器，要计算所有的grid，所以一定要用iisum和iisqsum来压缩计算时间，保证实时性
      if (getVar(grid[i],iisum,iisqsum)>=var){
          a++;
		  //方差分类器通过了，进入集合分类器
		  patch = img(grid[i]);
		  //得到patch的特征
          classifier.getFeatures(patch,grid[i].sidx,ferns);
		  //计算10个特征的后验概率的累加值
          conf = classifier.measure_forest(ferns);
          tmp.conf[i]=conf;
          tmp.patt[i]=ferns;
		  //后验概率大于阈值，则认为拥有前景目标
          if (conf>numtrees*fern_th){
              dt.bb.push_back(i);
          }
      }
      else
        tmp.conf[i]=0.0;
  }
  int detections = (int)(dt.bb.size());
  printf("%d Bounding boxes passed the variance filter\n",a);
  printf("%d Initial detection from Fern Classifier\n",detections);
  //如果有超过100个grid满足条件，只要最好的前100个
  if (detections>100){
      nth_element(dt.bb.begin(),dt.bb.begin()+100,dt.bb.end(),CComparator(tmp.conf));
      dt.bb.resize(100);
      detections=100;
  }
//  for (int i=0;i<detections;i++){
//        drawBox(img,grid[dt.bb[i]]);
//    }
//  imshow("detections",img);
  if (detections==0){
        detected=false;
        return;
      }
  printf("Fern detector made %d detections ",detections);
  t=(double)getTickCount()-t;
  printf("in %gms\n", t*1000/getTickFrequency());
                                                                       //  Initialize detection structure
  dt.patt = vector<vector<int> >(detections,vector<int>(10,0));        //  Corresponding codes of the Ensemble Classifier
  dt.conf1 = vector<float>(detections);                                //  Relative Similarity (for final nearest neighbour classifier)
  dt.conf2 =vector<float>(detections);                                 //  Conservative Similarity (for integration with tracker)
  dt.isin = vector<vector<int> >(detections,vector<int>(3,-1));        //  Detected (isin=1) or rejected (isin=0) by nearest neighbour classifier
  dt.patch = vector<Mat>(detections,Mat(patch_size,patch_size,CV_32F));//  Corresponding patches
  int idx;
  Scalar mean, stdev;
  float nn_th = classifier.getNNTh();//获取最近邻分类器的阈值
  for (int i=0;i<detections;i++){                                         //  for every remaining detection
      idx=dt.bb[i];                                                       //  Get the detected bounding box index
	  patch = frame(grid[idx]);
      getPattern(patch,dt.patch[i],mean,stdev);                //  Get pattern within bounding box
	  //dt.conf1是相关相似度，dt.conf2是保守相似度
      classifier.NNConf(dt.patch[i],dt.isin[i],dt.conf1[i],dt.conf2[i]);  //  Evaluate nearest neighbour classifier
      dt.patt[i]=tmp.patt[idx];
      //printf("Testing feature %d, conf:%f isin:(%d|%d|%d)\n",i,dt.conf1[i],dt.isin[i][0],dt.isin[i][1],dt.isin[i][2]);
      if (dt.conf1[i]>nn_th){                                               //  idx = dt.conf1 > tld.model.thr_nn; % get all indexes that made it through the nearest neighbour
          dbb.push_back(grid[idx]);                                         //  BB    = dt.bb(:,idx); % bounding boxes
          dconf.push_back(dt.conf2[i]);                                     //  Conf  = dt.conf2(:,idx); % conservative confidences
      }
  }                                                                         //  end
  if (dbb.size()>0){
      printf("Found %d NN matches\n",(int)dbb.size());
      detected=true;
  }
  else{
      printf("No NN matches found.\n");
      detected=false;
  }
}

void TLD::evaluate(){
}

void TLD::learn(const Mat& img){
  printf("[Learning] ");
  ///Check consistency
  BoundingBox bb;
  bb.x = max(lastbox.x,0);
  bb.y = max(lastbox.y,0);
  bb.width = min(min(img.cols-lastbox.x,lastbox.width),min(lastbox.width,lastbox.br().x));
  bb.height = min(min(img.rows-lastbox.y,lastbox.height),min(lastbox.height,lastbox.br().y));
  Scalar mean, stdev;
  Mat pattern;
  getPattern(img(bb),pattern,mean,stdev);
  vector<int> isin;
  float dummy, conf;
  //将pattern和正样本、负样本进行比较，计算得到的conf是相关相似度，dummy是保守相似度
  classifier.NNConf(pattern,isin,conf,dummy);
  if (conf<0.5) {
      printf("Fast change..not training\n");
      lastvalid =false;
      return;
  }
  //方差太小
  if (pow(stdev.val[0],2)<var){
      printf("Low variance..not training\n");
      lastvalid=false;
      return;
  }
  //被识别为负样本
  if(isin[2]==1){
      printf("Patch in negative data..not traing");
      lastvalid=false;
      return;
  }
/// Data generation
  //计算每个grid的重叠度
  //lastbox是当前这次循环，经过detect计算后的最好的目标框
  for (int i=0;i<grid.size();i++){
      grid[i].overlap = bbOverlap(lastbox,grid[i]);
  }
  vector<pair<vector<int>,int> > fern_examples;
  good_boxes.clear();
  bad_boxes.clear();
  //重新装载good_boxes和bad_boxes
  getOverlappingBoxes(lastbox,num_closest_update);
  //获取正样本
  if (good_boxes.size()>0)
    generatePositiveData(img,num_warps_update);
  else{
    lastvalid = false;
    printf("No good boxes..Not training");
    return;
  }
  //pX里面保存的都是正样本的13位特征（fern，1）
  fern_examples.reserve(pX.size()+bad_boxes.size());
  fern_examples.assign(pX.begin(),pX.end());
  int idx;
  //这一步是要保存大量的正样本，用于以后的学习，训练分类器
  for (int i=0;i<bad_boxes.size();i++){
      idx=bad_boxes[i];
	  //10个后验概率的累加值大于1
      if (tmp.conf[idx]>=1){
		  //保存当前的负样本数据，这些数据有可能在上次循环中后验概率比较大，
		  //那这就不合理，需要更新这些数据的后验概率
          fern_examples.push_back(make_pair(tmp.patt[idx],0));
      }
  }
  vector<Mat> nn_examples;
  //dt.bb是经过了集合分类器，即将进入最近邻分类器的grid，用的是这次循环的grid
  nn_examples.reserve(dt.bb.size()+1);
  nn_examples.push_back(pEx);
  for (int i=0;i<dt.bb.size();i++){
      idx = dt.bb[i];
      if (bbOverlap(lastbox,grid[idx]) < bad_overlap)
		  //把grid里面重合度太低的放到nn_examples容器里面
        nn_examples.push_back(dt.patch[i]);
  }
  /// Classifiers update
  //训练集合分类器
  classifier.trainF(fern_examples,2);
  //训练最近邻分类器
  classifier.trainNN(nn_examples);
  classifier.show();
}

void TLD::buildGrid(const cv::Mat& img, const cv::Rect& box){
	//平移步长
  const float SHIFT = 0.1f;
  //缩放系数
  const float SCALES[] = {0.16151f,0.19381f,0.23257f,0.27908f,0.33490f,0.40188f,0.48225f,
                          0.57870f,0.69444f,0.83333f,1.0f,1.20000f,1.44000f,1.72800f,
                          2.07360f,2.48832f,2.98598f,3.58318f,4.29982f,5.15978f,6.19174f};
  int width, height, min_bb_side;
  //Rect bbox;
  BoundingBox bbox;
  Size scale;
  int sc=0;
  for (int s=0;s<21;s++){
    width = (int)round(box.width*SCALES[s]);
    height = (int)round(box.height*SCALES[s]);
    min_bb_side = min(height,width);
    if (min_bb_side < min_win || width > img.cols || height > img.rows)
      continue;
    scale.width = width;
    scale.height = height;
	//将缩放后的图像存在scales容器里面
    scales.push_back(scale);
    for (int y=1;y<img.rows-height;y+=(int)round(SHIFT*min_bb_side)){
      for (int x=1;x<img.cols-width;x+=(int)round(SHIFT*min_bb_side)){
        bbox.x = x;
        bbox.y = y;
        bbox.width = width;
        bbox.height = height;
		//计算bbox和用鼠标圈出来的目标（box）的重叠度
        bbox.overlap = bbOverlap(bbox,BoundingBox(box));
        bbox.sidx = sc;
		//根据用鼠标圈出来的图像，按照一定的步长和缩放系数
		//将整个摄像头捕捉到的图像分割成若干个小的矩形区域
		//小的矩形区域相互重叠，都存放在grid容器里面
        grid.push_back(bbox);
      }
    }
    sc++;
  }
}

float TLD::bbOverlap(const BoundingBox& box1,const BoundingBox& box2){
  if (box1.x > box2.x+box2.width) { return 0.0; }
  if (box1.y > box2.y+box2.height) { return 0.0; }
  if (box1.x+box1.width < box2.x) { return 0.0; }
  if (box1.y+box1.height < box2.y) { return 0.0; }

  float colInt =  (float)min(box1.x+box1.width,box2.x+box2.width) - max(box1.x, box2.x);
  float rowInt =  (float)min(box1.y+box1.height,box2.y+box2.height) - max(box1.y,box2.y);

  float intersection = colInt * rowInt;
  float area1 = (int)(box1.width*box1.height);
  float area2 = (int)(box2.width*box2.height);
  return intersection / (area1 + area2 - intersection);
}

void TLD::getOverlappingBoxes(const cv::Rect& box1,int num_closest){
  float max_overlap = 0;
  for (int i=0;i<grid.size();i++){
      if (grid[i].overlap > max_overlap) {
          max_overlap = grid[i].overlap;
          best_box = grid[i];
      }
	  //good_boxes容器里面存的是索引
      if (grid[i].overlap > 0.6){
          good_boxes.push_back(i);
      }
      else if (grid[i].overlap < bad_overlap){
          bad_boxes.push_back(i);
      }
  }
  //Get the best num_closest (10) boxes and puts them in good_boxes
  //取重叠度最高的十个，并且重新分配good_boxes容器大小
  if (good_boxes.size()>num_closest){
    std::nth_element(good_boxes.begin(),good_boxes.begin()+num_closest,good_boxes.end(),OComparator(grid));
    good_boxes.resize(num_closest);
  }
  //获取good_boxes容器的壳（这时good_boxes容器只有10个成员了，获取他们组成的图像的边框），存在bbhull里面
  getBBHull();
}

void TLD::getBBHull(){
  int x1=INT_MAX, x2=0;
  int y1=INT_MAX, y2=0;
  int idx;
  for (int i=0;i<good_boxes.size();i++){
      idx= good_boxes[i];
      x1=min(grid[idx].x,x1);
      y1=min(grid[idx].y,y1);
      x2=max(grid[idx].x+grid[idx].width,x2);
      y2=max(grid[idx].y+grid[idx].height,y2);
  }
  bbhull.x = x1;
  bbhull.y = y1;
  bbhull.width = x2-x1;
  bbhull.height = y2 -y1;
}

bool bbcomp(const BoundingBox& b1,const BoundingBox& b2){
  TLD t;
    if (t.bbOverlap(b1,b2)<0.5)
      return false;
    else
      return true;
}
int TLD::clusterBB(const vector<BoundingBox>& dbb,vector<int>& indexes){
  //FIXME: Conditional jump or move depends on uninitialised value(s)
  const int c = (int)(dbb.size());
  //1. Build proximity matrix
  Mat D(c,c,CV_32F);
  float d;
  for (int i=0;i<c;i++){
      for (int j=i+1;j<c;j++){
        d = 1-bbOverlap(dbb[i],dbb[j]);
        D.at<float>(i,j) = d;
        D.at<float>(j,i) = d;
      }
  }
  //2. Initialize disjoint clustering
 float *L=new float[c-1]; //Level
 int **nodes=new int*[c-1];
 nodes[0] = new int[2];
 nodes[1] = new int[1];
 int *belongs=new int[c];
 int m=c;
 for (int i=0;i<c;i++){
    belongs[i]=i;
 }
 for (int it=0;it<c-1;it++){
 //3. Find nearest neighbor
     float min_d = 1;
     int node_a, node_b;
     for (int i=0;i<D.rows;i++){
         for (int j=i+1;j<D.cols;j++){
             if (D.at<float>(i,j)<min_d && belongs[i]!=belongs[j]){
                 min_d = D.at<float>(i,j);
                 node_a = i;
                 node_b = j;
             }
         }
     }
     if (min_d>0.5){
         int max_idx =0;
         bool visited;
         for (int j=0;j<c;j++){
             visited = false;
             for(int i=0;i<2*c-1;i++){
                 if (belongs[j]==i){
                     indexes[j]=max_idx;
                     visited = true;
                 }
             }
             if (visited)
               max_idx++;
         }
		 delete[]L;
		 delete[]nodes[0];
		 delete[]nodes[1];
		 delete[]nodes;
		 delete[]belongs;
         return max_idx;
     }

 //4. Merge clusters and assign level
     L[m]=min_d;
     nodes[it][0] = belongs[node_a];
     nodes[it][1] = belongs[node_b];
     for (int k=0;k<c;k++){
         if (belongs[k]==belongs[node_a] || belongs[k]==belongs[node_b])
           belongs[k]=m;
     }
     m++;
 }
 delete[]L;
 delete[]nodes[0];
 delete[]nodes[1];
 delete[]nodes;
 delete[]belongs;
 return 1;

}

void TLD::clusterConf(const vector<BoundingBox>& dbb,const vector<float>& dconf,vector<BoundingBox>& cbb,vector<float>& cconf){
  int numbb =(int)(dbb.size());
  vector<int> T;
  float space_thr = 0.5;
  int c=1;
  switch (numbb){
  case 1://如果只检测到一个bounding box，则没问题
    cbb=vector<BoundingBox>(1,dbb[0]);
    cconf=vector<float>(1,dconf[0]);
    return;
    break;
  case 2://如果检测到了两个bounding box
    T =vector<int>(2,0);
    if (1-bbOverlap(dbb[0],dbb[1])>space_thr){//如果这两个bounding box的重叠度小于0.5，就记录下来
      T[1]=1;
      c=2;
    }
    break;
  default://如果检测到多个bounding box，就把它们分成两类，
    T = vector<int>(numbb,0);
    c = partition(dbb,T,(*bbcomp));//记录重叠度小于0.5的一类的个数
    //c = clusterBB(dbb,T);
    break;
  }
  cconf=vector<float>(c);
  cbb=vector<BoundingBox>(c);
  printf("Cluster indexes: ");
  BoundingBox bx;
  for (int i=0;i<c;i++){//c是不同类别的box的个数
      float cnf=0;
      int N=0,mx=0,my=0,mw=0,mh=0;
      for (int j=0;j<T.size();j++){//T是原始的box的个数
          if (T[j]==i){//将聚类为同一个类别的box的坐标和大小进行累加 
              printf("%d ",i);
              cnf=cnf+dconf[j];
              mx=mx+dbb[j].x;
              my=my+dbb[j].y;
              mw=mw+dbb[j].width;
              mh=mh+dbb[j].height;
              N++;
          }
      }
      if (N>0){//然后求该类的box的坐标和大小的平均值，将平均值作为该类的box的代表
          cconf[i]=cnf/N;
          bx.x=cvRound(mx/N);
          bx.y=cvRound(my/N);
          bx.width=cvRound(mw/N);
          bx.height=cvRound(mh/N);
          cbb[i]=bx;//返回的是聚类，每一个类都有一个代表的bounding box 
      }
  }
  printf("\n");
}

