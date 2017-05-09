//#include "WrongWordAnalysis.h"
#include <iostream>

#include <stdio.h>
#include <vector>
#include "../GMMCodebook/GMMProbBatchCalc.h"
#include "../GMMCodebook/GMMCodebookSet.h"
#include "../CommonLib/FileFormat/FeatureFileSet.h"
#include "../CommonLib/Dict/WordDict.h"
//#include "boost/filesystem.hpp"
#include "../CommonLib/ReadConfig/RecParam.h"
#include "../OneBestRecAlgorithm/SimpleSpeechRec.h"
//#include "SRecTokenFactory.h"
#include <windows.h>
#include "../CommonLib/Droper.h"
#include "../CommonLib/Cluster.h"

//#define NBest 20
//int main2(int argc,char *argv[]){
//	
//	string outFileName,Configfilename,statfileName;
//	if (argc<3)
//	{
//		outFileName = "WrongRes.txt";
//		statfileName = "ErrorRes.txt";
//		if(argc<2)
//			Configfilename= "stat_config.txt";
//		else
//			Configfilename=argv[1];
//	}
//	else if(argc==3)
//	{
//		outFileName = argv[2];
//		Configfilename = argv[1];		
//	}	
//	else
//	{
//		printf("error input!");
//		exit(-1);
//	}
//	WrongWordAnalysis my(Configfilename,outFileName,statfileName,"D:/MyCodes/DDBHMMTasks/dict/worddict.txt");
//	
//	/*ifstream fid(Configfilename.c_str(),ios::in);
//	if (!fid)
//	{
//		printf("can't open config file %s",Configfilename);
//		exit(-1);
//	}
//	string ProcessNumStr;
//	getline(fid,ProcessNumStr);
//	trimStr(ProcessNumStr);
//	int ProcessNum=stoi(ProcessNumStr);
//	vector<string>recList(ProcessNum);
//	vector<string>ansList(ProcessNum);
//	for (int i=0;i<ProcessNum;i++)
//	{
//		string inString;
//		getline(fid,inString);
//		trimStr(inString);
//		stringstream ss(inString);
//		ss>>recList[i]>>ansList[i];
//	}
//	int outWrongList[1254]={0};
//	vector<vector<string>>outWrongIdList(1254);
//	FILE * outFid=fopen(outFileName.c_str(),"w+");
//	for (int i=0;i<ProcessNum;i++)
//	{
//		WrongWordAnalysis prf(recList[i],ansList[i],"D:/MyCodes/DDBHMMTasks/dict/worddict.txt");
//		fprintf(outFid,"-----------------------------------\n");
//		fprintf(outFid,"No.%d rec File:\n",i);
//		prf.printWrongRecResToFile(outFid);
//		prf.SaveWrongList(outWrongList,outWrongIdList);
//	}
//	printWrongListToFile(outFid,outWrongList,outWrongIdList);
//	fclose(outFid);*/
//	return 0;
//}


int main(int argc,char *argv[]) {

	char *recg;
	string cltDirname="";
	if(argc < 2 || argc > 3) {
		printf("usage:program_name config_file [basedir]\n");
		exit(-1);
	} else {
		recg = argv[1];
		if (argc == 3) {
			cltDirname = string(argv[2]);
			//etCurrentDirectory(argv[2]);
		}
	}

	RecParam rparam(recg);
	vector<RSpeechFile> inputs = rparam.getRecFiles();
	vector<int>sumRes(150);
	int recFileNum = rparam.getRecNum();
	for (int i = 0; i < recFileNum; i++) {

		RSpeechFile input = inputs.at(i);
		FeatureFileSet fs(input.getFeatureFileName(), input.getMaskFileName(), input.getAnswerFileName(), 45);

		Cluster cluster(input.getFeatureFileName(),fs,cltDirname);	
		int SentenceNum = fs.getSpeechNum();
		for (int j = 0; j < SentenceNum ; j++) {
			int fNum = fs.getFrameNumInSpeech(j);

		double* features = new double[fNum * 45];
		fs.getSpeechAt(j, features);
		cluster.clusterStat(features, j, 45, fNum, sumRes);//Èç¹û¾ÛÀà
		delete []features;

		}
	}
	FILE* f = fopen(string(cltDirname + "_Stat.txt").c_str(), "w");
	for (auto i : sumRes)
	{
		fprintf(f, "%d\n",i);
	}
	fclose(f);
}