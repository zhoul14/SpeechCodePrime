#include "../SpeechSegmentAlgorithm/SegmentAlgorithm.h"
#include "../CommonLib/ReadConfig/TrainParam.h"
#include "../CommonLib/Dict/WordDict.h"
#include "../CommonLib/FileFormat/FeatureFileSet.h"
#include "../SpeechSegmentAlgorithm/SimpleSTFactory.h"
#include "../GMMCodebook/GMMCodebookSet.h"
#include "../GMMCodebook/GMMProbBatchCalc.h"
#include "../GMMCodebook/GMMUpdateManager.h"
#include <windows.h>
#include <math.h>
#include <algorithm>
#include <time.h>
#include <cassert>
#include <vector>
#include <string>
#include <cuda.h>
#include "../StateProbabilityMap/StateProbabilityMap.h"
#include "assert.h"
#include "../NBestRecAlgorithm/NBestRecAlgorithm.h"
#include "omp.h"
#include "../CommonLib/Cluster.h"
//#include "vld.h"
using std::vector;
using std::string;

#define USEDURA 1
#define PRIME 0
#define HALF_FRAME_LEN 0
#define MULTIFRAME 0
#define PREPARE_PY 0
#define MAKE_NN_MODEL 0

#define FRAMECOLLECT 1
#define STATEPROBMAP 1
#define PRIME_DIM 45
#define WORD_LEVEL_MMIE 1
#define CLUSTER 0

int main(int argc, char** argv) {

	if (argc > 3) {
		printf("usage: program_name config_file [basedir]");
		exit(-1);
	}
	if (argc == 3) {
		if (GetFileAttributes(argv[2]) == INVALID_FILE_ATTRIBUTES) {
			CreateDirectory(argv[2], NULL);
		}
		SetCurrentDirectory(argv[2]);
		//current_path(argv[2]);
	}


	string* configName,* configName2;
	if (argc == 1) {
		configName = new string("train_config.xml");
	} else {
		configName = new string(argv[1]);
	}


	TrainParam tparam(configName->c_str());
	bool triPhone = tparam.getTriPhone();
	//////////////////////////////////////////////////////////////////////////
	//for (int crossfitIter = 0; crossfitIter < 50 ;crossfitIter++){
	//////////////////////////////////////////////////////////////////////////	
	//初始化u中各成员
	SegmentUtility u;
	SegmentUtility uHelper;

	SimpleSTFactory* stfact = new SimpleSTFactory();
	SimpleSTFactory* stfactHelper = NULL;

	u.factory = stfact;
	uHelper.factory = stfactHelper;

	WordDict* dict = new WordDict(tparam.getWordDictFileName().c_str(),triPhone);
	u.dict = dict;
	uHelper.dict = dict;
	dict->makeUsingCVidWordList();
	string initCb = tparam.getInitCodebook();

	double Coef = tparam.getDCoef();

	GMMCodebookSet* set = new GMMCodebookSet(initCb.c_str(),0);
	NBestRecAlgorithm* reca = new NBestRecAlgorithm();

	GCSTrimmer::fixSmallDurSigma(set, tparam.getMinDurSigma());
	int splitAddN = tparam.getSplitAddN();
	double splitOffset = tparam.getSplitOffset();
	if (splitAddN > 0) {
		printf("splitting codebooks, add %d mixtures\n", splitAddN);
		if (splitAddN > set->MixNum) {
			printf("splitAddN[%d] param cannot be larger than MixNum[%d]\n", splitAddN, set->MixNum);
			exit(-1);
		}
		set->splitAddN(splitAddN, splitOffset);
	}

	int cbNum = set->getCodebookNum();

	double durWeight = tparam.getDurWeight();
	bool useCuda = tparam.getUseCudaFlag();
	bool useSegmentModel = tparam.getSegmentModelFlag();
	if (useCuda) {
		printf("UseCUDA is true\n");
	} else {
		printf("UseCUDA is false\n");
	}

	if (useSegmentModel) {
		printf("UseSegmentModel is true\n");
	} else {
		printf("UseSegmentModel is false\n");
	}
	GMMProbBatchCalc* gbc =	new GMMProbBatchCalc(set, useCuda, useSegmentModel);
	printf("cbs fdim = %d, cbs statenum = %d, cbs mixture = %d\n",set->getFDim(),set->getCodebookNum(),set->getMixNum());
	gbc->setDurWeight(durWeight);
	bool clusterFlag = tparam.getCltDirName() != "";
	u.bc = gbc;//
	vector<TSpeechFile> inputs = tparam.getTrainFiles();

	SegmentAlgorithm sa;

	int maxEMIter = tparam.getEMIterNum();

	int trainIter = tparam.getTrainIterNum();

	string logPath = tparam.getLogPath();

	string lhRecordPath = logPath + "/lh_record.txt";
	string updateTimePath = logPath + "/update_time.txt";
	string updateIterPath = logPath + "/update_iter.txt";
	string summaryPath = logPath + "/summary.txt";

	FILE* lhRecordFile = fopen(lhRecordPath.c_str(), "w");
	if (!lhRecordFile) {
		printf("cannot open log file[%s]\n", lhRecordPath.c_str());
		exit(-1);
	}

	FILE* updateTimeFile = fopen(updateTimePath.c_str(), "w");
	if (!updateTimeFile) {
		printf("cannot open log file[%s]\n", updateTimePath.c_str());
		exit(-1);
	}

	FILE* summaryFile = fopen(summaryPath.c_str(), "w");
	if (!summaryFile) {
		printf("cannot open log file[%s]\n", summaryPath.c_str());
		exit(-1);
	}

	GMMUpdateManager ua(set, maxEMIter, dict, tparam.getMinDurSigma(), updateIterPath.c_str(), useCuda, useSegmentModel, STATEPROBMAP, 25);

	/*ua.getMMIEmatrix("ConMatrix.txt",false);
	ua.getMMIEmatrix("DataMatrix.txt",true);*/
	//set->printCodebookSetToFile(string(logPath + "/all_codebooks.txt").c_str());

	bool* trainedCb = new bool[cbNum];

	double lhOfLastIter = -1e300; 

	for (int iter = 0; iter < trainIter ; iter++) {

		clock_t begTime = clock();
		clock_t labTime = 0;
		clock_t prepareTime = 0;
		double lhOfThisIter = 0;

		int trainCnt = -1;

#if STATEPROBMAP
		CStateProbMap CSPM(cbNum);
#endif

		/************************************************start segment*************************************************/
		for (auto i = inputs.begin(); i != inputs.end(); i++) {
			trainCnt++;
			if (trainCnt == tparam.getTrainNum())
				break;
			const int fDim = tparam.getFdim();

			FeatureFileSet input((*i).getFeatureFileName(), (*i).getMaskFileName(), (*i).getAnswerFileName(), fDim);

			Cluster cluster((*i).getFeatureFileName(),input, tparam.getCltDirName());
			int speechNumInFile = input.getSpeechNum();
			for (int j = 0; j < (speechNumInFile); j++) {

				printf("process file %d, speech %d    \r", trainCnt, j);
				int fNum = input.getFrameNumInSpeech(j);

				//if(fNum!=input2.getFrameNumInSpeech(j))//
				//printf("fNum1:%d does not match fNum2:%d",fNum,input2.getFrameNumInSpeech(j));

				int ansNum = input.getWordNumInSpeech(j);

				if (fNum < ansNum * HMM_STATE_DUR * 2) {
					printf("\ntoo short speech, file = %d, speech = %d ignored in training (fNum = %d, ansNum = %d)\n", trainCnt, j, fNum, ansNum);
					continue;
				}
				double* frames = new double[fNum * fDim];
				input.getSpeechAt(j, frames);
				int* ansList = new int[ansNum];
				input.getWordListInSpeech(j, ansList);
				//分割前完成概率的预计算
				bool* mask = new bool[dict->getTotalCbNum()];
				clock_t t1 = clock();
				gbc->prepare(frames, fNum);
				clock_t t2 = clock();
				prepareTime += t2 - t1;
				vector<SegmentResult>res0;
				if (WORD_LEVEL_MMIE)
				{
					res0.resize(TOTAL_WORD_NUM);
				}
				t1 = clock();
				int answer = ansList[0];
				int* usedFrames = NULL;
				int totalFrameNum;
				SegmentResult res;

				if(WORD_LEVEL_MMIE)
				{
					//totalFrameNum = ua.collect(res.frameLabel, frames, FRAMECOLLECT);
					ua.collectWordGamma(res0,res.frameLabel,usedFrames, res.lh);
				}
				else
				{
					//res = sa.segmentSpeech(fNum, fDim, ansNum, ansList, u);
#if  MULTIFRAME
					res = sa.segmentSpeech(fNum, fDim, ansNum, ansList, u);

					totalFrameNum = ua.collectMultiFrames(res.frameLabel, multiFrames);

#else 
					if(clusterFlag){
						double* tmpFrames = new double[fDim * fNum];
						memcpy(tmpFrames, frames, sizeof(double) * fDim * fNum);
						fNum = cluster.clusterFrameSame(tmpFrames, j, fDim, fNum);
						gbc->prepare(tmpFrames, fNum);
						delete tmpFrames;
					}
					else{
						gbc->prepare(frames, fNum);
					}
					res = sa.segmentSpeech(fNum, fDim, ansNum, ansList, u);
					totalFrameNum = ua.collect(res.frameLabel, frames);
#endif // MULTIFRAMES_COLLECT

				}
				//if(!ua.collectWordGamma(res.frameLabel,res0,ansList[0],res.lh))printf("shit !ans is not in recognition result List!\n\n");
				//ua.collectWordGamma(res0, ansList[0], usedFrames);
				delete []usedFrames;

#endif
#if STATEPROBMAP
				CSPM.pushToMap(gbc,res.frameLabel);
#endif

#if !WORD_LEVEL_MMIE
				assert(stfact->getAllocatedNum() == 0);
#endif
				lhOfThisIter += res.lh;

#if PRIME
				delete [] frames2;
#endif

#if MULTIFRAME
				delete []multiFrames;
#endif
				delete [] ansList;
				delete [] frames;
				delete [] mask;
			}
			//input.PrintSegmentPointBuf("SegMent48_log.txt");
#if WORD_LEVEL_MMIE
			printf("No.%dFile,",trainCnt);
			ua.printfObjectFunVal();
#endif
			printf("\n");
		}
		/*******************************segment end****************************************/

		clock_t midTime = clock();
		int segTime = (midTime - begTime) / CLOCKS_PER_SEC;
		fprintf(lhRecordFile, "iter %d,\tlh = %e, segment time = %ds(%ds, %ds), TotalFrameNum = %d", iter, lhOfThisIter, segTime, labTime / CLOCKS_PER_SEC, prepareTime / CLOCKS_PER_SEC, ua.getFW()->getTotalFrameNum());
		fflush(lhRecordFile);

#if MULTIFRAME
		ua.getFW()->flush();
		cout<<"TotalFrameNum:"<<ua.getFW()->getTotalFrameNum()<<endl;
		//system("pause");
		//return 0;
#endif

#if STATEPROBMAP
		std::vector<std::vector<double>> m;
		m.resize(cbNum);
		CSPM.mergeMapToMatrix(m, ua.getFW());
		CSPM.saveOutMaptoFile(tparam.getInitCodebook() + "Map.txt", ua.getFW());

		int CorrelateCnt = ua.setMMIEmatrix(m);

		fprintf(lhRecordFile, ", Correlate num = %d", CorrelateCnt);
		fflush(lhRecordFile);
		/*return 0;*/
#endif
		vector<int> updateRes;
		if(STATEPROBMAP)
		{
			for (int uptime = 0; uptime != maxEMIter; uptime++)
			{
				ua.setGBC(gbc);
				updateRes = ua.updateStateLvMMIE();
				set->saveCodebook(tparam.getOutputCodebook());
				ua.summaryUpdateRes(updateRes, summaryFile, iter);
			}
		}
		else if(WORD_LEVEL_MMIE)
		{
			fprintf(lhRecordFile, ", object value = %lf", ua.getObjFV());
			updateRes = ua.updateWordLvMMIE();
		}
		else if(PREPARE_PY)
		{
			if(iter == 0)ua.trainKerasNN(MAKE_NN_MODEL, tparam.getOutputCodebook(), tparam.getInitCodebook());
			else{
				ua.trainKerasNN(MAKE_NN_MODEL, tparam.getOutputCodebook(), tparam.getOutputCodebook());
			}
		}
		else
		{
			updateRes = ua.update();
		}
		clock_t endTime = clock();

		int updTime = (endTime - midTime) / CLOCKS_PER_SEC;
		fprintf(lhRecordFile, ", update time = %ds\n", updTime);
		fflush(lhRecordFile);


		lhOfLastIter = lhOfThisIter;

		string allCbPath = logPath + "/all_codebooks.txt";
#if PRIME
		mySet->saveCodebook(tparam2.getOutputCodebook());//
		mySet->printCodebookSetToFile(allCbPath.c_str());//
		delete mySet;
#elif WORD_LEVEL_MMIE
		string ss = tparam.getOutputCodebook()+(char)(iter+48);
		cout<<ss<<endl;
		set->saveCodebook(ss);
		set->printCodebookSetToFile(allCbPath.c_str());
#else
		set->saveCodebook(tparam.getOutputCodebook());
		set->printCodebookSetToFile(allCbPath.c_str());
#endif
	}

	gbc->FinalizePython();
	for (int i = 0; i < ua.getUaCbnum(); i++) {
		fprintf(updateTimeFile, "%d\t%d\n", i, ua.getSuccessUpdateTime(i));
	}

	fclose(lhRecordFile);
	fclose(updateTimeFile);
	fclose(summaryFile);
	delete [] trainedCb;	
	delete set;
	delete stfact;
	delete dict;
	delete gbc;	
	delete configName;
	return 0;
}