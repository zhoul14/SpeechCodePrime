#ifndef WJQ_GMM_UPDATE_MANAGER_H
#define WJQ_GMM_UPDATE_MANAGER_H

#include "MMIEstimator.h"
#include "GMMEstimator.h"
#include "../CommonLib/Dict/WordDict.h"
#include "GMMCodebookSet.h"
#include "FrameWarehouse.h"
#include "WordLevelMMIEstimator.h"
#include "string"
#include "../SpeechSegmentAlgorithm/SegmentAlgorithm.h"

class SegmentResult;
class GMMUpdateManager {
private:

	GMMCodebookSet* codebooks;

	GMMEstimator* estimator;

	double* firstMoment;

	double* secondMoment;

	double* firstMomentOfLog;

	int* successUpdateTime;

	int updateIter;

	std::vector<std::vector<double>>SimilarMat;

	int* durCnt;

	bool useSegmentModel;

	bool m_bMMIE;

	bool m_bCuda;

	WordDict* dict;

	double **m_pMMIEres;

	double **m_pMMIEgen;

	double minDurVar;

	double ObjectFuncVal;

	FrameWarehouse* fw;

	GMMProbBatchCalc* gbc;

	FILE* logFile;

	std::vector<std::vector<int>> m_ConMatrix;//different data same state

	std::vector<std::vector<int>> m_DataMatrix;//different state same data

	std::vector<std::vector<double>> m_WordGamma;//different WordGamma

	std::vector<int>m_WordGammaLastPos;

	double m_dDsm;

	int makeMMIEmatrix();

	void prepareMMIELh();

public:

	GMMUpdateManager(GMMCodebookSet* codebooks, int maxIter, WordDict* dict, double minDurSigma, const char* logFileName, bool useCuda, bool useSegmentModel, bool m_bMMIE = false, double Dsm = 0.0);

	//void setShareCovFlag (int flag);\

	int getWordGammaTotalNum();

	double getObjFV(){return ObjectFuncVal;}

	//����ֵ��ʾִ���걾�κ������ۻ���ȫ���뱾����֡��
	int collect(const std::vector<int>& frameLabel, double* frames, bool bCollectFrame = true,int* clusterInfo = nullptr);

	double collectWordGamma(const std::vector<SegmentResult>& recLh, const std::vector<int>& frameLabel, int *pushedFrames, double& lh);

	void printfObjectFunVal(){
		printf("Object Function value:%lf\n",ObjectFuncVal);
		fprintf(logFile,"Object Function value:%lf\n",ObjectFuncVal);
	}

	int collect(const std::vector<int>& frameLabel, double* frames, int *pushedFrames, double& lh);

	int collectMultiFrames(const std::vector<int>& frameLabel, double* frames);

	int collectWordGamma(const std::vector<int>& frameLabel, std::vector<SWord>& recLh, int ans, double segLh);

	int collectWordGamma(const std::vector<SegmentResult>& recLh, int wordIdx, bool* usedFrame);

	int getUaCbnum(){return codebooks->CodebookNum;}

	//����ֵΪ����Ϊcbnum��vector��vector��ÿ��Ԫ�ش�����Ӧ���뱾�ĸ��½��
	std::vector<int> update();

	std::vector<int> updateStateLvMMIE();

	std::vector<int> updateWordLvMMIE();	

	FrameWarehouse* getFW(){return fw;}

	int getSuccessUpdateTime(int cbidx) const;

	void setGBC(GMMProbBatchCalc* pGbc){
		gbc = pGbc;
	}
	int makeConvMatrix();

	std::vector<std::vector<double>> &getSimilarMat(){
		return SimilarMat;
	}

	bool getMMIEmatrix(std::string filename, bool bDataState);

	int setMMIEmatrix(std::vector<std::vector<double>>&x);

	void setSimilarMatrix(std::vector<std::vector<double>>&x){
		SimilarMat = x;
	}

	void mergeIntoVec(vector<int> vec, vector<vector<int>>&v, bool* bList);

	void updateSMLmat(int idx);

	void trainKerasNN(bool bMakeNewModel, std::string ot, std::string it);

	~GMMUpdateManager();

	void summaryUpdateRes(const vector<int>& r, FILE* fid, int iterNum);
};

#endif