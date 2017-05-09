#include "..\include\CViterbi.h"

CVITERBI::CVITERBI(CVITERBI_CODEBOOK *pCodeBook,long *pCodeBookNo)
{
long	i;

	pCViterbiCodeBook = pCodeBook;
	pPathProb = new float[VITERBI_STATE_NUM];
	pPath	  = new long[VITERBI_STATE_NUM][VITERBI_STATE_NUM+1];

	if( pCodeBookNo != 0 )
	{
		for(i=0;i<VITERBI_STATE_NUM;i++)
		{
			CodeBookNo[i]=pCodeBookNo[i];
		}
	}
}

CVITERBI::~CVITERBI()
{
	delete	[]pPathProb;
	delete	[]pPath;
}
void CVITERBI::SetCodeBook(CVITERBI_CODEBOOK *pCodeBook)
{
	pCViterbiCodeBook=pCodeBook;
}
void CVITERBI::SetStateCodeBookNo(long *pCodeBookNo)
{
long	i;

	for(i=0;i<VITERBI_STATE_NUM;i++)
	{
		CodeBookNo[i]=pCodeBookNo[i];
	}
}

void	CVITERBI::SetupViterbi(VITERBI_FEA *pViterbiFea)
{
long	ViterbiStateNo;

	FrameNum=1;
	//��ʼ��·��
	pPath[VITERBI_STATE_NUM-1][0]					= 0;
	pPath[VITERBI_STATE_NUM-1][VITERBI_STATE_NUM]   = FrameNum;
	//��ʼ��·������
	pPathProb[0]=MIN_VITERBI_PROB;
	for(ViterbiStateNo=1;ViterbiStateNo<VITERBI_STATE_NUM;ViterbiStateNo++)
		pPathProb[ViterbiStateNo]=pPathProb[ViterbiStateNo-1]*1.2f;
	//�̶���һ֡�Ķ˵�Ϊ��0��״̬
	pPathProb[0]=pCViterbiCodeBook->pFeaToCodeBookProb[ CodeBookNo[0] ];
}


float * CVITERBI::FrameSynViterbi(VITERBI_FEA *pViterbiFea)
{
long	i,ViterbiStateNo;

	//��ʼVITERBI�㷨,�������ʣ��Ķ���������·��
	for(ViterbiStateNo = VITERBI_STATE_NUM-1; ViterbiStateNo > 0; ViterbiStateNo-- )
	{
		if( pPathProb[ViterbiStateNo-1] > pPathProb[ViterbiStateNo] )
		{
			pPathProb[ViterbiStateNo] = pPathProb[ViterbiStateNo-1];
			//��¼���뵱ǰ״̬��ʱ��
			pPath[ViterbiStateNo][ViterbiStateNo] = FrameNum;
			//���µ�ǰ״̬��·����¼
			for(i=ViterbiStateNo-1;i>0;i--)    
				pPath[ViterbiStateNo][i]=pPath[ViterbiStateNo-1][i];
		}
		pPathProb[ViterbiStateNo]+=pCViterbiCodeBook->pFeaToCodeBookProb[CodeBookNo[ViterbiStateNo]];
	}
	//·���ڵ�0��״̬פ��
	pPathProb[0]+=pCViterbiCodeBook->pFeaToCodeBookProb[CodeBookNo[0]];
	FrameNum++;
	pPath[VITERBI_STATE_NUM-1][VITERBI_STATE_NUM]  = FrameNum;
	return(pPathProb);
}
long * CVITERBI::GetSegmentation(void)
{
	return(pPath[VITERBI_STATE_NUM-1]);
}

float * CVITERBI::GetPathProb(void)
{
	return(pPathProb);
}
