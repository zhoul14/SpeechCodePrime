
#include <windows.h>
#include <stdio.h>
#include "CPhraseRecg.h"

int	*CPHRASE_RECG::SpeechRecg(float *SpeechBuffer,int SpeechLen)
{
FEATURE	*pFeature;
int	FeaFrameNum,CandNum;
int	FrameNo,*pCandNo;


	pFeature=TelephoneMFCC(SpeechBuffer,SpeechLen,&FeaFrameNum);

	ResetFirstPassRecg();//��һ��ʶ�����
	for( FrameNo = 0; FrameNo < FeaFrameNum-4; FrameNo++ )
	{
		FirstPassRecg(pFeature[FrameNo+2]);
	}
	CandNum = MAX_CANDIDATE_NUM;
	if( PhraseNum < MAX_CANDIDATE_NUM )
       CandNum=PhraseNum;
	pCandNo=GetFirstPassCand(CandNum);		//ȡ��ǰʮ����ѡ
		
//printf("\n�绰��:%ld -> FrameNum:%4ld  ʶ����:\n",LineNo,FeaFrameNum);
//for(i=0;i<CandNum;i++)
//	{
//		printf("%s\n",CmdText[ pCandNo[i] ]);
//	}

	ResetSecondPassRecg(pCandNo,CandNum);	//�ڶ���ʶ�����
	for( FrameNo = 0; FrameNo < FeaFrameNum-4; FrameNo++ )
	{
		SecondPassRecg(&pFeature[FrameNo+2],pCandNo,CandNum);
	}
	pCandNo=GetSecondPassCand(pCandNo,CandNum);//�����뱾������

	//printf("\n�绰��:%ld -> FrameNum:%4ld  ʶ����:\n",LineNo,FeaFrameNum);
	//for(i=0;i<3;i++)
//	{
//		printf("%s\n",CmdText[ pCandNo[i] ]);
//	}
//	delete	[]pFeature;
	return(pCandNo);

}


//����ȫ���ڵ�֡ͬ����������ʶ���� 
CPHRASE_RECG::CPHRASE_RECG(CCODE_BOOK *pSetCCodeBook, char *PhraseFileName)
{
FILE	*fp;
long	i,HmmStateNo,PhraseWord,WordNo,PhraseNo,WordNum;
	
	CurrentFrameNo	= -1;
	pCCodeBook		= pSetCCodeBook;
	CodeBookNum		= pCCodeBook->GetCodeBookNum();

	pCMfcc = new CMFCC(DEFAULT_SAMPLE_RATE,
						DEFAULT_FFT_LEN,
						DEFAULT_FRAME_LEN,
						DEFAULT_SUB_BAND_NUM,
						DEFAULT_CEP_COEF_NUM);

	fopen_s(&fp,PhraseFileName,"rt");
	fscanf_s(fp,"%ld\n",&PhraseNum);
	if( PhraseNum > 0 && PhraseNum < MAX_PHRASE_NUM )
	{
		for( PhraseNo = 0; PhraseNo < PhraseNum; PhraseNo++)
		{
			fgets(CmdText[PhraseNo],198,fp);
			for(i=0;i<198;i++)
			{
				if(CmdText[PhraseNo][i]=='\n')
				{
					CmdText[PhraseNo][i]=0;
					break;
				}
			}
			fscanf_s(fp,"%ld",&WordNum);
			StateNum[PhraseNo]=HMM_STATE_NUM*WordNum;
			for(WordNo=0;WordNo<WordNum;WordNo++)
			{
				fscanf_s(fp,"%ld ",&PhraseWord);
				for(HmmStateNo=0;HmmStateNo<HMM_STATE_NUM;HmmStateNo++)
				{
					StateCodeBookNo[PhraseNo][WordNo*HMM_STATE_NUM+HmmStateNo]=pCCodeBook->pCodeBookNo[PhraseWord][HmmStateNo];
				}
			}
		}
	}
	fclose(fp);

	for( PhraseNo = 0; PhraseNo < PhraseNum; PhraseNo++ )
	{
		for( HmmStateNo = 1 ; HmmStateNo < StateNum[PhraseNo]; HmmStateNo++ )	//��ʼ��ʱ����״̬��·��
		{
			PhrasePathProb[PhraseNo][HmmStateNo] =(float)(MIN_PROBABILITY*HmmStateNo);
		}
		PhrasePathProb[PhraseNo][0] = 0;
	}
}
CPHRASE_RECG::~CPHRASE_RECG()
{
	delete pCMfcc;
}


void CPHRASE_RECG::ResetFirstPassRecg(void)
{
long	PhraseNo,HmmStateNo;

	CurrentFrameNo	= -1;
	for( PhraseNo = 0; PhraseNo < PhraseNum; PhraseNo++ )
	{
		for( HmmStateNo = 1 ; HmmStateNo < StateNum[PhraseNo]; HmmStateNo++ )	//��ʼ��ʱ����״̬��·��
		{
			PhrasePathProb[PhraseNo][HmmStateNo] =(float)( MIN_PROBABILITY*HmmStateNo );
		}
		PhrasePathProb[PhraseNo][0] = 0;
	}
}

void CPHRASE_RECG::ResetSecondPassRecg(int *CandBuffer,int CandNum)
{
	ResetFirstPassRecg();

}

void CPHRASE_RECG::SecondPassRecg(FEATURE *pFeature,int *CandBuffer,int CandNum)
{
long			CandNo,PhraseNo,HmmStateNo,CodeBookNo;
float			StateProb;

	for( CodeBookNo = 0 ; CodeBookNo < CodeBookNum; CodeBookNo++ )
	{
		pCCodeBook->pFeaToCodeBookProb[CodeBookNo] = MIN_PROBABILITY;
	}


	CurrentFrameNo++;
	if( CurrentFrameNo == 0 )
	{	//�����һ��֡�ĸ���
		for( CandNo = 0 ;CandNo < CandNum; CandNo++ )
		{
			PhraseNo = CandBuffer[CandNo];
			if(PhraseNo < 0 ) continue;		//û���㹻��ĺ�ѡ
			CodeBookNo = StateCodeBookNo[PhraseNo][0];
			if( pCCodeBook->pFeaToCodeBookProb[CodeBookNo] == MIN_PROBABILITY )
				StateProb = pCCodeBook->FeaToCodeBookProb(pFeature[0],CodeBookNo);
			else
				StateProb = pCCodeBook->pFeaToCodeBookProb[CodeBookNo];
			PhrasePathProb[PhraseNo][0]  = StateProb;
		}	
	}
	else
	{
		for( CandNo = 0 ; CandNo < CandNum; CandNo++ )
		{
			PhraseNo = CandBuffer[CandNo];
			if(PhraseNo < 0 ) continue;		//û���㹻��ĺ�ѡ

			//��ÿ����ʶ��������֡ͬ��ʶ��		
			for( HmmStateNo = StateNum[PhraseNo]-1; HmmStateNo > 0; HmmStateNo-- )
			{
				//��ÿ��HMM״̬����֡ͬ������
				//���ȼ�����һʱ�̱�״̬��·����ԭ��פ������Ȼ����
				CodeBookNo = StateCodeBookNo[PhraseNo][HmmStateNo];
				if( pCCodeBook->pFeaToCodeBookProb[CodeBookNo] == MIN_PROBABILITY )
					StateProb = pCCodeBook->FeaToCodeBookProb(pFeature[0],CodeBookNo);
				else
					StateProb = pCCodeBook->pFeaToCodeBookProb[CodeBookNo];
				
				if( PhrasePathProb[PhraseNo][HmmStateNo] > PhrasePathProb[PhraseNo][HmmStateNo-1] )
				{
					PhrasePathProb[PhraseNo][HmmStateNo]  += StateProb;
				}
				else
				{
					PhrasePathProb[PhraseNo][HmmStateNo] = PhrasePathProb[PhraseNo][HmmStateNo-1]+StateProb;
				}
			}	//HmmStateNo=StateNum[PhraseNo]-1 ,...1
			HmmStateNo=0;
			CodeBookNo = StateCodeBookNo[PhraseNo][HmmStateNo];
			if( pCCodeBook->pFeaToCodeBookProb[CodeBookNo] == MIN_PROBABILITY )
				StateProb = pCCodeBook->FeaToCodeBookProb(pFeature[0],CodeBookNo);
			else
				StateProb = pCCodeBook->pFeaToCodeBookProb[CodeBookNo];
			PhrasePathProb[PhraseNo][HmmStateNo]  += StateProb;
		}	//PhraseNo=0 , ... , TOTAL_WORD_NUM-1
	}
}

void CPHRASE_RECG::FirstPassRecg(FEATURE Feature)
{
long			i,PhraseNo,HmmStateNo,CodeBookNo;
float			StateProb;
CODE_PRECISION	DotProduct,tmp;
CODE_PRECISION	*pMeanU;

	for( CodeBookNo = 0 ; CodeBookNo < CodeBookNum; CodeBookNo++ )
	{
		pCCodeBook->pFeaToCodeBookProb[CodeBookNo] = MIN_PROBABILITY;
	}

	CurrentFrameNo++;
	if( CurrentFrameNo == 0 )
	{	//�����һ��֡�ĸ���
		for( PhraseNo = 0 ;PhraseNo < PhraseNum; PhraseNo++ )
		{
			CodeBookNo = StateCodeBookNo[PhraseNo][0];
			if( pCCodeBook->pFeaToCodeBookProb[CodeBookNo] == MIN_PROBABILITY )
			{
				pMeanU = pCCodeBook->pCodeBook[CodeBookNo].MeanU;	//��ֵ�뱾
				DotProduct  = 0;
				for( i = 0; i < FEATURE_DIM; i++ )
				{
					tmp=Feature[i] - pMeanU[i];
					DotProduct += tmp * tmp;
				}
				StateProb = float(-DotProduct);
				pCCodeBook->pFeaToCodeBookProb[CodeBookNo] = StateProb;
			}
			else
				StateProb = pCCodeBook->pFeaToCodeBookProb[CodeBookNo];

			PhrasePathProb[PhraseNo][0]  = StateProb;
		}	
	}
	else
	{
		for( PhraseNo = 0 ;PhraseNo < PhraseNum; PhraseNo++ )
		{
			//��ÿ����ʶ��������֡ͬ��ʶ��		
			for( HmmStateNo = StateNum[PhraseNo]-1; HmmStateNo > 0; HmmStateNo-- )
			{
				//��ÿ��HMM״̬����֡ͬ������
				//���ȼ�����һʱ�̱�״̬��·����ԭ��פ������Ȼ����
				CodeBookNo = StateCodeBookNo[PhraseNo][HmmStateNo];
				if( pCCodeBook->pFeaToCodeBookProb[CodeBookNo] == MIN_PROBABILITY )
				{
					pMeanU = pCCodeBook->pCodeBook[CodeBookNo].MeanU;	//��ֵ�뱾
					DotProduct  = 0;
					for( i = 0; i < FEATURE_DIM; i++ )
					{
						tmp=Feature[i] - pMeanU[i];
						DotProduct += tmp * tmp;
					}
					StateProb = float(-DotProduct);
					pCCodeBook->pFeaToCodeBookProb[CodeBookNo] = StateProb;
				}
				else
					StateProb = pCCodeBook->pFeaToCodeBookProb[CodeBookNo];
				
				if( PhrasePathProb[PhraseNo][HmmStateNo] > PhrasePathProb[PhraseNo][HmmStateNo-1] )
				{
					PhrasePathProb[PhraseNo][HmmStateNo]  += StateProb;
				}
				else
				{
					PhrasePathProb[PhraseNo][HmmStateNo] = PhrasePathProb[PhraseNo][HmmStateNo-1]+StateProb;
				}
			}	//HmmStateNo=StateNum[PhraseNo]-1 ,...1
			HmmStateNo=0;
			CodeBookNo = StateCodeBookNo[PhraseNo][HmmStateNo];
			if( pCCodeBook->pFeaToCodeBookProb[CodeBookNo] == MIN_PROBABILITY )
			{
				pMeanU = pCCodeBook->pCodeBook[CodeBookNo].MeanU;	//��ֵ�뱾
				DotProduct  = 0;
				for( i = 0; i < FEATURE_DIM; i++ )
				{
					tmp = Feature[i] - pMeanU[i];
					DotProduct += tmp * tmp;
				}
				StateProb = float(-DotProduct);
				pCCodeBook->pFeaToCodeBookProb[CodeBookNo] = StateProb;
			}
			else
				StateProb = pCCodeBook->pFeaToCodeBookProb[CodeBookNo];
			PhrasePathProb[PhraseNo][HmmStateNo]  += StateProb;
		}	//PhraseNo=0 , ... , TOTAL_WORD_NUM-1
	}
}

int  *CPHRASE_RECG::GetFirstPassCand(int CandNum)
{
long			BestPhraseNo,PhraseNo,HmmStateNo;
float			MaxProb;
static int		CandList[MAX_CANDIDATE_NUM],CandNo;

	for( int i = 0; i < MAX_CANDIDATE_NUM; i++ )
	{
		CandList[i] = -1;
	}

	//��ʱ���ǵ�ǰ֡������������������

	for( CandNo = 0; CandNo < CandNum; CandNo++ )
	{
		MaxProb			= MIN_PROBABILITY;
		BestPhraseNo	= -1;
		for( PhraseNo = 0; PhraseNo < PhraseNum; PhraseNo++ )
		{
			HmmStateNo = StateNum[PhraseNo]-1;
			if( MaxProb < PhrasePathProb[PhraseNo][HmmStateNo] )		//Ѱ��������ʶ������������ʣ��ִ����룩
			{
				MaxProb			= PhrasePathProb[PhraseNo][HmmStateNo];
				BestPhraseNo	= PhraseNo;
			}
		}
		if( BestPhraseNo == -1 ) break;			//˵��û���㹻��ĺ�ѡ,�������������ر��ʱ,�п��ܷ���
		CandList[CandNo] = BestPhraseNo;
		HmmStateNo = StateNum[BestPhraseNo]-1;
		PhrasePathProb[BestPhraseNo][HmmStateNo] = MIN_PROBABILITY;
//		printf("PhraseNo=%ld Dist=%f\n",BestPhraseNo,MaxProb);
	}
	return(CandList);
}

int  *CPHRASE_RECG::GetSecondPassCand(int *CandBuffer,int CandNum)
{
int			BestPhraseNo,PhraseNo,HmmStateNo;
float			MaxProb;
static int		CandList[MAX_CANDIDATE_NUM],FineCandNo,CandNo;
	

	for( int i = 0; i < MAX_CANDIDATE_NUM; i++ )
	{
		CandList[i]		= -1;
		BestProbList[i] = MIN_PROBABILITY;
	}
	//��ʱ���ǵ�ǰ֡������������������
	for( FineCandNo = 0; FineCandNo < CandNum; FineCandNo++ )
	{
		MaxProb			= MIN_PROBABILITY;
		BestPhraseNo	= -1;
	
		for( CandNo = 0; CandNo < CandNum; CandNo++ )
		{
			PhraseNo	 = CandBuffer[CandNo];
			HmmStateNo	 = StateNum[PhraseNo] -1;
			if( MaxProb < PhrasePathProb[PhraseNo][HmmStateNo] )		// Ѱ��������ʶ������������ʣ��ִ����룩
			{
				MaxProb			= PhrasePathProb[PhraseNo][HmmStateNo];
				BestPhraseNo	= PhraseNo;
			}
		}
		if( BestPhraseNo == -1 )	break;
		CandList[FineCandNo] = BestPhraseNo;
		BestProbList[FineCandNo] = MaxProb;
		HmmStateNo = StateNum[BestPhraseNo]-1;
		PhrasePathProb[BestPhraseNo][HmmStateNo] = MIN_PROBABILITY;
//		printf("PhraseNo=%ld-> %3ld :Dist=%f\n",FineCandNo,BestPhraseNo,MaxProb);

	}
	return(CandList);
}



