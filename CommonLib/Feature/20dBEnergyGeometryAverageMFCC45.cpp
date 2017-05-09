#include <windows.h>
#include <math.h>
#include "MFCC.h"
#include "20dBEnergyGeometryAverageMFCC45.h"

CMFCC	CMfcc(DEFAULT_SAMPLE_RATE,
			  DEFAULT_FFT_LEN,
			  DEFAULT_FRAME_LEN,
			  DEFAULT_SUB_BAND_NUM,
			  DEFAULT_CEP_COEF_NUM);

float	aa[5] = {-0.75f, -0.375f, 0.0f, 0.375f, 0.75f};

/*-----------------------------------------------------------------------��������������-----*/
/*   �����޸ĺ���������򣬸��������źŵ�Ԫ�����������������������ź������ļ���ƽ��ֵ       */ 
/*   Ԫ���źŵ�����ֵ���������֡������-20dB                                                */ 
/*---------------------------------------------------------------------------��������������-*/
bool get20dBEnergyGeometryAveragMfcc(short *SpeecBuffer,float (*FeatureBuffer)[DIM],long FrameNum)
{
long	i,FrameCounter,FrameNo;
float	FrameEnergy,MaxEn,AveEn;

	//�� 0 ~ 13 ά��ŵ��ף��� 14 ~ 27 ά���һ�׵���
	for( FrameNo = 0 ; FrameNo < FrameNum; FrameNo++ )
	{
		//�� 0 ~ 13 ά��ŵ���
		FrameEnergy = CMfcc.DoMFCC( &SpeecBuffer[FrameNo * FRAME_STEP], FeatureBuffer[FrameNo] );	
		FeatureBuffer[FrameNo][3*D] = (float)(10.0*log10(FrameEnergy));
	}

	//����������������֡ ------ 2001,6,11�޶�
	MaxEn = -1000;
	for( FrameNo = 0; FrameNo < FrameNum; FrameNo++ )
	{
		if( FeatureBuffer[FrameNo][3*D] > MaxEn )
			MaxEn = FeatureBuffer[FrameNo][3*D];
	}
	//����������ƽ������	
	AveEn		 = 0.0;
	FrameCounter = 0;
	for( FrameNo = 0; FrameNo < FrameNum; FrameNo++ )
	{	//�����������-20dBΪԪ���źŵ�����ֵ����������ƽ������
		if( FeatureBuffer[FrameNo][3*D] > MaxEn -20 )
		{
			AveEn += FeatureBuffer[FrameNo][3*D];
			FrameCounter++;
		}
	}
	AveEn = AveEn/FrameCounter;
	// ��һ������֡����
	for( FrameNo =0 ; FrameNo < FrameNum; FrameNo++ )
		FeatureBuffer[FrameNo][3*D] = FeatureBuffer[FrameNo][3*D]-AveEn;

	//����ͷ��֡�������֡��һ�ײ�ֵ���
	for( i = 0; i < D; i++ )
	{
		FeatureBuffer[0][D+i]=(float)(2.0*(aa[1]*FeatureBuffer[0][i]+aa[3]*FeatureBuffer[1][i]));		//��0֡
		FeatureBuffer[1][D+i]=(float)(2.0*(aa[1]*FeatureBuffer[0][i]+aa[3]*FeatureBuffer[2][i]));		//��1֡
		FeatureBuffer[FrameNum-2][D+i]=(float)(2.0*(aa[1]*FeatureBuffer[FrameNum-3][i]+aa[3]*FeatureBuffer[FrameNum-1][i]) );
		FeatureBuffer[FrameNum-1][D+i]=(float)(2.0*(aa[1]*FeatureBuffer[FrameNum-2][i]+aa[3]*FeatureBuffer[FrameNum-1][i]) );
	}
	//�����м�֡��һ�ײ�ֵ���
	for( FrameNo = 2; FrameNo < FrameNum-2; FrameNo++ ) 
	{
		for( i = 0; i < D; i++ )
			FeatureBuffer[FrameNo][D+i] = 
				aa[0]*FeatureBuffer[FrameNo-2][i] +
				aa[1]*FeatureBuffer[FrameNo-1][i] +
				aa[3]*FeatureBuffer[FrameNo+1][i] +
				aa[4]*FeatureBuffer[FrameNo+2][i];
	}

	/************************************/
	//����ͷ��֡�������֡�Ķ��ײ�ֵ���
	for( i = 0; i < D; i++ )
	{
		FeatureBuffer[0][2*D+i]=(float)(2.0*(aa[1]*FeatureBuffer[0][D+i]+aa[3]*FeatureBuffer[1][D+i]));
		FeatureBuffer[1][2*D+i]=(float)(2.0*(aa[1]*FeatureBuffer[0][D+i]+aa[3]*FeatureBuffer[2][D+i]));
		FeatureBuffer[FrameNum-2][2*D+i]=(float)(2.0*(aa[1]*FeatureBuffer[FrameNum-3][D+i]+aa[3]*FeatureBuffer[FrameNum-1][D+i]) );
		FeatureBuffer[FrameNum-1][2*D+i]=(float)(2.0*(aa[1]*FeatureBuffer[FrameNum-2][D+i]+aa[3]*FeatureBuffer[FrameNum-1][D+i]) );
	}
	//����� FrameNo ֡���׵Ķ��ײ��
	for( FrameNo = 2; FrameNo < FrameNum-2; FrameNo++ ) 
	{
		for( i = 0; i < D; i++ )
			FeatureBuffer[FrameNo][2*D+i] = 
				aa[0]*FeatureBuffer[FrameNo-2][D+i] +
				aa[1]*FeatureBuffer[FrameNo-1][D+i] +
				aa[3]*FeatureBuffer[FrameNo+1][D+i] +
				aa[4]*FeatureBuffer[FrameNo+2][D+i];
	}

	//����ͷ��֡�������֡��һ���������
	FeatureBuffer[0][3*D+1]	= (float)(0.8*(aa[1]*FeatureBuffer[0][3*D] + aa[3]*FeatureBuffer[1][3*D]));
	FeatureBuffer[1][3*D+1]	= (float)(0.8*(aa[1]*FeatureBuffer[0][3*D] + aa[3]*FeatureBuffer[2][3*D]));
	FeatureBuffer[FrameNum-2][3*D+1] = (float)(0.8*(aa[1]*FeatureBuffer[FrameNum-3][3*D] + aa[3]*FeatureBuffer[FrameNum-1][3*D]));
	FeatureBuffer[FrameNum-1][3*D+1] = (float)(0.8*(aa[1]*FeatureBuffer[FrameNum-2][3*D] + aa[3]*FeatureBuffer[FrameNum-1][3*D]));
	//�����м�֡��һ���������
	for( FrameNo = 2; FrameNo < FrameNum-2; FrameNo++ )
		FeatureBuffer[FrameNo][3*D+1] = 
			(float)(0.4*(aa[0]*FeatureBuffer[FrameNo-2][3*D]
					   + aa[1]*FeatureBuffer[FrameNo-1][3*D]
					   + aa[3]*FeatureBuffer[FrameNo+1][3*D]
					   + aa[4]*FeatureBuffer[FrameNo+2][3*D]));

	//����ͷ��֡�������֡�Ķ����������
	FeatureBuffer[0][3*D+2] = (float)(1.5*(FeatureBuffer[1][3*D+1]-FeatureBuffer[0][3*D+1]));
	FeatureBuffer[FrameNum-1][3*D+2] = (float)(1.5*(FeatureBuffer[FrameNum-1][3*D+1] - FeatureBuffer[FrameNum-2][3*D+1]));
	//�����м�֡�Ķ����������
	for(FrameNo=1;FrameNo<FrameNum-1;FrameNo++)
		FeatureBuffer[FrameNo][3*D+2] = (float)(1.5*(FeatureBuffer[FrameNo+1][3*D+1] - FeatureBuffer[FrameNo-1][3*D+1]));

	return TRUE;
}


