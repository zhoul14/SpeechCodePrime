#include <windows.h>
#include <math.h>
#include "MFCC.h"
#include "20dBEnergyGeometryAverageMFCC45.h"

CMFCC	CMfcc(DEFAULT_SAMPLE_RATE,
			  DEFAULT_FFT_LEN,
			  DEFAULT_FRAME_LEN,
			  DEFAULT_SUB_BAND_NUM,
			  DEFAULT_CEP_COEF_NUM);

float	a[9] = {-0.75f, -(0.375f+ 0.75f)/2, -0.375f, -0.375f/2, 0.0f, 0.375f/2, 0.375f, (0.375f+ 0.75f)/2, 0.75f};

/*-----------------------------------------------------------------------��������������-----*/
/*   �����޸ĺ���������򣬸��������źŵ�Ԫ�����������������������ź������ļ���ƽ��ֵ       */ 
/*   Ԫ���źŵ�����ֵ���������֡������-20dB                                                */ 
/*---------------------------------------------------------------------------��������������-*/

bool get20dBEnergyGeometryAveragMfcc_5ms(short *SpeecBuffer,float (*FeatureBuffer)[DIM],long FrameNum)
{
	long	i,FrameCounter,FrameNo;
	float	FrameEnergy,MaxEn,AveEn;

	//�� 0 ~ 13 ά��ŵ��ף��� 14 ~ 27 ά���һ�׵���
	for( FrameNo = 0 ; FrameNo < FrameNum; FrameNo++ )
	{
		//�� 0 ~ 13 ά��ŵ���
		FrameEnergy = CMfcc.DoSMFCC( &SpeecBuffer[FrameNo * FRAME_STEP], FeatureBuffer[FrameNo] );	
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
		//��0֡
		FeatureBuffer[0][D]=(float)(2.0*(a[3]*FeatureBuffer[0][i]+a[5]*FeatureBuffer[1][i]+a[6]*FeatureBuffer[2][i]+a[7]*FeatureBuffer[3][i]+a[8]*FeatureBuffer[4][i]));
		//��1֡
		FeatureBuffer[1][D]=(float)(2.0*(a[3]*FeatureBuffer[0][i]+a[5]*FeatureBuffer[2][i]+a[6]*FeatureBuffer[3][i]+a[7]*FeatureBuffer[4][i]+a[8]*FeatureBuffer[5][i]));
		//��2֡
		FeatureBuffer[2][D]=(float)(2.0*(a[2]*FeatureBuffer[0][i]+a[3]*FeatureBuffer[1][i]+a[5]*FeatureBuffer[4][i]+a[6]*FeatureBuffer[5][i]+a[7]*FeatureBuffer[6][i]+a[8]*FeatureBuffer[7][i]));
		//��3֡
		FeatureBuffer[3][D]=(float)(2.0*(a[1]*FeatureBuffer[0][i]+a[2]*FeatureBuffer[1][i]+a[3]*FeatureBuffer[2][i]+a[5]*FeatureBuffer[4][i]+a[6]*FeatureBuffer[5][i]+a[7]*FeatureBuffer[6][i]+a[8]*FeatureBuffer[7][i]));
		FeatureBuffer[FrameNum-4][D]=(float)(2.0*(a[0]*FeatureBuffer[FrameNum-8][i]+a[1]*FeatureBuffer[FrameNum-7][i]+a[2]*FeatureBuffer[FrameNum-6][i]+a[3]*FeatureBuffer[FrameNum-5][i]+a[5]*FeatureBuffer[FrameNum-3][i]+a[6]*FeatureBuffer[FrameNum-2][i]+a[7]*FeatureBuffer[FrameNum-1][i]));
		FeatureBuffer[FrameNum-3][D]=(float)(2.0*(a[0]*FeatureBuffer[FrameNum-7][i]+a[1]*FeatureBuffer[FrameNum-6][i]+a[2]*FeatureBuffer[FrameNum-5][i]+a[3]*FeatureBuffer[FrameNum-4][i]+a[5]*FeatureBuffer[FrameNum-2][i]+a[6]*FeatureBuffer[FrameNum-1][i]));
		FeatureBuffer[FrameNum-2][D]=(float)(2.0*(a[0]*FeatureBuffer[FrameNum-6][i]+a[1]*FeatureBuffer[FrameNum-5][i]+a[2]*FeatureBuffer[FrameNum-4][i]+a[3]*FeatureBuffer[FrameNum-3][i]+a[5]*FeatureBuffer[FrameNum-1][i]) );
		FeatureBuffer[FrameNum-1][D]=(float)(2.0*(a[0]*FeatureBuffer[FrameNum-5][i]+a[1]*FeatureBuffer[FrameNum-4][i]+a[2]*FeatureBuffer[FrameNum-3][i]+a[3]*FeatureBuffer[FrameNum-2][i]+a[5]*FeatureBuffer[FrameNum-1][i]) );
	}
	//�����м�֡��һ�ײ�ֵ���
	for( FrameNo = 4; FrameNo < FrameNum-4; FrameNo++ ) 
	{
		for( i = 0; i < D; i++ )
			FeatureBuffer[FrameNo][D] = 
			a[0]*FeatureBuffer[FrameNo-4][i] +
			a[1]*FeatureBuffer[FrameNo-3][i] +
			a[2]*FeatureBuffer[FrameNo-2][i] +
			a[3]*FeatureBuffer[FrameNo-1][i] +
			a[5]*FeatureBuffer[FrameNo+1][i] +
			a[6]*FeatureBuffer[FrameNo+2][i] +
			a[7]*FeatureBuffer[FrameNo+3][i] +
			a[8]*FeatureBuffer[FrameNo+4][i];
	}

	/************************************/
	//����ͷ��֡�������֡�Ķ��ײ�ֵ���
	for( i = 0; i < D; i++ )
	{
		//��0֡
		FeatureBuffer[0][2*D]=(float)(2.0*(a[3]*FeatureBuffer[0][D]+a[5]*FeatureBuffer[1][D]+a[6]*FeatureBuffer[2][D]+a[7]*FeatureBuffer[3][D]+a[8]*FeatureBuffer[4][D]));
		//��1֡
		FeatureBuffer[1][2*D]=(float)(2.0*(a[3]*FeatureBuffer[0][D]+a[5]*FeatureBuffer[2][D]+a[6]*FeatureBuffer[3][D]+a[7]*FeatureBuffer[4][D]+a[8]*FeatureBuffer[5][D]));
		//��2֡
		FeatureBuffer[2][2*D]=(float)(2.0*(a[2]*FeatureBuffer[0][D]+a[3]*FeatureBuffer[1][D]+a[5]*FeatureBuffer[4][D]+a[6]*FeatureBuffer[5][D]+a[7]*FeatureBuffer[6][D]+a[8]*FeatureBuffer[7][D]));
		//��3֡
		FeatureBuffer[3][2*D]=(float)(2.0*(a[1]*FeatureBuffer[0][D]+a[2]*FeatureBuffer[1][D]+a[3]*FeatureBuffer[2][D]+a[5]*FeatureBuffer[4][D]+a[6]*FeatureBuffer[5][D]+a[7]*FeatureBuffer[6][D]+a[8]*FeatureBuffer[7][D]));
		FeatureBuffer[FrameNum-4][2*D]=(float)(2.0*(a[0]*FeatureBuffer[FrameNum-8][D]+a[1]*FeatureBuffer[FrameNum-7][D]+a[2]*FeatureBuffer[FrameNum-6][D]+a[3]*FeatureBuffer[FrameNum-5][D]+a[5]*FeatureBuffer[FrameNum-3][D]+a[6]*FeatureBuffer[FrameNum-2][D]+a[7]*FeatureBuffer[FrameNum-1][D]));
		FeatureBuffer[FrameNum-3][2*D]=(float)(2.0*(a[0]*FeatureBuffer[FrameNum-7][D]+a[1]*FeatureBuffer[FrameNum-6][D]+a[2]*FeatureBuffer[FrameNum-5][D]+a[3]*FeatureBuffer[FrameNum-4][D]+a[5]*FeatureBuffer[FrameNum-2][D]+a[6]*FeatureBuffer[FrameNum-1][D]));
		FeatureBuffer[FrameNum-2][2*D]=(float)(2.0*(a[0]*FeatureBuffer[FrameNum-6][D]+a[1]*FeatureBuffer[FrameNum-5][D]+a[2]*FeatureBuffer[FrameNum-4][D]+a[3]*FeatureBuffer[FrameNum-3][D]+a[5]*FeatureBuffer[FrameNum-1][D]) );
		FeatureBuffer[FrameNum-1][2*D]=(float)(2.0*(a[0]*FeatureBuffer[FrameNum-5][D]+a[1]*FeatureBuffer[FrameNum-4][D]+a[2]*FeatureBuffer[FrameNum-3][D]+a[3]*FeatureBuffer[FrameNum-2][D]+a[5]*FeatureBuffer[FrameNum-1][D]) );
	}
	//����� FrameNo ֡���׵Ķ��ײ��
	for( FrameNo = 4; FrameNo < FrameNum-4; FrameNo++ ) 
	{
		for( i = 0; i < D; i++ )
			FeatureBuffer[FrameNo][2*D] = 
			a[0]*FeatureBuffer[FrameNo-4][D] +
			a[1]*FeatureBuffer[FrameNo-3][D] +
			a[2]*FeatureBuffer[FrameNo-2][D] +
			a[3]*FeatureBuffer[FrameNo-1][D] +
			a[5]*FeatureBuffer[FrameNo+1][D] +
			a[6]*FeatureBuffer[FrameNo+2][D] +
			a[7]*FeatureBuffer[FrameNo+3][D] +
			a[8]*FeatureBuffer[FrameNo+4][D];
	}

	//����ͷ��֡�������֡��һ���������

	//��0֡
	FeatureBuffer[0][3*D+1]=(float)(0.8*(a[3]*FeatureBuffer[0][3*D]+a[5]*FeatureBuffer[1][3*D]+a[6]*FeatureBuffer[2][3*D]+a[7]*FeatureBuffer[3][3*D]+a[8]*FeatureBuffer[4][3*D]));
	//��1֡
	FeatureBuffer[1][3*D+1]=(float)(0.8*(a[3]*FeatureBuffer[0][3*D]+a[5]*FeatureBuffer[2][3*D]+a[6]*FeatureBuffer[3][3*D]+a[7]*FeatureBuffer[4][3*D]+a[8]*FeatureBuffer[5][3*D]));
	//��2֡
	FeatureBuffer[2][3*D+1]=(float)(0.8*(a[2]*FeatureBuffer[0][3*D]+a[3]*FeatureBuffer[1][3*D]+a[5]*FeatureBuffer[4][3*D]+a[6]*FeatureBuffer[5][3*D]+a[7]*FeatureBuffer[6][3*D]+a[8]*FeatureBuffer[7][3*D]));
	//��3֡
	FeatureBuffer[3][3*D+1]=(float)(0.8*(a[1]*FeatureBuffer[0][3*D]+a[2]*FeatureBuffer[1][3*D]+a[3]*FeatureBuffer[2][3*D]+a[5]*FeatureBuffer[4][3*D]+a[6]*FeatureBuffer[5][3*D]+a[7]*FeatureBuffer[6][3*D]+a[8]*FeatureBuffer[7][3*D]));
	FeatureBuffer[FrameNum-4][3*D+1]=(float)(0.8*(a[0]*FeatureBuffer[FrameNum-8][3*D]+a[1]*FeatureBuffer[FrameNum-7][3*D]+a[2]*FeatureBuffer[FrameNum-6][3*D]+a[3]*FeatureBuffer[FrameNum-5][3*D]+a[5]*FeatureBuffer[FrameNum-3][3*D]+a[6]*FeatureBuffer[FrameNum-2][3*D]+a[7]*FeatureBuffer[FrameNum-1][3*D]));
	FeatureBuffer[FrameNum-3][3*D+1]=(float)(0.8*(a[0]*FeatureBuffer[FrameNum-7][3*D]+a[1]*FeatureBuffer[FrameNum-6][3*D]+a[2]*FeatureBuffer[FrameNum-5][3*D]+a[3]*FeatureBuffer[FrameNum-4][3*D]+a[5]*FeatureBuffer[FrameNum-2][3*D]+a[6]*FeatureBuffer[FrameNum-1][3*D]));
	FeatureBuffer[FrameNum-2][3*D+1]=(float)(0.8*(a[0]*FeatureBuffer[FrameNum-6][3*D]+a[1]*FeatureBuffer[FrameNum-5][3*D]+a[2]*FeatureBuffer[FrameNum-4][3*D]+a[3]*FeatureBuffer[FrameNum-3][3*D]+a[5]*FeatureBuffer[FrameNum-1][3*D]) );
	FeatureBuffer[FrameNum-1][3*D+1]=(float)(0.8*(a[0]*FeatureBuffer[FrameNum-5][3*D]+a[1]*FeatureBuffer[FrameNum-4][3*D]+a[2]*FeatureBuffer[FrameNum-3][3*D]+a[3]*FeatureBuffer[FrameNum-2][3*D]+a[5]*FeatureBuffer[FrameNum-1][3*D]) );
	//�����м�֡��һ���������
	for( FrameNo = 4; FrameNo < FrameNum-4; FrameNo++ )
		FeatureBuffer[FrameNo][3*D+1] =
		(float)(0.4*(a[0]*FeatureBuffer[FrameNo-4][3*D] +
		a[1]*FeatureBuffer[FrameNo-3][3*D] +
		a[2]*FeatureBuffer[FrameNo-2][3*D] +
		a[3]*FeatureBuffer[FrameNo-1][3*D] +
		a[5]*FeatureBuffer[FrameNo+1][3*D] +
		a[6]*FeatureBuffer[FrameNo+2][3*D] +
		a[7]*FeatureBuffer[FrameNo+3][3*D] +
		a[8]*FeatureBuffer[FrameNo+4][3*D]));

	//����ͷ��֡�������֡�Ķ����������
	FeatureBuffer[0][3*D+2] = (float)(1.5*(FeatureBuffer[1][3*D+1]-FeatureBuffer[0][3*D+1]));
	FeatureBuffer[FrameNum-1][3*D+2] = (float)(1.5*(FeatureBuffer[FrameNum-1][3*D+1] - FeatureBuffer[FrameNum-2][3*D+1]));
	//////�����м�֡�Ķ����������
	for(FrameNo=1;FrameNo<FrameNum-1;FrameNo++)
		FeatureBuffer[FrameNo][3*D+2] = (float)(1.5*(FeatureBuffer[FrameNo+1][3*D+1] - FeatureBuffer[FrameNo-1][3*D+1]));

	return TRUE;
}



bool get20dBEnergyGeometryAveragMfcc(short *SpeecBuffer,float (*FeatureBuffer)[DIM],long FrameNum)
{
long	i,FrameCounter,FrameNo;
float	FrameEnergy,MaxEn,AveEn;

	//�� 0 ~ 13 ά��ŵ��ף��� 14 ~ 27 ά���һ�׵���
	for( FrameNo = 0 ; FrameNo < FrameNum; FrameNo++ )
	{
		//�� 0 ~ 13 ά��ŵ���
		FrameEnergy = CMfcc.DoSMFCC( &SpeecBuffer[FrameNo * FRAME_STEP], FeatureBuffer[FrameNo] );	
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
		FeatureBuffer[0][D]=(float)(2.0*(a[1]*FeatureBuffer[0][i]+a[3]*FeatureBuffer[1][i]));		//��0֡
		FeatureBuffer[1][D]=(float)(2.0*(a[1]*FeatureBuffer[0][i]+a[3]*FeatureBuffer[2][i]));		//��1֡
		FeatureBuffer[FrameNum-2][D]=(float)(2.0*(a[1]*FeatureBuffer[FrameNum-3][i]+a[3]*FeatureBuffer[FrameNum-1][i]) );
		FeatureBuffer[FrameNum-1][D]=(float)(2.0*(a[1]*FeatureBuffer[FrameNum-2][i]+a[3]*FeatureBuffer[FrameNum-1][i]) );
	}
	//�����м�֡��һ�ײ�ֵ���
	for( FrameNo = 2; FrameNo < FrameNum-2; FrameNo++ ) 
	{
		for( i = 0; i < D; i++ )
			FeatureBuffer[FrameNo][D] = 
				a[0]*FeatureBuffer[FrameNo-2][i] +
				a[1]*FeatureBuffer[FrameNo-1][i] +
				a[3]*FeatureBuffer[FrameNo+1][i] +
				a[4]*FeatureBuffer[FrameNo+2][i];
	}

	/************************************/
	//����ͷ��֡�������֡�Ķ��ײ�ֵ���
	for( i = 0; i < D; i++ )
	{
		FeatureBuffer[0][2*D]=(float)(2.0*(a[1]*FeatureBuffer[0][D]+a[3]*FeatureBuffer[1][D]));
		FeatureBuffer[1][2*D]=(float)(2.0*(a[1]*FeatureBuffer[0][D]+a[3]*FeatureBuffer[2][D]));
		FeatureBuffer[FrameNum-2][2*D]=(float)(2.0*(a[1]*FeatureBuffer[FrameNum-3][D]+a[3]*FeatureBuffer[FrameNum-1][D]) );
		FeatureBuffer[FrameNum-1][2*D]=(float)(2.0*(a[1]*FeatureBuffer[FrameNum-2][D]+a[3]*FeatureBuffer[FrameNum-1][D]) );
	}
	//����� FrameNo ֡���׵Ķ��ײ��
	for( FrameNo = 2; FrameNo < FrameNum-2; FrameNo++ ) 
	{
		for( i = 0; i < D; i++ )
			FeatureBuffer[FrameNo][2*D] = 
				a[0]*FeatureBuffer[FrameNo-2][D] +
				a[1]*FeatureBuffer[FrameNo-1][D] +
				a[3]*FeatureBuffer[FrameNo+1][D] +
				a[4]*FeatureBuffer[FrameNo+2][D];
	}

	//����ͷ��֡�������֡��һ���������
	FeatureBuffer[0][3*D+1]	= (float)(0.8*(a[1]*FeatureBuffer[0][3*D] + a[3]*FeatureBuffer[1][3*D]));
	FeatureBuffer[1][3*D+1]	= (float)(0.8*(a[1]*FeatureBuffer[0][3*D] + a[3]*FeatureBuffer[2][3*D]));
	FeatureBuffer[FrameNum-2][3*D+1] = (float)(0.8*(a[1]*FeatureBuffer[FrameNum-3][3*D] + a[3]*FeatureBuffer[FrameNum-1][3*D]));
	FeatureBuffer[FrameNum-1][3*D+1] = (float)(0.8*(a[1]*FeatureBuffer[FrameNum-2][3*D] + a[3]*FeatureBuffer[FrameNum-1][3*D]));
	//�����м�֡��һ���������
	for( FrameNo = 2; FrameNo < FrameNum-2; FrameNo++ )
		FeatureBuffer[FrameNo][3*D+1] = 
			(float)(0.4*(a[0]*FeatureBuffer[FrameNo-2][3*D]
					   + a[1]*FeatureBuffer[FrameNo-1][3*D]
					   + a[3]*FeatureBuffer[FrameNo+1][3*D]
					   + a[4]*FeatureBuffer[FrameNo+2][3*D]));

	//����ͷ��֡�������֡�Ķ����������
	FeatureBuffer[0][3*D+2] = (float)(1.5*(FeatureBuffer[1][3*D+1]-FeatureBuffer[0][3*D+1]));
	FeatureBuffer[FrameNum-1][3*D+2] = (float)(1.5*(FeatureBuffer[FrameNum-1][3*D+1] - FeatureBuffer[FrameNum-2][3*D+1]));
	//�����м�֡�Ķ����������
	for(FrameNo=1;FrameNo<FrameNum-1;FrameNo++)
		FeatureBuffer[FrameNo][3*D+2] = (float)(1.5*(FeatureBuffer[FrameNo+1][3*D+1] - FeatureBuffer[FrameNo-1][3*D+1]));

	return TRUE;
}


