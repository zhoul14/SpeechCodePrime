#include <math.h>
#include "..\include\TelePhoneMfcc.h"

#define PREE        0.98f
#define MELFLOOR    1.0f


CTelePhoneMfcc::CTelePhoneMfcc(float CutOffFreq,float SampleFreq,long DefFFTLen,long DefFrameLen,long DefSubBandNum, long DefCepstrumNum):CFFTanalyser(DefFFTLen)
{
long	M, SubBandNo, i, j;
float	ms,pi_factor, mfnorm, a, melk, t;
	m_DCT_DIM		= 24;					//DCT�任��ά��		
	m_FrameLen		= DefFrameLen;
	m_SubBandNum	= DefSubBandNum;
	m_CepstrumNum	= DefCepstrumNum;

	m_FFT_LEN = GetFFTAnalyseLen();
//	std::cout <<"FFT_LEN = "<<FFT_LEN<<std::endl;

	cosTab = new float[(m_DCT_DIM+1)*(m_DCT_DIM+1)];	//DCT�任ϵ��
	hamWin = new float[m_FrameLen];						//hamming ��ϵ��
	cepWin = new float[m_DCT_DIM+1];
	MelBandBoundary = new float[m_DCT_DIM+2];
	SubBandWeight	= new float[m_FFT_LEN/2];			//weighting coefficients for 
														//energy of each FFT frequence component
	SubBandIndex	= new long[m_FFT_LEN/2];			//mapping of the frequence to sub-band No.
	SubBandEnergy	= new float[m_DCT_DIM+2];			//for accumulating the corresponding 

	FFTFrame = new float[m_FFT_LEN];
	Cepstrum = new float[m_DCT_DIM + 1 ];

	m_fres = SampleFreq/(m_FFT_LEN*700.0f);				//FFTƵ�ʷ����ķֱ��ʣ�������Ƶ�ʵ�Mel�̶ȵĻ���

	//caculating the mel scale sub-band boundary
	//�����Ӵ�0��ϵ����ֱ��������������MFCC�������㣬�����Ӵ�����Ҫ��һ��
	M	 = m_SubBandNum + 1;	

	//�����ֹƵ������Ӧ��FFT��
	m_MaxFFT =(long)( (CutOffFreq/SampleFreq)*m_FFT_LEN );	
	if( m_MaxFFT > m_FFT_LEN/2 ) m_MaxFFT = m_FFT_LEN/2;

//	ms	 = Mel(m_FFT_LEN/2);		//����1/2����������Ӧ��MEL�̶�
	ms	 = Mel(m_MaxFFT);			//�����ֹƵ������Ӧ��MEL�̶ȣ�2012��11��10�޸�

	//Note that the sub-band 0 is not used for cepstrum caculating
	for ( SubBandNo = 0; SubBandNo <= M; SubBandNo++ )
	{	//����ÿ���Ӵ�����ʼMEL�̶�
		MelBandBoundary[SubBandNo] = ( (float)SubBandNo/(float)M )*ms;
	}

	//mapping the FFT frequence component into the corresponding sub-band
//	for ( i=0,SubBandNo=1; i< FFT_LEN/2; i++)
	for ( i=0,SubBandNo=1; i< m_MaxFFT; i++)
	{
		melk = Mel(i);
		while( MelBandBoundary[SubBandNo] < melk ) SubBandNo++;
		SubBandIndex[i] = SubBandNo-1;
	}
	//caculating the weighting coefficients for each FFT frequence components	
//	for(i=0; i< FFT_LEN/2; i++)
	for(i=0; i< m_MaxFFT; i++)
	{	//���Ӵ�����ʼMELƵ��Ϊ���ģ��������Ǵ���Ȩϵ��
		SubBandNo = SubBandIndex[i];
		SubBandWeight[i] = (MelBandBoundary[SubBandNo+1]-Mel(i))/(MelBandBoundary[SubBandNo+1]-MelBandBoundary[SubBandNo]);
	}

	//DCT�任ϵ��
	pi_factor = (float)( asin(1.0)*2.0/(float)m_DCT_DIM );
	mfnorm	  = (float)sqrt(2.0f/(float)m_DCT_DIM);
	for( i=1; i<= m_CepstrumNum; i++ )
	{
		t = (float)i*pi_factor;
		for(j=1; j<=m_DCT_DIM; j++)
			cosTab[i*(m_DCT_DIM+1)+j] = (float)cos(t*(j-0.5f))*mfnorm;
	}

	a =(float)( asin(1.0)*4/(m_FrameLen-1) );
	for( i = 0 ; i < m_FrameLen; i++ )
		hamWin[i] = 0.54f - 0.46f * (float)cos(a*i);

	for(i=1;i<=m_CepstrumNum;i++)
		cepWin[i-1] = (float)i * (float)exp(-(float)i*2.0/(float)m_CepstrumNum);
}



CTelePhoneMfcc::~CTelePhoneMfcc(void)
{
	delete []cosTab;
	delete []hamWin;
	delete []cepWin;
	delete []MelBandBoundary;
	delete []SubBandWeight;
	delete []SubBandIndex;
	delete []SubBandEnergy;
	delete []FFTFrame;
	delete []Cepstrum;
}




float CTelePhoneMfcc::Mel(int k)
{
	return (float)(1127 * log(1 + k*m_fres));
}


float CTelePhoneMfcc::DoMFCC(short *inData, float *outVect)
{
int		i, SubBandNo;
float	FrameEnergy,ek,temp1,temp2;
float	*pCosTable;

	FrameEnergy = 0.0f;
	for( i = 0 ; i < m_FrameLen; i++ )
	{
		FFTFrame[i] = (float)inData[i];
		FrameEnergy += (FFTFrame[i]*FFTFrame[i]);
	}
	for( i= m_FrameLen-1; i > 0; i-- )	//���������źŲ�ֲ����мӴ�
	{
		FFTFrame[i] -= FFTFrame[i-1]*PREE;
		FFTFrame[i] *= hamWin[i];
	}
	FFTFrame[0] *= (1.0f-PREE);
	FFTFrame[0] *= hamWin[0];
	for( i = m_FrameLen; i< m_FFT_LEN; i++ )	FFTFrame[i]=0.0f;

	// ����FFT�任
	DoRealFFT(FFTFrame);
	//����ÿ���Ӵ�������
	for( SubBandNo = 0 ; SubBandNo <= m_DCT_DIM; SubBandNo++ )
	{
		SubBandEnergy[SubBandNo]=0.0f;
	}

//	for( i = 1; i < FFT_LEN/2; i++ )	//the component at zero frequence is discarded
	for( i = 1; i < m_MaxFFT; i++ )	//the component at zero frequence is discarded
	{
		temp1 = FFTFrame[2*i]; temp2 = FFTFrame[2*i+1];
		ek = temp1*temp1+temp2*temp2;	// �����k��Ƶ�ʴ�������
		SubBandNo = SubBandIndex[i];	// �����õ�k��Ƶ���������Ӵ���
		temp1 = SubBandWeight[i]*ek;
		if( SubBandNo > 0 )				SubBandEnergy[SubBandNo]	+= temp1;		//�Ӵ�0�ǲ��ۼ�������
		if( SubBandNo < m_SubBandNum)	SubBandEnergy[SubBandNo+1]	+= ek-temp1;
	}
	// ����ÿ���˲�����������Ķ���ֵ
	for( SubBandNo = 1; SubBandNo<= m_SubBandNum; SubBandNo++ )
	{
		temp1 = SubBandEnergy[SubBandNo];
		if(temp1<MELFLOOR)	temp1 = MELFLOOR;		// clipping 
		SubBandEnergy[SubBandNo] = (float)(0.5*log(temp1));    
	}

	pCosTable = cosTab + m_DCT_DIM + 1;
	for( i = 1; i <= m_CepstrumNum; i++ )
	{
		Cepstrum[i]=0.0f; 
		for( SubBandNo = 1 ; SubBandNo <= m_DCT_DIM; SubBandNo++ )
			Cepstrum[i] += SubBandEnergy[SubBandNo]*pCosTable[SubBandNo];
		Cepstrum[i] *= cepWin[i-1];
//		cout<<"["<<i<<"]"<<Cepstrum[i]<<endl;
		pCosTable = pCosTable + m_DCT_DIM + 1; 
	}
// Output the cepstrum and energy
	for( i = 1; i <= m_CepstrumNum; i++ )
		outVect[i-1] = Cepstrum[i];	//�����ǰ֡�ĵ���ϵ��
	return(FrameEnergy);			//���ص�ǰ֡������
}





