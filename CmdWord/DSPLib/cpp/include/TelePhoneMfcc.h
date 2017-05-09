#pragma once

#include "CFFT.h"

class CTelePhoneMfcc:public CFFTanalyser
{
public:
	CTelePhoneMfcc(
		float	CutOffFreq		= 3400,		//�źŴ���,16.8���Ӵ�
		float	sampleFreq		= 8000,		//���ݵĲ����ʣ�Hz��
		long	DefFFTLen		= 512,		//Ĭ�ϵ�FFT����Ϊ512��
		long	DefFrameLen		= 160,		//Ĭ�ϵ�֡��Ϊ160��(20mS@8KHz)
		long	DefSubBandNum	= 17,		//Ĭ�ϵ��Ӵ��ĸ���Ϊ24@16KHz
		long	DefCepstrumNum	= 14		//Ĭ�ϵĵ���ϵ������Ϊ14
	);

	~CTelePhoneMfcc(void);

private:
		long	m_MaxFFT;
		float	m_fres;				//FFT������Ƶ�ʷֱ��ʣ���������Ƶ�ʵ�Mel�̶ȵĻ���
		long	m_FFT_LEN,m_FrameLen,m_SubBandNum, m_DCT_DIM, m_CepstrumNum;
		float	*cosTab;			//cosine table for DCT transformation 
		float	*hamWin;			//hamming window coefficients
		float	*cepWin;			//���״���Ȩϵ��
		float	*MelBandBoundary;	//��¼ÿ���Ӵ�����ʼMEL�̶�
		float	*SubBandWeight;		//weighting coefficients for 
									//energy of each FFT frequence component
		long	*SubBandIndex;		//mapping of the FFT frequence component to sub-band No.
		float	*SubBandEnergy;		//for accumulating the corresponding subband energy
		float	*FFTFrame,*Cepstrum;
	public:
/*	
		~CMFCC();
*/
		float	Mel(int k);
		float	DoMFCC(short *inData, float *outVect);	//���ص�ǰ֡������


};

