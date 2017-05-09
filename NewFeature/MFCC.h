#ifndef	_XIAOXI_CMFCC_H_
#define	_XIAOXI_CMFCC_H_

#include "CFFT.h"

class CMFCC:public CFFTanalyser
{
	private:
		float	fres;
		long	FFT_LEN,FrameLen, SubBandNum, CepstrumNum;
		float	*cosTab;			//cosine table for DCT transformation 
		float	*hamWin;			//hamming window coefficients
		float	*cepWin;			//���״���Ȩϵ��
		float	*MelBandBoundary;	//��¼ÿ���Ӵ�����ʼMEL�̶�
		float	*SubBandWeight;		//weighting coefficients for 
									//energy of each FFT frequence component
		int		*SubBandIndex;		//mapping of the FFT frequence component to sub-band No.
		float	*SubBandEnergy;		//for accumulating the corresponding subband energy
		float	*FFTFrame,*Cepstrum;
		float	*ImgBuf;
	public:
		CMFCC(
				float	sampleF,			//���ݵĲ����ʣ�Hz��
				long	DefFFTLen=512,		//Ĭ�ϵ�FFT����Ϊ512��
				long	DefFrameLen=320,	//Ĭ�ϵ�֡��Ϊ320��
				long	DefSubBandNum=24,	//Ĭ�ϵ��Ӵ��ĸ���Ϊ24
				long	DefCepstrumNum=14	//Ĭ�ϵĵ���ϵ������Ϊ14
			 );
		~CMFCC();
		float	Mel(int k);
		float	DoMFCC(float *inData, float *outVect);	//���ص�ǰ֡������
		float	DoCepMFCC(short *inData, float *outVect);
		float	DoSMFCC(short *inData, float *outVect);
				float	DoSSMFCC(short *inData, float *outVect);
};	

#endif
