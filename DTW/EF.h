#include "CFFT.h"
class EF:public CFFTanalyser{
private:
	long	FFT_LEN,FrameLen, SubBandNum, CepstrumNum;
	float	*hamWin;			//hamming window coefficients
	float	*FFTFrame;
public:
	EF(
		float sampleF,
		long	DefFFTLen=512,		//Ĭ�ϵ�FFT����Ϊ512��
		long	DefFrameLen=320	//Ĭ�ϵ�֡��Ϊ320��
		);
	~EF();
	float DoEF(short* inData, float* outVect);




};