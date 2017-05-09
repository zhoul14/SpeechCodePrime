#ifndef _XIAOXI_MFCC46_H
#define	_XIAOXI_MFCC46_H

#define DIM     45                  /* feature vector dimemsion */
#define D       14

#define	FRAME_STEP	160
#define	FRAME_LEN	320

#define	DEFAULT_SAMPLE_RATE		16000		// ������
#define	DEFAULT_FFT_LEN			512			// FFT����
#define	DEFAULT_FRAME_LEN		FRAME_LEN	// ������֡��
#define	DEFAULT_SUB_BAND_NUM	24			// �Ӵ�����
#define	DEFAULT_CEP_COEF_NUM	14			// ����ϵ������

bool get20dBEnergyGeometryAveragMfcc(short *sentData,float (*ftrVect)[DIM],long FrameNum);


#endif



