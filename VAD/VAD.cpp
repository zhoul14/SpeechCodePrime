#include "VAD.h"


CVAD::CVAD(void)
{
	m_pfDoRecg = 0;
	m_pRecognizer = 0;
	m_hRecgComplete=CreateEvent(
						NULL,	
						FALSE,			//FALSE= create auto-reset event, TRUE=manual reset
						FALSE,			//Nonsignaled state
						_T("RecgCompleteEvent"));
}

CVAD::~CVAD(void)
{
	::CloseHandle(m_hRecgComplete);
}

// ������������Ļص�����
void CVAD::SoundDataCallBack(void *pCallbackInfo,void *Buffer,long BytesLength)
{
	CVAD *pVDA =(CVAD *)pCallbackInfo;
	pVDA->DoVAD((short *)Buffer,BytesLength/sizeof(short));	//����������VAD�����
}

void	CVAD::IniClass(void *pRecognizer)
{
	m_SpeechBuffer.IniClass();
	m_SpeechDetection.IniClass();

	m_pRecognizer = pRecognizer;
	//��ʼ������
	m_Recorder.m_pCallbackInfo = this;	//���ûص�����
	m_Recorder.SetWaveDataReadyCallbackFunction(SoundDataCallBack);	//����¼���ص�����
	//��ʼ¼��
	m_Recorder.Open();
	m_Recorder.Record();
}


void	CVAD::ResetVAD(void)
{
	m_SpeechBuffer.ClearSpeechBuffer();
	m_SpeechDetection.ResetDetector();
}

void	CVAD::SetRecgCallback(long (*pfDoRecg)(void *pInfo,short *SpeechBuf,long DataNum))
{
	m_pfDoRecg = pfDoRecg;
}




//����ֵ��-1��ʾ��δ��⵽����
//		  -2��ʾ�������ݷ�������
//        >= 0 ��ʾʶ����

long	CVAD::DoVAD(short *pSpeechBuf,long DataNum)
{
long	FrameLen,SpeechStatus;
short	tmpBuffer[320];

	FrameLen = m_SpeechDetection.m_FrameLen;
	if( DataNum != FrameLen )	return -2;
	if( FrameLen > 320 )		return -2;			//����ʱ��������Խ�籣��

//	m_SpeechBuffer.WriteSpeechToBuffer(pSpeechBuf,FrameLen);	
//	SpeechStatus = m_SpeechBuffer.WaitForSpeech(tmpBuffer,FrameLen);	//���10mS������֡
//	if( SpeechStatus == FrameLen )
	{
		//��10mS�����͵�����������н��ж˵��о�
//		if( m_SpeechDetection.DetectSpeeeh(tmpBuffer)==SPEECH_DETECTED )
		if( m_SpeechDetection.DetectSpeeeh(pSpeechBuf)==SPEECH_DETECTED )
		{
//			this->m_Recorder.Close();
			//��⵽����
			short *pVADSpeechBuf	= m_SpeechDetection.m_SpeechBuffer;
			long VADDataNum			= m_SpeechDetection.m_SpeechFrameNum*m_SpeechDetection.m_FrameLen;

			long RecgReturn = -1;
			if( m_pfDoRecg != 0 )
			{
				RecgReturn =  m_pfDoRecg(m_pRecognizer,pVADSpeechBuf,VADDataNum);
				::SetEvent(m_hRecgComplete);		//֪ͨ���߳�ʶ�����	
			}
//			this->m_Recorder.Open();
//			this->m_Recorder.Record();
			this->ResetVAD();
			return RecgReturn;
		}
		else
		{
			return -1;	//��ʾ��δ��⵽����
		}
	}
	return -2;	//����
}
	
void CVAD::WaitForSpeechDetected(void)
{
	WaitForSingleObject(m_hRecgComplete,INFINITE);
}
