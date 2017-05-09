//#include "stdafx.h"
#include "windows.h"
#include "atlstr.h"
#include <process.h>
#include "CRecorder.h"

DWORD WINAPI pWavInMonitorThread(LPVOID lpArg)
{
CRECORDER	*pRecorder = (CRECORDER *)lpArg;
long	AudioBlockNum,ReadPtr;

	HANDLE hThread = ::GetCurrentThread();
	::SetThreadPriority(hThread,THREAD_PRIORITY_TIME_CRITICAL);
	pRecorder->m_bMonitorThreadExit = false;
	while(1)
	{
		::WaitForSingleObject(pRecorder->m_hAuidoReadyEvent,INFINITE);
		if( pRecorder->m_bDeviceOpen == false ) 
			break;

		::EnterCriticalSection( &(pRecorder->m_csNotifyAuidoBuffer) );
		AudioBlockNum = pRecorder->m_AudioBlockNum;
		::LeaveCriticalSection( &(pRecorder->m_csNotifyAuidoBuffer) );

		for( long i = 0 ; i < AudioBlockNum ; i++ )
		{
			ReadPtr = pRecorder->m_ReadPtr;
			pRecorder->m_ReadPtr++;
			pRecorder->m_ReadPtr = pRecorder->m_ReadPtr % pRecorder->m_DeviceBufferNum;	//�޸Ķ���ַ
			
			::EnterCriticalSection(&(pRecorder->m_csSetCallback));
			if( pRecorder->pWaveDataReadyCallBackFunction != NULL )
			{	// �ص�Ӧ�ó����ṩ�Ĵ�����
				pRecorder->pWaveDataReadyCallBackFunction( 
					pRecorder->m_pCallbackInfo,
					pRecorder->m_AudioBlockBuffer + ReadPtr*pRecorder->m_DeviceBlockBufferByteSize, 
					pRecorder->m_AudioBlockBufferLen[ReadPtr] 
					);
			}
			::LeaveCriticalSection(&(pRecorder->m_csSetCallback));
			
			::EnterCriticalSection( &(pRecorder->m_csNotifyAuidoBuffer) );
			pRecorder->m_AudioBlockNum--;
			::LeaveCriticalSection( &(pRecorder->m_csNotifyAuidoBuffer) );
		}
	}
	pRecorder->m_bMonitorThreadExit = true;
	return 0;
}


// ���߳���������¼���豸�����Ļص���Ϣ
DWORD WINAPI pWavInCallbackThread(LPVOID lpArg)
{
CRECORDER	*pRecorder = (CRECORDER *)lpArg;
LPWAVEHDR	lpWaveInHdr;	
MSG			Msg;

	pRecorder->m_bWaveInThreadExit = false;

	HANDLE hThread = ::GetCurrentThread();
	::SetThreadPriority(hThread,THREAD_PRIORITY_TIME_CRITICAL);
	
	// ���߳���Ϣ���н���ѭ����ѯ��ֱ�����յ�WM_QUIT��Ϣ�����ֹ�߳�
	// ϵͳ�ڴ����߳�ʱ����Ĭ�ϵ�����²��ᴴ���̵߳���Ϣ���У�ֱ�����߳��е�һ��
	// ����USER32����Windows GDI����ʱ���Ż��Զ������̵߳���Ϣ���С�

	while( GetMessage(&Msg,NULL,0,0) )
	{	// ��һ�ε���GetMessage����ʱ���߳���Ϣ���б�����
		switch(Msg.message)
		{
		case MM_WIM_CLOSE:
			// pRecorder->m_bDeviceOpen = false;	//���ﲻ�������豸��־����Ϊ�п������߳���Ӧ����Ϣ֮ǰ��
													//�豸�ֱ����´򿪡�
			PostQuitMessage(0);						//���̵߳���Ϣ����Ͷ��WM_QUIT��Ϣ�����̷Ż�
			break;
		case MM_WIM_DATA:

			::EnterCriticalSection( &(pRecorder->m_csDevice) );
				lpWaveInHdr=(LPWAVEHDR)Msg.lParam;
				//¼�����ݺ����ص�
				if(  lpWaveInHdr->dwBytesRecorded > 0  )
				{
					::EnterCriticalSection( &(pRecorder->m_csNotifyAuidoBuffer) );
					if( pRecorder->m_AudioBlockNum < pRecorder->m_DeviceBufferNum )
					{	//�������ݵ���������
						memcpy(	pRecorder->m_AudioBlockBuffer + pRecorder->m_WritePtr*pRecorder->m_DeviceBlockBufferByteSize,
								lpWaveInHdr->lpData,
								lpWaveInHdr->dwBytesRecorded 
							   );
						pRecorder->m_AudioBlockBufferLen[pRecorder->m_WritePtr] = lpWaveInHdr->dwBytesRecorded;
						pRecorder->m_WritePtr++;
						pRecorder->m_WritePtr = pRecorder->m_WritePtr%pRecorder->m_DeviceBufferNum;
						pRecorder->m_AudioBlockNum++;
						SetEvent(pRecorder->m_hAuidoReadyEvent);
					}
					::LeaveCriticalSection( &(pRecorder->m_csNotifyAuidoBuffer) );
				}

				if( pRecorder->m_bDeviceOpen == true )
				{
					waveInUnprepareHeader( pRecorder->hWaveIn, lpWaveInHdr, sizeof(WAVEHDR) );
					lpWaveInHdr->dwBufferLength  = pRecorder->m_DeviceBlockBufferByteSize;
					lpWaveInHdr->dwBytesRecorded = 0L ;
					// lpWaveInHdr->dwUser       = 0L ;
					lpWaveInHdr->dwFlags         = 0L ;
					lpWaveInHdr->dwLoops         = 0L ;
					lpWaveInHdr->lpNext          = NULL ;
					lpWaveInHdr->reserved        = 0L ;
					waveInPrepareHeader( pRecorder->hWaveIn , lpWaveInHdr , sizeof(WAVEHDR) );
					waveInAddBuffer( pRecorder->hWaveIn, lpWaveInHdr, sizeof(WAVEHDR) );
				}
			::LeaveCriticalSection( &(pRecorder->m_csDevice) );
			break;
		case MM_WIM_OPEN:
			pRecorder->m_bDeviceOpen = true;
			break;
		default:
			break;
		}
	}
	pRecorder->m_bWaveInThreadExit = true;
	return 0;
}


CRECORDER::CRECORDER(long DefSampleRate,long DefChannelNum,long DefBitNum,long DefBlockSampleLen,long DefDeviceBufferNum,
					 PWAVE_DATA_READY_CALLBACK_FUNCTION pDefWaveDataReadyCallBackFunction
					 )
{
long	i;

	m_DeviceID						= WAVE_MAPPER;
	m_DeviceBufferNum				= DefDeviceBufferNum;									//�豸¼���������ĸ���
	m_DeviceBlockBufferByteSize		= ((DefBitNum + 7 )/8)*DefBlockSampleLen*DefChannelNum;	//ÿ��¼�����������ֽڴ�С
	pWaveDataReadyCallBackFunction	= pDefWaveDataReadyCallBackFunction;					//¼�����ݻص�����

	hWaveInHdr		= new GLOBALHANDLE[m_DeviceBufferNum];
	lpInBuffer		= new LPBYTE[m_DeviceBufferNum];
	lpWaveInHdr		= new LPWAVEHDR[m_DeviceBufferNum];
	hInBuffer		= new GLOBALHANDLE[m_DeviceBufferNum];

	m_AudioBlockBuffer	= new char[m_DeviceBufferNum*m_DeviceBlockBufferByteSize]; 
	m_AudioBlockBufferLen  = new long[m_DeviceBufferNum];
	m_WritePtr		= 0;
	m_ReadPtr		= 0;
	m_AudioBlockNum	= 0;
	
	m_hAuidoReadyEvent = ::CreateEvent(
		NULL,
		FALSE,		// Auto Reset,
		FALSE,		// No Signal initially
		NULL
		);
	InitializeCriticalSection( &m_csNotifyAuidoBuffer);
	InitializeCriticalSection( &m_csSetCallback);
	InitializeCriticalSection( &m_csDevice );

	for( i = 0 ; i < m_DeviceBufferNum ; i++ )
	{
		//allocate WaveIn header
		hWaveInHdr[i]	= GlobalAlloc( GHND | GMEM_SHARE , sizeof(WAVEHDR) ) ;
		lpWaveInHdr[i]	= (LPWAVEHDR)GlobalLock(hWaveInHdr[i]) ;
		//allocate WaveIn audio buffer
		hInBuffer[i]	= GlobalAlloc(GHND| GMEM_SHARE , m_DeviceBlockBufferByteSize) ;
		lpInBuffer[i]	= (LPBYTE) GlobalLock(hInBuffer[i]) ;
		lpWaveInHdr[i]->lpData			= (char *)lpInBuffer[i];
		lpWaveInHdr[i]->dwBufferLength  = m_DeviceBlockBufferByteSize ;
	}

	SetWaveFormat(DefChannelNum,DefSampleRate,DefBitNum);

	m_bDeviceOpen = false;
	m_bRecording  = false;
	m_bWaveInThreadExit = true;

	return;
}

long	CRECORDER::GetWaveInDeviceNum(void)
{
	return(waveInGetNumDevs()); 
}
bool	CRECORDER::GetWaveInDeviceName(long DeviceID,CString &strDeviceName)
{
WAVEINCAPS	WaveInCaps;

	if( MMSYSERR_NOERROR == waveInGetDevCaps(DeviceID, &WaveInCaps,sizeof(WaveInCaps)) )
	{
		strDeviceName = WaveInCaps.szPname;
		return true;
	}
	else
		return false;
}


bool	CRECORDER::QuerySupportedFormat(WAVEFORMATEX *pWaveFormatEx,long WaveInDeviceID)
{
	if( WAVERR_BADFORMAT == waveInOpen(	NULL, 
				WaveInDeviceID, 
				pWaveFormatEx,
				(DWORD_PTR)NULL,
				(DWORD_PTR)NULL,
				WAVE_FORMAT_QUERY
				)	)
		return false;
	else
	    return true;
}


bool CRECORDER::Open(long WaveInID)
{
	if( m_bDeviceOpen == true ) return true;
	if( !QuerySupportedFormat( &m_WaveFormatEx ,WaveInID ) ) 
		return false;
	// ����¼����Ϣ�����߳�
	m_hWaveInThread = ::CreateThread(NULL,0,(LPTHREAD_START_ROUTINE)pWavInCallbackThread,this,0,&dwThreadId);

	if( MMSYSERR_NOERROR == waveInOpen(	&hWaveIn, 
				WaveInID, 
				&m_WaveFormatEx,
				(DWORD_PTR)dwThreadId,
				(DWORD_PTR)this,
				CALLBACK_THREAD
				)	)
	{
		m_bDeviceOpen = true;
	}

	if(m_bDeviceOpen == false) return false;
	m_WritePtr		= 0;
	m_ReadPtr		= 0;
	m_AudioBlockNum	= 0;
	::ResetEvent(m_hAuidoReadyEvent);
	m_hMonitorThread = ::CreateThread(NULL,0,(LPTHREAD_START_ROUTINE)pWavInMonitorThread,this,0,&dwMonitorThreadId);

	waveInGetID(hWaveIn,(LPUINT)(&m_DeviceID));		//��ȡ��ǰ¼���豸���豸��ʶ�� 

	for( long i = 0 ; i < m_DeviceBufferNum ; i++ )
	{
       	lpWaveInHdr[i]->dwBytesRecorded = 0L ;
		lpWaveInHdr[i]->dwUser          = i ;
		lpWaveInHdr[i]->dwFlags         = 0L ;
		lpWaveInHdr[i]->dwLoops         = 0L ;
		lpWaveInHdr[i]->lpNext          = NULL ;
		lpWaveInHdr[i]->reserved        = 0L ;
		waveInPrepareHeader(hWaveIn,lpWaveInHdr[i],sizeof(WAVEHDR));
		waveInAddBuffer(hWaveIn, lpWaveInHdr[i], sizeof(WAVEHDR));
	}
	return true;
}

// ��ʼ¼��
bool CRECORDER::Record(void)
{
	if( m_bDeviceOpen == false ) return false;

	::EnterCriticalSection(&m_csDevice);
	waveInStart(hWaveIn);			//����¼��
	m_bRecording = true;
	::LeaveCriticalSection(&m_csDevice);
	return true;
}

// ��ͣ¼��
bool CRECORDER::Stop(void)
{
	if( m_bDeviceOpen == false )	return false;
	::EnterCriticalSection(&m_csDevice);
	//waveInStop(hWaveIn);	// waveInStop����ֻ��ֹͣ¼��,�����᷵����δ�����Ļ�����,
	waveInReset(hWaveIn);	// ��waveInReset�����򲻵���ֹͣ¼������λ¼���豸 ,
							// ���¼���豸�������Ŷӵȴ���¼�������������һ��Ὣ������δ
							// �����Ļ�����ȫ������־ΪDONE,����Ӧ�ó���
	m_bRecording = false;
	::LeaveCriticalSection(&m_csDevice);

	// waveInReset�����еķ��صĻ�����������Ϣ����ʽ��֪ͨ¼����Ϣ�����̡߳�
	// ¼����Ϣ�����̣߳����̵߳���Ϣ���������δ�����Щ��Ϣ��������Ӧ�ó������ṩ�ĺ������������ݡ�
	// �����ж����е�¼�������Ƿ񶼱�Ӧ�ó�����մ������
	while ( IsCallbackUserApplicationComplete() == false )
	{	// ��ʱ����Ϣ�ص��̻߳��Ὣ¼�����������¼����豸
		// �ȴ�Ӧ�ó����������е��Ѿ��ɼ�������
		Sleep(100);
	}
	return true;
}

bool CRECORDER::Close(void)
{
long i;

	if( m_bDeviceOpen == false )	return false;

	::EnterCriticalSection( &m_csDevice );	
		m_bDeviceOpen = false;		// ������ʱm_bDeviceOpen = false,����֪ͨ��Ϣ�ص��߳�ϵͳ
									// ���ڹر�¼���豸,�����Ϣ�ص��̲߳��Ὣ¼�����������¼����豸
		waveInReset(hWaveIn);		// �ڹر�¼���豸֮ǰ,¼���豸�в��ܻ���¼��������
		for( i = 0 ; i < m_DeviceBufferNum ; i++ )
		{
			waveInUnprepareHeader(hWaveIn, lpWaveInHdr[i], sizeof(WAVEHDR));
			lpWaveInHdr[i]->dwFlags  = 0L; 
		}
		waveInClose(hWaveIn);	//�ر�¼���豸��¼���豸�رպ�¼����Ϣ�̻߳��Զ����١�
	::LeaveCriticalSection( &m_csDevice );

	while( m_bWaveInThreadExit == false )
	{
		Sleep(10);
	}

	::SetEvent(m_hAuidoReadyEvent);
	while( m_bMonitorThreadExit == false )
	{
		Sleep(10);
	}
	::CloseHandle(m_hWaveInThread);		// �ر��߳̾��
	::CloseHandle(m_hMonitorThread);
	return true;
}

CRECORDER::~CRECORDER()
{
long	i;

	Close();
	for( i = 0 ; i < m_DeviceBufferNum ; i++ )
	{
		GlobalUnlock(hWaveInHdr[i]);
		GlobalFree(hWaveInHdr[i]);
		GlobalUnlock(hInBuffer[i]);
		GlobalFree(hInBuffer[i]);
	}
	DeleteCriticalSection( &m_csDevice );
	DeleteCriticalSection( &m_csNotifyAuidoBuffer );
	DeleteCriticalSection( &m_csSetCallback);
	::CloseHandle(m_hAuidoReadyEvent);
	delete []hWaveInHdr;
	delete []lpInBuffer;
	delete []lpWaveInHdr;
	delete []hInBuffer;
	delete []m_AudioBlockBuffer;
	delete []m_AudioBlockBufferLen;
}


void CRECORDER::SetWaveDataReadyCallbackFunction(PWAVE_DATA_READY_CALLBACK_FUNCTION pCallBackFunction)
{
	::EnterCriticalSection(&m_csSetCallback);
	pWaveDataReadyCallBackFunction = pCallBackFunction;
	::LeaveCriticalSection(&m_csSetCallback);
}

// ����¼�����ݸ�ʽ
void	CRECORDER::SetWaveFormat(long WaveChannelNum,long WaveSampleRate,long WaveBitNum)
{
	m_WaveFormatEx.wFormatTag		= WAVE_FORMAT_PCM; 
    m_WaveFormatEx.nChannels		= (unsigned short)WaveChannelNum; 
    m_WaveFormatEx.nSamplesPerSec	= WaveSampleRate;	//sample rate in Hz
    m_WaveFormatEx.nAvgBytesPerSec	= (WaveChannelNum*WaveBitNum/8)*WaveSampleRate; 
    m_WaveFormatEx.nBlockAlign		= (WORD)(WaveChannelNum*WaveBitNum/8); 
    m_WaveFormatEx.wBitsPerSample	= (unsigned short)WaveBitNum; 
    m_WaveFormatEx.cbSize			= 0; 
}


bool	CRECORDER::IsCallbackUserApplicationComplete(void)
{
long	i;
bool	bComplete;

	::EnterCriticalSection(&m_csDevice);
	bComplete = true;
	for( i = 0 ; i < m_DeviceBufferNum ; i++ )
	{
		if( lpWaveInHdr[i]->dwFlags & WHDR_DONE )
		{
			bComplete = false;
			break;
		}
	}
	::LeaveCriticalSection(&m_csDevice);
	return bComplete;
}

//
//�߳�ģʽ 
//waveInOpen(&hWaveIn,WAVE_MAPPER,&waveform,m_ThreadID,NULL,CALLBACK_THREAD)��
//���ǿ��Լ̳�MFC��CwinThread�ֻ࣬Ҫ��Ӧ�Ĵ����߳���Ϣ���ɡ� 
//MFC�߳���Ϣ�ĺ�Ϊ�� 
//ON_THREAD_MESSAGE, 
//�������������Ϣӳ�䣺 ON_THREAD_MESSAGE(MM_WIM_CLOSE, OnMM_WIM_CLOSE) 
