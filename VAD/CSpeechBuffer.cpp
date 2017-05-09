#include <stdio.h>
#include <windows.h>
//#include "CFIRfilter.h"
#include "CSpeechBuffer.h"

//DefRawSpeechBufferLen��Ԥ������ݻ���������
//DefFilterSpeechBufferLen��Ԥ����˲���������ݻ���������
//CSPEECH_BUFFER���ÿ�����뵽�����������е����ݶ���Ҫ����FIRԤ�˲�����Ȼ���ٽ��з�������
//���ÿ�����뻺���������ݳ��Ȳ��ܳ���DefFilterSpeechBufferLen����Ļ���������

CSPEECH_BUFFER::CSPEECH_BUFFER()
{
}

CSPEECH_BUFFER::~CSPEECH_BUFFER()
{
//	delete pCBPSoundPreFilter;

//	DeleteCriticalSection(&BufferCriticalSection);
//	CloseHandle(hBufferReady);
	return;
}


void	CSPEECH_BUFFER::IniClass()
{
	bRecorderCloseFlag		= false;
	RawSpeechBufferLen		= BUFFER_LEN;
	FilterSpeechBufferLen	= BUFFER_LEN;

//	InitializeCriticalSection(&BufferCriticalSection);

	//���õ绰��ͨ�˲�����8KHz�����ʣ�300Hz--3300Hz����301��FIR�˲���
//	pCBPSoundPreFilter= new CFIRFilter(100,3300,16000,301);
//	hBufferReady=CreateEvent(NULL,	//lpEventAttributes
//				TRUE,				//bManualReset 
//				FALSE,				//bInitialState
//				NULL);				//lpName   

	WritePtr	= 0;
	ReadPtr		= 0;
	Mask		= RawSpeechBufferLen-1;

	return;

}



bool CSPEECH_BUFFER::WriteSpeechToBuffer(short *Buffer,long DataNum)
{
long	i,EmptyBufferLen;
bool	flag;

	if( DataNum > FilterSpeechBufferLen )	// FilterSpeechBufferLen Ϊ������Ԥ�����õ��˲������
		return(false);						// �Ļ�����pTmpFilterBuffer�ĳ��ȡ�
											// ԭʼ���������Ҫ����Ԥ�˲�����
	
//	EnterCriticalSection(&BufferCriticalSection);
	
	//�ô�ͨ�˲����˲�
//	pCBPSoundPreFilter->DoFirFilter((short *)Buffer,pTmpFilterBuffer,DataNum);
	memcpy(m_TmpFilterBuffer,Buffer,sizeof(short)*DataNum);	//pTmpFilterBuffer�д�ŵ���Ԥ�˲�������

	//�������Ļ���������
	if( WritePtr >= ReadPtr )											//WritePtrָ���һ���հ׵Ļ�����
		EmptyBufferLen = RawSpeechBufferLen-1 - ( WritePtr - ReadPtr );	//ReadPtrָ���һ�����õ����ݻ�����
	else
		EmptyBufferLen = ReadPtr - WritePtr-1;

	if( EmptyBufferLen >= DataNum )
	{
		for(i=0;i<DataNum;i++)
		{
			m_RawSpeechBuffer[WritePtr++]=m_TmpFilterBuffer[i];
			WritePtr=WritePtr&Mask;
		}
		flag = true;
	}
	else
	{
		for( i = 0;i< EmptyBufferLen; i++ )
		{
			m_RawSpeechBuffer[WritePtr++] = m_TmpFilterBuffer[i];
			WritePtr=WritePtr&Mask;
		}
		flag = false;
	}

//	SetEvent(hBufferReady);
//	LeaveCriticalSection(&BufferCriticalSection);
	return(flag);
}
/*
long CSPEECH_BUFFER::ReadSpeechFromBuffer(short *Buffer,long DataNum)
{
long	DataReadyNum,i;

	EnterCriticalSection(&BufferCriticalSection);
	
	if( WritePtr == ReadPtr )
	{
		LeaveCriticalSection(&BufferCriticalSection);
		return(0);
	}
	if( WritePtr > ReadPtr )
	{
		DataReadyNum = WritePtr- ReadPtr;
	}
	else
	{
		DataReadyNum=RawSpeechBufferLen-ReadPtr+WritePtr;
	}
	if( DataReadyNum < DataNum )
	{
		LeaveCriticalSection(&BufferCriticalSection);
		return(0);
	}
	
	for(i=0;i<DataNum;i++)
	{
		Buffer[i]=pRawSpeechBuffer[ReadPtr++];
		ReadPtr=ReadPtr&Mask;
	}
	
	LeaveCriticalSection(&BufferCriticalSection);
	
	return(DataReadyNum);
}
*/

//�ȴ����DataNum����������
long CSPEECH_BUFFER::WaitForSpeech(short *Buffer,long DataNum)
{
bool	bReadyFlag;
long	DataReadyNum,i;

	bReadyFlag = false;
	while( bReadyFlag == false )
	{
		bReadyFlag = true;	
//		EnterCriticalSection(&BufferCriticalSection);
		if( WritePtr == ReadPtr )
			bReadyFlag = false;
		else
		{
			if( WritePtr > ReadPtr )
				DataReadyNum = WritePtr- ReadPtr;
			else
				DataReadyNum=RawSpeechBufferLen-ReadPtr+WritePtr;

			if( DataReadyNum < DataNum )
				bReadyFlag = false;		//û���㹻�������
		}

		if( bReadyFlag == true )
		{
			for(i=0;i<DataNum;i++)
			{
				Buffer[i]=m_RawSpeechBuffer[ReadPtr++];		//������������
				ReadPtr=ReadPtr&Mask;
			}
		}
//		LeaveCriticalSection(&BufferCriticalSection);

		if( bReadyFlag == true )
			return(DataReadyNum);
		else
		{
//			ResetEvent(hBufferReady);
//			if( WaitForSingleObject(hBufferReady,1000)==WAIT_TIMEOUT )
			{
				if( bRecorderCloseFlag == true )
					return(-2);
				else
					return(-1);
			}
		}
	}
	return(-1);
}

void	CSPEECH_BUFFER::ClearSpeechBuffer(void)
{
//	EnterCriticalSection(&BufferCriticalSection);
	WritePtr	= 0;
	ReadPtr		= 0;
//	LeaveCriticalSection(&BufferCriticalSection);
}