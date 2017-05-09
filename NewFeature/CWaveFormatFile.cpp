/******************************************
*    2006��12��08�ո���
*******************************************/
// 2007��01��01��(Ԫ��)���壬xiaoxi 
// 2009,02,28  �����ļ��ֽڴ�С����
// 2009,08,21 �����ݿƼ����޸ĳ�֧��Unicode�ַ�����ʽ
//#include "stdafx.h"
#include <TCHAR.h>
#include "CWaveFormatFile.h"
#include "io.h"

long CWaveFile::tagRIFF_FOURCC = 0x46464952;
long CWaveFile::tagWAVE_FOURCC = 0x45564157;
long CWaveFile::tagfmt_FOURCC  = 0x20746D66;
long CWaveFile::tagdata_FOURCC = 0x61746164;

CWaveFile::CWaveFile()
{
	// Wave��ʽ����������
	memset( &m_WaveFormatEx, 0, sizeof(m_WaveFormatEx) );
	InitializeCriticalSection( &WaveFileCriticalSection );
	m_bIsFileOpen = false;
	m_FileByteSize	= 0;

	for( long i = 0 ; i < 255; i++ )
	{
		m_Table_8bit_LinearPCM_2_16bitPCM[i] = (short)( (i - 0x80)<<8 );
	}

}

CWaveFile::~CWaveFile()
{
	Close();
	DeleteCriticalSection( &WaveFileCriticalSection );
}

void	CWaveFile::EnterCriticalSection(void)
{
	::EnterCriticalSection( &WaveFileCriticalSection );
}
void	CWaveFile::LeaveCriticalSection(void)
{
	::LeaveCriticalSection( &WaveFileCriticalSection );
}


/*************************************
SearchChunk����ͼ�ǣ�
20051031
*************************************/
long CWaveFile::SearchChunk( FILE *fp, long Offset, long Length, long FOURCC_Chunk )
{
long	*pFOURCC,i,ReadLen;
char	ChunkBuffer[2000];


	fseek( fp , Offset , SEEK_SET );
	ReadLen = (long)fread( ChunkBuffer, sizeof(char), 2000, fp );
	for( i = 0 ; i < ReadLen-4; i++ )
	{
		pFOURCC = (long * ) ( &ChunkBuffer[i] );
		if( pFOURCC[0] == FOURCC_Chunk )
		{
			return( i + Offset );
		}
	}
	return(-1);
}

long CWaveFile::UpdateWaveFileHeader(long ByteSize)
{
long	OffsetAfterDataChunk;

	dataChunk.data_FOURCC   = tagdata_FOURCC;
	dataChunk.dataChunkSize = ByteSize;

	fmtChunk.fmt_FOURCC		= tagfmt_FOURCC;
	fmtChunk.fmtChunkSize	= sizeof(WAVEFORMATEX);

	WaveFileHeader.RIFF_FOURCC  = tagRIFF_FOURCC;
	WaveFileHeader.WaveFileSize = sizeof(tagWAVE_FOURCC)  +
								  sizeof(fmtChunk)        + fmtChunk.fmtChunkSize   +
								  sizeof( dataChunk )     + dataChunk.dataChunkSize;
	// ����"data"���ݿ�֮��ĵ�һ���ֽ���Wave�ļ��е���ʼƫ����λ��			
	OffsetAfterDataChunk	= sizeof(WaveFileHeader) + WaveFileHeader.WaveFileSize;
	return(OffsetAfterDataChunk);
}


bool CWaveFile::Open(const TCHAR *WaveFileName,long DefReadWriteMode )
{
	if( m_bIsFileOpen == true ) Close();
	ReadWriteMode	=  DefReadWriteMode;
	m_bIsFileOpen	= false;

	switch(ReadWriteMode)
	{
		case modeReadWrite:			//READ_WRITE_MODE:
		{	//����д�ļ�
			if( _tfopen_s(&fp, WaveFileName, _T("w+b")) != 0 ) return false;
			if( fp == NULL ) 
				return(false);
			else
			{
				m_bIsFileOpen	= true;
				// ���ó�ʼ�ġ�WAVE�����ݿ���ֽڴ�СΪ0��
				// Ȼ���趨��data�����ݿ���Wave��ʽ�ļ��е�ƫ����λ��
				MaxDataByteSize			= 0;		
				DataBufferFileOffset	= UpdateWaveFileHeader(MaxDataByteSize);	
				ReadDataBufferPtr		= 0;
				WriteDataBufferPtr		= 0;
				m_WaveFormatEx.wFormatTag = WAVE_FORMAT_PCM;
				m_FileByteSize			= 0;
				return(true);
			}
			break;
		}
		case modeRead:				//READ_ONLY_MODE:
		{
			if( _tfopen_s(&fp,WaveFileName,_T("rb")) != 0 ) return false;
			if(fp == NULL ) return(false);
			m_FileByteSize = ::_filelength(::_fileno(fp));

			fseek(fp,0,SEEK_SET);
			// ����WAVE��ʽ�ļ�ͷ
			fread( &WaveFileHeader , sizeof(WAVE_FILE_HEADER) , 1 , fp );
			if( WaveFileHeader.RIFF_FOURCC != tagRIFF_FOURCC )
			{
				fclose(fp);
				return(false);
			}
		
			//printf("FileSize = %ld\n",WaveFileHeader.WaveFileSize + sizeof(WaveFileHeader) );
			// �ж��Ƿ�Ϊ��WAVE����ʽ�ļ�
			if( SearchChunk(fp, sizeof(WAVE_FILE_HEADER), WaveFileHeader.WaveFileSize,tagWAVE_FOURCC)== -1 )
			{
				fclose(fp);
				return(false);
			}
			// ������ʽ��������־λ��
			fmtChunkOffset = SearchChunk(fp,sizeof(WAVE_FILE_HEADER)+sizeof(tagWAVE_FOURCC),WaveFileHeader.WaveFileSize,tagfmt_FOURCC);
			if( fmtChunkOffset == -1 ) return(false);

			fseek(fp,fmtChunkOffset,SEEK_SET);
			fread(&fmtChunk,sizeof(FMT_CHUNK),1,fp);

			if( fmtChunk.fmtChunkSize > sizeof(WAVEFORMATEX) )
				fread(&m_WaveFormatEx, sizeof(WAVEFORMATEX),1,fp);
			else
				fread(&m_WaveFormatEx,fmtChunk.fmtChunkSize,1,fp);

			if( m_WaveFormatEx.wFormatTag != WAVE_FORMAT_PCM )
			{
				fclose(fp); 
				return(false); 
			}
			// �������ݿ�������־λ��
			dataChunkOffset = fmtChunkOffset +sizeof( fmtChunk) + fmtChunk.fmtChunkSize;
			dataChunkOffset = SearchChunk(fp, dataChunkOffset, WaveFileHeader.WaveFileSize-dataChunkOffset,tagdata_FOURCC);
			if( dataChunkOffset == -1 ) return(false);

			fseek(fp,dataChunkOffset,SEEK_SET);
			fread(&dataChunk,sizeof(DATA_CHUNK),1,fp);
			m_bIsFileOpen = true;
			// ����"data"���ݿ�����ļ��е���ʼλ�úʹ�С
			DataBufferFileOffset = dataChunkOffset + sizeof( dataChunk );
			MaxDataByteSize		 = dataChunk.dataChunkSize;
			ReadDataBufferPtr	 = 0;
			WriteDataBufferPtr	 = 0;
			return(true);
			break;
		}
		default:
			m_bIsFileOpen = false;
	}
	return(false);
}

void	 CWaveFile::SetWaveFormat(WAVEFORMATEX *pWaveFormatEx)
{
	m_WaveFormatEx = *pWaveFormatEx; 
}

long CWaveFile::Write(char *DataBuffer,long ByteSize)
{
long	BytesWritten;
	//�ƶ��ļ�ָ��,д������
	fseek( fp, DataBufferFileOffset + WriteDataBufferPtr, SEEK_SET ); 
	BytesWritten =(long) fwrite( DataBuffer, sizeof(char), ByteSize, fp );
	WriteDataBufferPtr += BytesWritten;
	if( WriteDataBufferPtr  > MaxDataByteSize )
	{
		MaxDataByteSize = WriteDataBufferPtr;
	}
	m_FileByteSize = ftell( fp );
	return(BytesWritten);
}

long CWaveFile::Write(short *DataBuffer,long ByteSize)		//2006,12,18����
{
long	BytesWritten;
	//�ƶ��ļ�ָ��,д������
	fseek( fp, DataBufferFileOffset + WriteDataBufferPtr, SEEK_SET ); 
	BytesWritten =(long) fwrite( DataBuffer, sizeof(char), ByteSize, fp );
	WriteDataBufferPtr += BytesWritten;
	if( WriteDataBufferPtr  > MaxDataByteSize )
	{
		MaxDataByteSize = WriteDataBufferPtr;
	}
	m_FileByteSize = ftell( fp );
	return(BytesWritten);
}


long CWaveFile::Read(char *Buffer,long ByteSize)
{
long	ByteRead;

	// ����ɹ���ȡ�������ֽ���
	ByteRead = MaxDataByteSize - ReadDataBufferPtr;
	if( ByteRead > ByteSize )
		ByteRead = ByteSize;

	fseek( fp, DataBufferFileOffset+ReadDataBufferPtr, SEEK_SET );
	ByteRead =(long) fread(Buffer,sizeof(char),ByteRead,fp);

	ReadDataBufferPtr += ByteRead;	//�ƶ�������ָ��
	return( ByteRead );
}

long CWaveFile::Read(short *Buffer,long ByteSize)			//2006,12,18����
{
long	ByteRead;

	// ����ɹ���ȡ�������ֽ���
	ByteRead = MaxDataByteSize - ReadDataBufferPtr;
	if( ByteRead > ByteSize )
		ByteRead = ByteSize;

	fseek( fp, DataBufferFileOffset+ReadDataBufferPtr, SEEK_SET );
	ByteRead =(long) fread(Buffer,sizeof(char),ByteRead,fp);

	ReadDataBufferPtr += ByteRead;	//�ƶ�������ָ��
	return( ByteRead );
}


long CWaveFile::Seek(long Offset)
{
	ReadDataBufferPtr  =  Offset;
	WriteDataBufferPtr =  Offset;

	if( ReadWriteMode == modeRead )
	{	//�ڶ�ģʽ�����ݻ�����ָ�벻�ܳ������ݵĴ�С
		if( Offset > MaxDataByteSize )
			ReadDataBufferPtr = MaxDataByteSize;
		return( ReadDataBufferPtr );
	}
	else
	{
		return( WriteDataBufferPtr );
	}
}

long CWaveFile::SeekReadPtr(long Offset)
{
	ReadDataBufferPtr  =  Offset;
	if( ReadWriteMode == modeRead )
	{	//�ڶ�ģʽ�����ݻ�����ָ�벻�ܳ������ݵĴ�С
		if( Offset > MaxDataByteSize )
			ReadDataBufferPtr = MaxDataByteSize;
	}
	return( ReadDataBufferPtr );
}

long CWaveFile::SeekWritePtr(long Offset)
{
	WriteDataBufferPtr =  Offset;
	return( WriteDataBufferPtr );
}

void CWaveFile::Close(void)
{
	if( m_bIsFileOpen == true )	
	{
		if( ReadWriteMode == modeReadWrite )
		{
			UpdateWaveFileHeader( MaxDataByteSize );					// ����Wave�ļ�ͷ����
			fseek(  fp, 0L, SEEK_SET );
			fwrite( &WaveFileHeader, sizeof(WaveFileHeader), 1, fp );	// д��RIFF�ļ�ͷ
			fwrite( &tagWAVE_FOURCC, sizeof(tagWAVE_FOURCC), 1, fp );	// д��WAVE��ʽ���
			fwrite( &fmtChunk, sizeof(fmtChunk), 1, fp );				// д��fmt�ṹ����
			fwrite( &m_WaveFormatEx, sizeof(char), fmtChunk.fmtChunkSize, fp );
			fwrite( &dataChunk, sizeof(dataChunk), 1, fp );				// д��data�ṹ����
		}
		fclose(fp);
		m_bIsFileOpen = false;
	}
		
}


void CWaveFile::SetWaveFormat(long ChannelNum, long SampleRate,long BitsPerSample)
{
	m_WaveFormatEx.wFormatTag		= WAVE_FORMAT_PCM;
	m_WaveFormatEx.nChannels		= (WORD)ChannelNum;
	m_WaveFormatEx.nSamplesPerSec	= SampleRate;
    m_WaveFormatEx.nAvgBytesPerSec	= ChannelNum*SampleRate*BitsPerSample/8; 
    m_WaveFormatEx.nBlockAlign		= (WORD)(ChannelNum*BitsPerSample/8);     
	m_WaveFormatEx.wBitsPerSample	= (WORD)BitsPerSample;
	m_WaveFormatEx.cbSize			= 0;
}

long	CWaveFile::GetChannelNum(void)
{
	return(m_WaveFormatEx.nChannels);
}

long CWaveFile::GetSampleRate(void)
{
	return( m_WaveFormatEx.nSamplesPerSec );
}

long	CWaveFile::GetBitsPerSample(void)
{
	return(m_WaveFormatEx.wBitsPerSample);
}

long	CWaveFile::GetBlockAlign(void)
{
	return(m_WaveFormatEx.nBlockAlign);
}

long	CWaveFile::GetDataFileOffset(void)
{
	return(DataBufferFileOffset);
}

long	CWaveFile::GetDataByteSize(void)
{
	return(MaxDataByteSize);
}

//===================================================================
//����������Ƶ�ĳ��ȣ������������
long	CWaveFile::GetStereoAudio( short (*AudioBuf)[2] ,long DataNum)
{
	if( m_bIsFileOpen == false )
	{
		for( long i = 0 ;i < DataNum; i++ )
		{
			AudioBuf[i][0] = 0;
			AudioBuf[i][1] = 0;
		}
		return 0;
	}


	switch( m_WaveFormatEx.wFormatTag )
	{
	case WAVE_FORMAT_PCM:
		if( m_WaveFormatEx.nChannels == 2 )
		{
			if( m_WaveFormatEx.wBitsPerSample == 16 )
			{	//������16��������PCM��Ƶ����
				long ByteRead = this->Read((char *)AudioBuf,DataNum*sizeof(short)*2);
				long DataNumRead = ByteRead/(sizeof(short)*2);
				for(long i = DataNumRead; i< DataNum; i++ )
				{
					AudioBuf[i][0] = 0;
					AudioBuf[i][1] = 0;
				}
				return DataNumRead;		
			}
			else if( m_WaveFormatEx.wBitsPerSample == 8 )
			{
				//������8��������PCM��Ƶ����
				long ByteRead = this->Read((char *)AudioBuf,DataNum*sizeof(char)*2);
				long DataNumRead = ByteRead/(sizeof(char)*2);
				for(long i = DataNumRead; i< DataNum; i++ )
				{
					AudioBuf[i][0] = 0;
					AudioBuf[i][1] = 0;
				}

				unsigned char (*p8bitPCM)[2] = (unsigned char (*)[2])AudioBuf;
				short L_16bitPCM,R_16bitPCM;  
				for( long i = DataNumRead-1; i >= 0; i-- )
				{	//8��������PCMת����16��������PCM
					L_16bitPCM = m_Table_8bit_LinearPCM_2_16bitPCM[ p8bitPCM[i][0] ];
					R_16bitPCM = m_Table_8bit_LinearPCM_2_16bitPCM[ p8bitPCM[i][1] ];
					AudioBuf[i][0] = L_16bitPCM;
					AudioBuf[i][1] = R_16bitPCM;
				}
				return DataNumRead;		
			}
			else
			{	//��֧��24bit�أ�32���ص�����PCM
				for( long i = 0 ;i < DataNum; i++ )
				{
					AudioBuf[i][0] = 0;
					AudioBuf[i][1] = 0;
				}
				return 0;
			}
		}
		else
		{
			if( m_WaveFormatEx.wBitsPerSample == 16 )
			{	//������16��������PCM��Ƶ����
				long ByteRead = this->Read((char *)AudioBuf,DataNum*sizeof(short));
				long DataNumRead = ByteRead/sizeof(short);
				for(long i = DataNumRead; i< DataNum; i++ )
				{
					AudioBuf[i][0] = 0;
					AudioBuf[i][1] = 0;
				}

				short *pMonoAudio = (short *)AudioBuf;
				for( long i = DataNumRead-1; i >= 0 ; i-- )
				{
					AudioBuf[i][0] = pMonoAudio[i];
					AudioBuf[i][1] = pMonoAudio[i];
				}

				return DataNumRead;		
			}
			else if( m_WaveFormatEx.wBitsPerSample == 8 )
			{
				//������8��������PCM��Ƶ����
				long ByteRead = this->Read((char *)AudioBuf,DataNum*sizeof(char));
				long DataNumRead = ByteRead/(sizeof(char));
				for(long i = DataNumRead; i< DataNum; i++ )
				{
					AudioBuf[i][0] = 0;
					AudioBuf[i][1] = 0;
				}

				unsigned char *pMono8bitPCM = (unsigned char *)AudioBuf;
				short Mono16bitPCM;  
				for( long i = DataNumRead-1; i >= 0; i-- )
				{	//8��������PCMת����16��������PCM
					Mono16bitPCM = m_Table_8bit_LinearPCM_2_16bitPCM[ pMono8bitPCM[i] ];
					AudioBuf[i][0] = Mono16bitPCM;
					AudioBuf[i][1] = Mono16bitPCM;
				}
				return DataNumRead;		
			}
			else
			{	//��֧��24bit�أ�32���ص�����PCM
				for( long i = 0 ;i < DataNum; i++ )
				{
					AudioBuf[i][0] = 0;
					AudioBuf[i][1] = 0;
				}
				return 0;
			}

		}
		break;
	case WAVE_FORMAT_ALAW:
		break;
	case WAVE_FORMAT_MULAW:
		break;
	}

	for( long i = 0 ;i < DataNum; i++ )
	{
			AudioBuf[i][0] = 0;
			AudioBuf[i][1] = 0;
	}
	return 0;
}