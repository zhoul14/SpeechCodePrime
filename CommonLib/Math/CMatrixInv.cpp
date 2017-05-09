/*#include "stdafx.h"*/
#include <cmath>

void MatrixMul(double *a,double *b,double *c,long DIM)
{
long	i,j,k,ii,jj;
double	Sum;

	for(i=0;i<DIM;i++)
	for(j=0;j<DIM;j++)
	{
		Sum = 0;
		ii=i*DIM;
		jj=j;
		for(k=0;k<DIM;k++)
		{
			Sum += a[ii]*b[jj];
			ii++;
			jj+=DIM;
		}
		c[i*DIM+j]=Sum;
	}
}

//------------------------------------------------------------------------------
// MatrixInv ��ȫѡ��Ԫ��ַͬ�����������
// ���룺
//		a:	 ָ�����Ļ�������ַ������Ԫ���ǰ���˳��洢��
//		DIM: DIM X DIM �����ά��
// ���أ�
//		a:	ָ�������Ļ�������ַ
//		ret:�����ķ���ֵ���������a������ʽֵ
//	�㷨��	 Gauss-Jordan���������㷨�����㷨���ó����б任�ķ���������Ԫ��
//			����Ԫ�Ĺ����в���ȫѡ��Ԫ�ķ�����ȫѡ��Ԫ�󽫽�����Ԫ���Խ�����
//			����֤�����ֽ������ڶԾ�������֮ǰ���н������������ǵȼ۵ġ����൱��
//			��(Q*A*R)���档����Q���б任����R���б任���������INV(Q*A*R)=B
//			��INV(A)=R*B*Q����ȫѡ��Ԫʱ���н����ڻָ�ʱ�ͱ�����н�����
//------------------------------------------------------------------------------

double MatrixInv(double *a,long DIM)
{   
long	*is,*js,i,j,k,l,u,v,x,y;
double	d,fTmp,Det;

    is=new long[DIM];
    js=new long[DIM];

	Det = 1;

    for (k=0; k<DIM; k++)
    {
		//ȫѡ��Ԫ
		d=0.0;
		for( i=k,x=k*DIM; i<DIM; i++,x+=DIM )
		for( j=k,l=x+k; j<DIM; j++,l++ )
		{
			fTmp=( a[l] > 0 )?  a[l] : -a[l];
			if (fTmp>d)
			{ d=fTmp; is[k]=i; js[k]=j;}
		}

		if ( fabs(d) < 1e-50 )
		{	//��ԪΪ0�����󲻿���
			delete []is;
			delete []js;
            return(0);
		}
		//����Ԫ�������Խ�����
		if ( is[k] != k )
		{	//��������ʽ������ʽֵ���
			Det=-Det;
			for ( j=0,u=k*DIM,v=is[k]*DIM ; j<DIM ; j++,u++,v++ )
			{	//��k�к͵�is[k]�н�������a[k][j] <=> a[ is[k] ][j]
				fTmp=a[u];
				a[u]=a[v];
				a[v]=fTmp;
			}
		}
		if ( js[k] != k )
		{	//��������ʽ������ʽֵ���
			Det=-Det;
			for (i=0,u=k,v=js[k]; i<DIM; i++,u+=DIM,v+=DIM)
			{	//��k�к�js[k]�н���,��a[i][k] <=> a[i][ js[k] ]
				fTmp=a[u]; 
				a[u]=a[v];
				a[v]=fTmp;
			}
		}

        l=k*DIM+k;
		Det = Det * a[l];		//��������ʽ��ֵ
        a[l]=1.0/a[l];
		for( j=0,u=k*DIM; j<DIM; j++,u++ )	//�Ե�k�н�����Ԫ��һ��
        if( j != k )
	    { 
			a[u] *= a[l]; 
		}
	    //���жԾ��������Ԫ
		v=k*DIM;	//�����k�еĵ�ַ
		for( i=0,y=0; i<DIM; i++,y+=DIM )
		{
			if ( i != k )
			{	//�Ե�i����Ԫ
				fTmp=a[y+k];			//fTmp=a[i][k];
				a[y+k]=0;
				for( j=0,u=y,x=v; j<DIM; j++,u++,x++ )
				{
					a[u] -= fTmp*a[x];	//a[i][j]-=fTmp*a[k][j]
				}
			}
		}
    }
	//�ָ�ԭʼ������󣬼�: R*B*Q,����R���б任����Q���б任����
	//������Ԫʱ���н����ڻָ�ʱ�Ǳ�����н���
	for( k=DIM-1; k>=0; k-- )
    {   
		if ( js[k] != k )
		{
			for( j=0,u=k*DIM,v=js[k]*DIM; j<DIM; j++,u++,v++ )
			{	//��k�к�js[k]�н���
				//a[k][j] <=> a[ js[k] ][j];
				fTmp=a[u];	a[u]=a[v];	a[v]=fTmp;
			}
		}

		if ( is[k] != k )
		for( i=0,u=k,v=is[k]; i<DIM; i++,u+=DIM,v+=DIM )
		{	//��k�к�is[k]�н���
			//a[i][k] <=> a[i][ is[k] ]
			fTmp=a[u];	a[u]=a[v];	a[v]=fTmp;
		}
    }
	delete []is;
	delete []js;
    return(Det);
}

float MatrixInv(float *a,long DIM)
{   
long	*is,*js,i,j,k,l,u,v,x,y;
float	d,fTmp,Det;

    is=new long[DIM];
    js=new long[DIM];

	Det = 1;

    for (k=0; k<DIM; k++)
    {
		//ȫѡ��Ԫ
		d=0.0;
		for( i=k,x=k*DIM; i<DIM; i++,x+=DIM )
		for( j=k,l=x+k; j<DIM; j++,l++ )
		{
			fTmp=( a[l] > 0 )?  a[l] : -a[l];
			if (fTmp>d)
			{ d=fTmp; is[k]=i; js[k]=j;}
		}

		if ( d == 0.0 )
		{	//��ԪΪ0�����󲻿���
			delete []is;
			delete []js;
            return(0);
		}
		//����Ԫ�������Խ�����
		if ( is[k] != k )
		{	//��������ʽ������ʽֵ���
			Det=-Det;
			for ( j=0,u=k*DIM,v=is[k]*DIM ; j<DIM ; j++,u++,v++ )
			{	//��k�к͵�is[k]�н�������a[k][j] <=> a[ is[k] ][j]
				fTmp=a[u];
				a[u]=a[v];
				a[v]=fTmp;
			}
		}
		if ( js[k] != k )
		{	//��������ʽ������ʽֵ���
			Det=-Det;
			for (i=0,u=k,v=js[k]; i<DIM; i++,u+=DIM,v+=DIM)
			{	//��k�к�js[k]�н���,��a[i][k] <=> a[i][ js[k] ]
				fTmp=a[u]; 
				a[u]=a[v];
				a[v]=fTmp;
			}
		}

        l=k*DIM+k;
		Det = Det * a[l];		//��������ʽ��ֵ
        a[l]=1.0f/a[l];
		for( j=0,u=k*DIM; j<DIM; j++,u++ )	//�Ե�k�н�����Ԫ��һ��
        if( j != k )
	    { 
			a[u] *= a[l]; 
		}
	    //���жԾ��������Ԫ
		v=k*DIM;	//�����k�еĵ�ַ
		for( i=0,y=0; i<DIM; i++,y+=DIM )
		{
			if ( i != k )
			{	//�Ե�i����Ԫ
				fTmp=a[y+k];			//fTmp=a[i][k];
				a[y+k]=0;
				for( j=0,u=y,x=v; j<DIM; j++,u++,x++ )
				{
					a[u] -= fTmp*a[x];	//a[i][j]-=fTmp*a[k][j]
				}
			}
		}
    }
	//�ָ�ԭʼ������󣬼�: R*B*Q,����R���б任����Q���б任����
	//������Ԫʱ���н����ڻָ�ʱ�Ǳ�����н���
	for( k=DIM-1; k>=0; k-- )
    {   
		if ( js[k] != k )
		{
			for( j=0,u=k*DIM,v=js[k]*DIM; j<DIM; j++,u++,v++ )
			{	//��k�к�js[k]�н���
				//a[k][j] <=> a[ js[k] ][j];
				fTmp=a[u];	a[u]=a[v];	a[v]=fTmp;
			}
		}

		if ( is[k] != k )
		for( i=0,u=k,v=is[k]; i<DIM; i++,u+=DIM,v+=DIM )
		{	//��k�к�is[k]�н���
			//a[i][k] <=> a[i][ is[k] ]
			fTmp=a[u];	a[u]=a[v];	a[v]=fTmp;
		}
    }
	delete []is;
	delete []js;
    return(Det);
}
