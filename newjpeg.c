#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "ppm.h"


#define PI 3.14159265358979
#define E 2.7182818284
#define SIGMA 2
#define MAX_SOURCE_SIZE (0x100000)



cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue queue = NULL;
cl_program program = NULL;

/* Set Local and Global Worksize */
int setWorkSize(size_t* gws, size_t* lws, cl_int x, cl_int y)
{
switch(y) {
case 1:
gws[0] = x;
gws[1] = 1;
lws[0] = 1;
lws[1] = 1;
break;
default:
gws[0] = x;
gws[1] = y;
lws[0] = 8;
lws[1] = 16;
 
break;}

return 0;
}

/* JPEG Markers */
typedef enum {
      M_SOF0 = 0xc0,
      M_SOF1 = 0xc1,
      M_SOF2 = 0xc2,
      M_SOF3 = 0xc3,

      M_SOF5 = 0xc5,
      M_SOF6 = 0xc6,
      M_SOF7 = 0xc7,

      M_JPG = 0xc8,
      M_SOF9 = 0xc9,
      M_SOF10 = 0xca,
      M_SOF11 = 0xcb,

      M_SOF13 = 0xcd,
      M_SOF14 = 0xce,
      M_SOF15 = 0xcf,

      M_DHT = 0xc4,

      M_DAC = 0xcc,

      M_RST0 = 0xd0,
      M_RST1 = 0xd1,
      M_RST2 = 0xd2,
      M_RST3 = 0xd3,
      M_RST4 = 0xd4,
      M_RST5 = 0xd5,
      M_RST6 = 0xd6,
      M_RST7 = 0xd7,

      M_SOI = 0xd8,
      M_EOI = 0xd9,
      M_SOS = 0xda,
      M_DQT = 0xdb,
      M_DNL = 0xdc,
      M_DRI = 0xdd,
      M_DHP = 0xde,
      M_EXP = 0xdf,

      M_APP0 = 0xe0,
      M_APP15 = 0xef,

      M_JPG0 = 0xf0,
      M_JPG13 = 0xfd,
      M_COM = 0xfe,

      M_TEM = 0x01,

      M_ERROR = 0x100
} JpegMarker;

/* buf is used to extract MSB */
unsigned char buf[2]={7,0};

unsigned const char quant1[64]={16,11,12,14,12,10,16,14,13,14,18,17,16,19,24,40,26,24,22,22,24,49,35,37,29,40,58,51,61,60,57,51,56,55,64,72,92,78,64,68,87,69,55,56,80,109,81,87,95,98,103,104,103,62,77,113,121,112,100,120,92,101,103,99};
/*
float quant11[64]={16,11,10,16,24,40,51,61,12,12,14,19,26,58,60,55,14,13,16,24,40,57,69,56,14,17,22,29,51,87,80,62,18,22,37,56,68,109,103,77,24,35,55,64,81,104,113,92,49,64,78,87,103,121,120,101,72,92,95,98,112,100,103,99};

unsigned const char quant111[64]={16,11,10,16,24,40,51,61,12,12,14,19,26,58,60,55,14,13,16,24,40,57,69,56,14,17,22,29,51,87,80,62,18,22,37,56,68,109,103,77,24,35,55,64,81,104,113,92,49,64,78,87,103,121,120,101,72,92,95,98,112,100,103,99};
*/
unsigned const char quant2[64]={17,18,18,24,21,24,47,26,26,47,99,66,56,66,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99};
/*
float quant22[64]={17,18,24,47,99,99,99,99,18,21,26,66,99,99,99,99,24,26,56,99,99,99,99,99,47,66,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99};

unsigned const char quant222[64]={17,18,24,47,99,99,99,99,18,21,26,66,99,99,99,99,24,26,56,99,99,99,99,99,47,66,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99};
*/
/* DC Luminance */
unsigned const char dcluc[16]={0,0,7,1,1,1,1,1,0,0,0,0,0,0,0,0};	//dcluc[i] --> count of number of codes that require i bits
unsigned const char dclus[12]={4,5,3,2,6,1,0,7,8,9,0xa,0xb};		//Huffman codes 
/* DC Chrominance */
unsigned const char dccrc[16]={0,2,2,3,1,1,1,1,1,0,0,0,0,0,0,0}; 
unsigned const char dccrs[12]={1,0,2,3,4,5,6,7,8,9,0xa,0xb};
/* AC Chrominance */
unsigned const char accrc[16]={0,2,2,1,2,3,5,5,4,5,6,4,8,3,3,0x6d};
/* AC Luminance */
unsigned const char acluc[16]={0,2,1,3,3,2,4,2,6,7,3,4,2,6,2,0x73}; 
unsigned const char aclus[162]={0x1,0x2,0x3,0x11,0x4,0x0,0x5,0x21,0x12,0x31,0x41,0x51,0x6,0x13,0x61,0x22,0x71,0x81,0x14,0x32,0x91,0xa1,0x7,0x15,0xb1,0x42,0x23,0xc1,0x52,0xd1,0xe1,0x33,0x16,0x62,0xf0,0x24,0x72,0x82,0xf1,0x25,0x43,0x34,0x53,0x92,0xa2,0xb2,0x63,0x73,0xC2,0x35,0x44,0x27,0x93,0xa3,0xb3,0x36,0x17,0x54,0x64,0x74,0xc3,0xd2,0xe2,0x08,0x26,0x83,0x9,0xa,0x18,0x19,0x84,0x94,0x45,0x46,0xa4,0xb4,0x56,0xd3,0x55,0x28,0x1a,0xf2,0xe3,0xf3,0xc4,0xd4,0xe4,0xf4,0x65,0x75,0x85,0x95,0xa5,0xb5,0xc5,0xd5,0xe5,0xf5,0x66,0x76,0x86,0x96,0xa6,0xb6,0xc6,0xd6,0xe6,0xf6,0x37,0x47,0x57,0x67,0x77,0x87,0x97,0xa7,0xb7,0xc7,0xd7,0xe7,0xf7,0x38,0x48,0x58,0x68,0x78,0x88,0x98,0xa8,0xb8,0xc8,0xd8,0xe8,0xf8,0x29,0x39,0x49,0x59,0x69,0x79,0x89,0x99,0xa9,0xb9,0xc9,0xd9,0xe9,0xf9,0x2a,0x3a,0x4a,0x5a,0x6a,0x7a,0x8a,0x9a,0xaa,0xba,0xca,0xda,0xea,0xfa};
/* AC chrominance */
unsigned const char accrs[162]={0x1,0x0,0x2,0x11,0x3,0x4,0x21,0x12,0x31,0x41,0x5,0x51,0x13,0x61,0x22,0x06,0x71,0x81,0x91,0x32,0xa1,0xb1,0xf0,0x14,0xc1,0xd1,0xe1,0x23,0x42,0x15,0x52,0x62,0x72,0xf1,0x33,0x24,0x34,0x43,0x82,0x16,0x92,0x53,0x25,0xa2,0x63,0xb2,0xc2,0x07,0x73,0xd2,0x35,0xe2,0x44,0x83,0x17,0x54,0x93,0x8,0x9,0xa,0x18,0x19,0x26,0x36,0x45,0x1a,0x27,0x64,0x74,0x55,0x37,0xf2,0xa3,0xb3,0xc3,0x28,0x29,0xd3,0xe3,0xf3,0x84,0x94,0xa4,0xb4,0xc4,0xd4,0xe4,0xf4,0x65,0x75,0x85,0x95,0xa5,0xb5,0xc5,0xd5,0xe5,0xf5,0x46,0x56,0x66,0x76,0x86,0x96,0xa6,0xb6,0xc6,0xd6,0xe6,0xf6,0x47,0x57,0x67,0x77,0x87,0x97,0xa7,0xb7,0xc7,0xd7,0xe7,0xf7,0x38,0x48,0x58,0x68,0x78,0x88,0x98,0xa8,0xb8,0xc8,0xd8,0xe8,0xf8,0x39,0x49,0x59,0x69,0x79,0x89,0x99,0xa9,0xb9,0xC9,0xd9,0xe9,0xf9,0x2a,0x3a,0x4a,0x5a,0x6a,0x7a,0x8a,0x9a,0xaa,0xba,0xca,0xda,0xea,0xfa};
/* AC Luminance values */
int acvallu[162]={0, 1, 4, 10, 11, 12, 26, 27, 28, 58, 59, 120, 121, 122, 123, 248, 249, 500, 501, 502, 503, 504, 505, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 2038, 2039, 2040, 4082, 4083, 4084, 4085, 8172, 8173, 16348, 16349, 16350, 16351, 16352, 16353, 32708, 32709, 65420, 65421, 65422, 65423, 65424, 65425, 65426, 65427, 65428, 65429, 65430, 65431, 65432, 65433, 65434, 65435, 65436, 65437, 65438, 65439, 65440, 65441, 65442, 65443, 65444, 65445, 65446, 65447, 65448, 65449, 65450, 65451, 65452, 65453, 65454, 65455, 65456, 65457, 65458, 65459, 65460, 65461, 65462, 65463, 65464, 65465, 65466, 65467, 65468, 65469, 65470, 65471, 65472, 65473, 65474, 65475, 65476, 65477, 65478, 65479, 65480, 65481, 65482, 65483, 65484, 65485, 65486, 65487, 65488, 65489, 65490, 65491, 65492, 65493, 65494, 65495, 65496, 65497, 65498, 65499, 65500, 65501, 65502, 65503, 65504, 65505, 65506, 65507, 65508, 65509, 65510, 65511, 65512, 65513, 65514, 65515, 65516, 65517, 65518, 65519, 65520, 65521, 65522, 65523, 65524, 65525, 65526, 65527, 65528, 65529, 65530, 65531, 65532, 65533, 65534};
/* AC Chrominance values */
int acvalcr[162]={0, 1, 4, 5, 12, 26, 27, 56, 57, 58, 118, 119, 120, 121, 122, 246, 247, 248, 249, 250, 502, 503, 504, 505, 1012, 1013, 1014, 1015, 1016, 2034, 2035, 2036, 2037, 2038, 2039, 4080, 4081, 4082, 4083, 8168, 8169, 8170, 8171, 8172, 8173, 8174, 8175, 16352, 16353, 16354, 32710, 32711, 32712, 65426, 65427, 65428, 65429, 65430, 65431, 65432, 65433, 65434, 65435, 65436, 65437, 65438, 65439, 65440, 65441, 65442, 65443, 65444, 65445, 65446, 65447, 65448, 65449, 65450, 65451, 65452, 65453, 65454, 65455, 65456, 65457, 65458, 65459, 65460, 65461, 65462, 65463, 65464, 65465, 65466, 65467, 65468, 65469, 65470, 65471, 65472, 65473, 65474, 65475, 65476, 65477, 65478, 65479, 65480, 65481, 65482, 65483, 65484, 65485, 65486, 65487, 65488, 65489, 65490, 65491, 65492, 65493, 65494, 65495, 65496, 65497, 65498, 65499, 65500, 65501, 65502, 65503, 65504, 65505, 65506, 65507, 65508, 65509, 65510, 65511, 65512, 65513, 65514, 65515, 65516, 65517, 65518, 65519, 65520, 65521, 65522, 65523, 65524, 65525, 65526, 65527, 65528, 65529, 65530, 65531, 65532, 65533, 65534};

/* Discrete Cosine Transform matrices */
float dct[8][8];
float dct1[8][8];
float dct2[8][8];
float dct3[8][8];
float dct4[8][8];


int diff[3]={0};
int prev[3]={0};

/* Conversion of 2-D array into 1-D array using zig-zag method */
void zigzag(int aa[63],int a[8][8],int flag)
{
int i;
diff[flag]=a[0][0]-prev[flag];
int zig[63]={
    a[0][1],a[1][0],
    a[2][0],a[1][1],a[0][2],
    a[0][3],a[1][2],a[2][1],a[3][0],
    a[4][0],a[3][1],a[2][2],a[1][3],a[0][4],
    a[0][5],a[1][4],a[2][3],a[3][2],a[4][1],a[5][0],
    a[6][0],a[5][1],a[4][2],a[3][3],a[2][4],a[1][5],a[0][6],
    a[0][7],a[1][6],a[2][5],a[3][4],a[4][3],a[5][2],a[6][1],a[7][0],
    a[7][1],a[6][2],a[5][3],a[4][4],a[3][5],a[2][6],a[1][7],
    a[2][7],a[3][6],a[4][5],a[5][4],a[6][3],a[7][2],
    a[7][3],a[6][4],a[5][5],a[4][6],a[3][7],
    a[4][7],a[5][6],a[6][5],a[7][4],
    a[7][5],a[6][6],a[5][7],
    a[6][7],a[7][6],
    a[7][7]};


for(i=0;i<63;i++)
aa[i]=zig[i];

prev[flag]=a[0][0];
}



int zerocount=0;


/* return Msb */
int getbit(int n,int b)
{
	
	int temp;
	temp=pow(2,b);
	temp=temp&(n);
	if(temp==0)	
		temp=0;
	else 
		temp=1;
	
	return temp; 
}

//writes the bit to the file, when the buffer is full.
void putbit(FILE *f,int bit){
	
	
	
	buf[1]+=(int)bit<<buf[0];
	if(buf[0]==0){
		
		fputc(buf[1],f);
		if(buf[1]==0xff)
		fputc(0x00,f);		/* Character written into file is buf[1] */
		buf[0]=7;
		buf[1]=0;
	}
	else
	buf[0]--;
}

/* Huffman coding of AC Luminance values */
void huffaclu(int val,FILE *pFile)
{
	if(val==0)
	{
		zerocount++;
		
		if(zerocount==16)
		{
			int num=4083;
			int totsize=12;
			while(totsize!=0)
		{
		totsize--;
		int b=getbit(num,totsize);
		putbit(pFile,b);
		}
		zerocount=0;	
		}
	return;
	}
	else
	{
	int i=0,size;
	
	while(pow(2,i)<=abs(val))
	i++;
	size=i;
	
	int num=zerocount*16+size;
	for(i=0;i<162;i++)
	{
		if((int)aclus[i]==num)
		{
			
			int val1=acvallu[i];
			int j=0,k=0;
			while(j<=i)
			{
			j+=acluc[k];k++;
			}
			int totsize=k;
		while(totsize!=0)
		{
		totsize--;
		int b=getbit(val1,totsize);
		putbit(pFile,b);
		}
		int aval;int asize=size;
		if(val>0)
		{
			aval=val;
		}
		else
		{
			aval=pow(2,size)+val-1;
		}
		while(asize!=0)
		{
		asize--;
		int b=getbit(aval,asize);
		putbit(pFile,b);
		}
		zerocount=0;break;
		}
	}
	}
	}

/* Huffman values for DC Luminance */
void huffdclus(int diff1,int zig[63], FILE *pFile)
{
	int valarr[12]={0x06,0x05,0x03,0x02,0x00,0x01,0x04,0x0e,0x1e,0x3e,0x7e,0xfe};
	int i=0,size,bits;
	if(diff1==0)
	{
		putbit(pFile,1);
		putbit(pFile,1);
		putbit(pFile,0);
	}
	else
	{
	if(diff1==0)
	size=0;
	else
	{
	while(pow(2,i)<=abs(diff1))
	i++;
	size=i;
	}
	switch(size)
	{
	case 0:
	case 1:
	case 2:
	case 3:
	case 4:
	case 5:
	case 6:bits=3;break;
	case 7:bits=4;break;
	case 8:bits=5;break;
	case 9:bits=6;break;
	case 10:bits=7;break;
	case 11:bits=8;break;
	}
	int num,num2;
	if(diff1>=0)
	num=diff1;
	else
	num=pow(2,size)+(diff1-1);
	num2=valarr[size];
	num2=num2<<size;
	num=num|num2;
	int totsize=size+bits;
	
	while(totsize!=0)
	{
		totsize--;
		int b=getbit(num,totsize);
		putbit(pFile,b);
	}}
	zerocount=0;
	for(i=0;i<63;i++)
	{
	huffaclu(zig[i],pFile);
	}
	if(zerocount!=0)
	{
	putbit(pFile,1);
	putbit(pFile,1);
	putbit(pFile,0);
	putbit(pFile,0);
	}
	
}

/* Huffman coding of AC chrominance values */
void huffaccr(int val,FILE *pFile)
{
	if(val==0)
	{
		zerocount++;
		
		if(zerocount==16)
		{
			int num=504;
			int totsize=9;
			while(totsize!=0)
		{
		totsize--;
		int b=getbit(num,totsize);
		putbit(pFile,b);
		}
		zerocount=0;	
		}
	return;
	}
	else
	{
	int i=0,size;
	
	while(pow(2,i)<=abs(val))
	i++;
	size=i;
	
	int num=zerocount*16+size;
	for(i=0;i<162;i++)
	{
		if((int)accrs[i]==num)
		{
			int val1=acvalcr[i];
			int j=0,k=0;
			while(j<=i)
			{
			j+=accrc[k];k++;
			}
			int totsize=k;
		while(totsize!=0)
		{
		totsize--;
		int b=getbit(val1,totsize);
		putbit(pFile,b);
		}
		int aval;int asize=size;
		if(val>0)
		{
			aval=val;
		}
		else
		{
			aval=pow(2,size)+val-1;
		}
		while(asize!=0)
		{
		asize--;
		int b=getbit(aval,asize);
		putbit(pFile,b);
		}
		zerocount=0;break;
		}
	}
	}
	}

/* Huffman coding of DC chrominance values */
void huffdccr(int diff1,int zig[63], FILE *pFile)
{
	int valarr[12]={0x01,0x00,0x04,0x05,0x0c,0x0d,0x0e,0x1e,0x3e,0x7e,0xfe,0x1fe};
	int i=0,size,bits;
	if(diff1==0)
	{
		putbit(pFile,0);
		putbit(pFile,1);
	}
	else
	{
	
	
	
	
	while(pow(2,i)<=abs(diff1))
	i++;
	size=i;
	
	switch(size)
	{
	case 0:
	case 1:bits=2;break;
	case 2:
	case 3:bits=3;break;
	case 4:
	case 5:
	case 6:bits=4;break;
	case 7:bits=5;break;
	case 8:bits=6;break;
	case 9:bits=7;break;
	case 10:bits=8;break;
	case 11:bits=9;break;
	}
	int num,num2;
	if(diff1>=0)
	num=diff1;
	else
	num=pow(2,size)+(diff1-1);
	num2=valarr[size];
	num2=num2<<size;
	num=num|num2;
	int totsize=size+bits;
	
	while(totsize!=0)
	{
		totsize--;
		int b=getbit(num,totsize);
		putbit(pFile,b);
	}
	}
	zerocount=0;
	for(i=0;i<63;i++)
	{
	huffaccr(zig[i],pFile);
	}
	if(zerocount!=0)
	{
	putbit(pFile,0);
	putbit(pFile,1);
	}
	
}



void writeFile(int n,int *y2,int *cb1,int *cr1)
{
int i,j;
FILE *pFile1;
pFile1 = fopen ( "x.jpg" , "wb" );
  if (pFile1==NULL) {fputs ("File error",stderr); exit (1);}
	char *ch;

	
	ch=(char*)malloc(4*sizeof(char));
	//SOI
	fprintf(pFile1,"%c%c",0xFF,M_SOI);

	//APP0
	fprintf(pFile1,"%c%c",0xFF,M_APP0);
	fprintf(pFile1,"%c%c",0x00,16);
	fprintf(pFile1,"%c%c%c%c%c",'J','F','I','F',0x00);
	fprintf(pFile1,"%c%c",0x01,0x02);
	fprintf(pFile1,"%c",0x01);	
	fprintf(pFile1,"%c%c%c%c",0x00,0x60,0x00,0x60);
	fprintf(pFile1,"%c%c",0x00,0x00);
	
	//DQT
	fprintf(pFile1,"%c%c",0xFF,M_DQT);
	fprintf(pFile1,"%c%c",0,0x84);
	fprintf(pFile1,"%c",0x00);
	for(i=0;i<64;i++)
	fprintf(pFile1,"%c",quant1[i]);
	fprintf(pFile1,"%c",0x01);
	for(i=0;i<64;i++)
	fprintf(pFile1,"%c",quant2[i]);

	//SOF
	fprintf(pFile1,"%c%c",0xFF,M_SOF0);
	fprintf(pFile1,"%c%c",0x00,17);
	fprintf(pFile1,"%c",0x08);
	
	if(n==4096)
	{
	fprintf(pFile1,"%c%c",0x10,0x00);
	fprintf(pFile1,"%c%c",0x10,0x00);
	}
	else if(n==2048)
	{
	fprintf(pFile1,"%c%c",0x08,0x00);
	fprintf(pFile1,"%c%c",0x08,0x00);
	}
	else if(n==1024)
	{
	fprintf(pFile1,"%c%c",0x04,0x00);
	fprintf(pFile1,"%c%c",0x04,0x00);
	}
	else if(n==512)
	{
	fprintf(pFile1,"%c%c",0x02,0x00);
	fprintf(pFile1,"%c%c",0x02,0x00);
	}
	fprintf(pFile1,"%c",3);

	fprintf(pFile1,"%c",1);
	fprintf(pFile1,"%c",0x11);
	fprintf(pFile1,"%c",0);
	
	fprintf(pFile1,"%c",2);
	fprintf(pFile1,"%c",0x11);
	fprintf(pFile1,"%c",1);

	fprintf(pFile1,"%c",3);
	fprintf(pFile1,"%c",0x11);
	fprintf(pFile1,"%c",1);
	

	//DHT
	fprintf(pFile1,"%c%c",0xFF,M_DHT);
	fprintf(pFile1,"%c%c",0x01,0xa2);
	fprintf(pFile1,"%c",0x00);	
	for(i=0;i<16;i++)
	{
		fprintf(pFile1,"%c",dcluc[i]);
	}
		for(i=0;i<12;i++)
	{
		fprintf(pFile1,"%c",dclus[i]);
	}
	fprintf(pFile1,"%c",0x01);
	for(i=0;i<16;i++)
	{
		fprintf(pFile1,"%c",dccrc[i]);
	}
		for(i=0;i<12;i++)
	{
		fprintf(pFile1,"%c",dccrs[i]);
	}
	fprintf(pFile1,"%c",0x10);
	for(i=0;i<16;i++)
	{
		fprintf(pFile1,"%c",acluc[i]);
	}
		for(i=0;i<162;i++)
	{
		fprintf(pFile1,"%c",aclus[i]);
	}
	fprintf(pFile1,"%c",0x11);
	for(i=0;i<16;i++)
	{
		fprintf(pFile1,"%c",accrc[i]);
	}
		for(i=0;i<162;i++)
	{
		fprintf(pFile1,"%c",accrs[i]);
	}
		
	//SOS
	fprintf(pFile1,"%c%c",0xFF,M_SOS);
	fprintf(pFile1,"%c%c",0x00,12);
	fprintf(pFile1,"%c",3);
	fprintf(pFile1,"%c",1);
	fprintf(pFile1,"%c",0x00);
	fprintf(pFile1,"%c",2);
	fprintf(pFile1,"%c",0x11);
	fprintf(pFile1,"%c",3);
	fprintf(pFile1,"%c",0x11);
	fprintf(pFile1,"%c",0);
	fprintf(pFile1,"%c",63);
	fprintf(pFile1,"%c",0x00);
	
	int aa[8][8],ab[8][8],ac[8][8],ii,jj;int *zig;
	zig=(int*)malloc(63*sizeof(int));
	for(i=0;i<n;i+=8)
	{
		for(j=0;j<n;j+=8)
		{
			for(ii=i;ii<i+8;ii++)
			{
					for(jj=j;jj<j+8;jj++)
					{
					aa[ii-i][jj-j]=y2[ii*n+jj];
					
					
					ab[ii-i][jj-j]=cb1[ii*n+jj];
					
					ac[ii-i][jj-j]=cr1[ii*n+jj];
					}	
			}
					zigzag(zig,aa,0);
					huffdclus(diff[0],zig,pFile1);
					
					zigzag(zig,ab,1);
					huffdccr(diff[1],zig,pFile1);
					zigzag(zig,ac,2);
					huffdccr(diff[2],zig,pFile1);
					
					
				
			
		}
	}
	
	while(buf[0]!=7)
	putbit(pFile1,0);
	
	fprintf(pFile1,"%c%c",0xFF,M_EOI);
	
 fclose (pFile1);


}

int main()
{


cl_mem yobj = NULL;
cl_mem ipgmobj = NULL;
cl_mem cbobj = NULL;
cl_mem crobj = NULL;
cl_mem y2obj = NULL;
cl_mem cb1obj = NULL;
cl_mem cr1obj = NULL;


cl_kernel hpfl = NULL;


cl_platform_id platform_id = NULL;

cl_uint ret_num_devices;
cl_uint ret_num_platforms;

cl_int ret;


cl_int *y2;
cl_int *cb1;
cl_int *cr1;

cl_event event;
cl_ulong start;
cl_ulong end;

ppm_t ipgm;
FILE *fp;
const char fileName[] = "jpeg.cl";
size_t source_size;
char *source_str;

cl_int n;
size_t gws[2];
size_t lws[2];

/* Load kernel source code */
fp = fopen(fileName, "r");
if (!fp) {
fprintf(stderr, "Failed to load kernel.¥n");
exit(1);
}
source_str = (char *)malloc(MAX_SOURCE_SIZE);
source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
fclose( fp );

char fname[30];

printf("Enter the name of image:");
scanf("%s",fname);
/* Read image */
readPPM(&ipgm, fname);
n = ipgm.width;

y2 = (int *)malloc(n * n * sizeof(cl_int));
cb1 = (int *)malloc(n * n * sizeof(cl_int));
cr1 = (int *)malloc(n * n * sizeof(cl_int));

/* Get platform/device*/
ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
printf("1. %d\n",ret);
ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id,&ret_num_devices);
printf("2. %d\n",ret);

/* Create OpenCL context */
context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

/* Create Command queue */
queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);

/* Create Buffer Objects */
ipgmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, 3*n*n*sizeof(unsigned char), NULL,&ret);

yobj = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, n*n*sizeof(cl_float), NULL,&ret);
cbobj = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, n*n*sizeof(cl_float), NULL,&ret);
crobj = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, n*n*sizeof(cl_float), NULL,&ret);
y2obj = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, n*n*sizeof(cl_int), NULL,&ret);
cb1obj = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, n*n*sizeof(cl_int), NULL,&ret);
cr1obj = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, n*n*sizeof(cl_int), NULL,&ret);

/* Transfer data to memory buffer */


ret = clEnqueueWriteBuffer(queue, cbobj, CL_TRUE, 0, n*n*sizeof(cl_float), NULL, 0,NULL, &event);
ret = clEnqueueWriteBuffer(queue, crobj, CL_TRUE, 0, n*n*sizeof(cl_float), NULL, 0,NULL, &event);
ret = clEnqueueWriteBuffer(queue, yobj, CL_TRUE, 0, n*n*sizeof(cl_float), NULL, 0,NULL, &event);

ret = clEnqueueWriteBuffer(queue, ipgmobj, CL_TRUE, 0, 3*n*n*sizeof(unsigned char), ipgm.buf, 0,NULL, &event);

clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start,NULL);
printf("3. %d\n",ret);


/* Create kernel program from source */

program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
if(!ret==CL_SUCCESS)
printf("12 %d\n",ret);
/* Build kernel program */
ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
if(!ret==CL_SUCCESS)
printf("15 %d\n",ret);

/* Compute luminance and chrominance values from rgb values */
hpfl = clCreateKernel(program, "compute1", &ret);
if(!ret==CL_SUCCESS)
printf("1 %d\n",ret);

ret = clSetKernelArg(hpfl, 0, sizeof(cl_mem), (void *)&ipgmobj);
ret = clSetKernelArg(hpfl, 1, sizeof(cl_mem), (void *)&yobj);
ret = clSetKernelArg(hpfl, 2, sizeof(cl_mem), (void *)&cbobj);
ret = clSetKernelArg(hpfl, 3, sizeof(cl_mem), (void *)&crobj);
ret = clSetKernelArg(hpfl, 4, sizeof(cl_int), (void *)&n);
setWorkSize(gws, lws,n, n);
ret = clEnqueueNDRangeKernel(queue, hpfl, 2, NULL, gws, lws, 0, NULL, &event);
if(!ret==CL_SUCCESS)
printf("1 %d\n",ret);
clWaitForEvents(1, &event);



hpfl = clCreateKernel(program, "mul1", &ret);
if(!ret==CL_SUCCESS)
printf("1 %d\n",ret);

/* DCT and quantization on luminance values */

int fl=1;
ret = clSetKernelArg(hpfl, 0, sizeof(cl_mem), (void *)&yobj);
ret = clSetKernelArg(hpfl, 1, sizeof(cl_mem), (void *)&y2obj);
ret = clSetKernelArg(hpfl,2, sizeof(cl_int), (void *)&n);
ret = clSetKernelArg(hpfl, 3, sizeof(cl_int), (void *)&fl);
setWorkSize(gws, lws,n, n);
ret = clEnqueueNDRangeKernel(queue, hpfl, 2, NULL, gws, lws, 0, NULL, &event);
if(!ret==CL_SUCCESS)
printf("1 %d\n",ret);
clWaitForEvents(1, &event);

/* DCT and quantization on cb */

fl=2;
ret = clSetKernelArg(hpfl, 0, sizeof(cl_mem), (void *)&cbobj);
ret = clSetKernelArg(hpfl, 1, sizeof(cl_mem), (void *)&cb1obj);
ret = clSetKernelArg(hpfl,2, sizeof(cl_int), (void *)&n);
ret = clSetKernelArg(hpfl, 3, sizeof(cl_int), (void *)&fl);
setWorkSize(gws, lws,n, n);
ret = clEnqueueNDRangeKernel(queue, hpfl, 2, NULL, gws, lws, 0, NULL, &event);
if(!ret==CL_SUCCESS)
printf("111 %d\n",ret);
clWaitForEvents(1, &event);

/* DCT and quantization on cr */
fl=2;
ret = clSetKernelArg(hpfl, 0, sizeof(cl_mem), (void *)&crobj);
ret = clSetKernelArg(hpfl, 1, sizeof(cl_mem), (void *)&cr1obj);
ret = clSetKernelArg(hpfl,2, sizeof(cl_int), (void *)&n);
ret = clSetKernelArg(hpfl, 3, sizeof(cl_int), (void *)&fl);
setWorkSize(gws, lws,n, n);
ret = clEnqueueNDRangeKernel(queue, hpfl, 2, NULL, gws, lws, 0, NULL, &event);
if(!ret==CL_SUCCESS)
printf("1 %d\n",ret);
clWaitForEvents(1, &event);




ret = clEnqueueReadBuffer(queue, y2obj, CL_TRUE, 0, n*n*sizeof(cl_int), y2, 0,NULL, NULL);
if(!ret==CL_SUCCESS)
printf("15 %d\n",ret);
ret = clEnqueueReadBuffer(queue, cb1obj, CL_TRUE, 0, n*n*sizeof(cl_int), cb1, 0,NULL, NULL);
if(!ret==CL_SUCCESS)
printf("15 %d\n",ret);
ret = clEnqueueReadBuffer(queue, cr1obj, CL_TRUE, 0, n*n*sizeof(cl_int), cr1, 0,NULL, &event);
if(!ret==CL_SUCCESS)
printf("15 %d\n",ret);


clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
printf(" memory buffer write: %10.5f [ms]¥n", (end - start)/1000000.0);


writeFile(n,y2,cb1,cr1);

/* Finalizations*/
ret = clFlush(queue);
ret = clFinish(queue);
ret = clReleaseKernel(hpfl);

ret = clReleaseProgram(program);

ret = clReleaseMemObject(yobj);
ret = clReleaseMemObject(cbobj);
ret = clReleaseMemObject(crobj);
ret = clReleaseMemObject(y2obj);
ret = clReleaseMemObject(cb1obj);
ret = clReleaseMemObject(cr1obj);

ret = clReleaseCommandQueue(queue);
ret = clReleaseContext(context);


free(source_str);


free(y2);
free(cb1);
free(cr1);

return 0;
}

