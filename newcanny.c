#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif


#include "pgm.h"
#include "ppm.h"

#define PI 3.14159265358979
#define E 2.7182818284
#define SIGMA 0.999999
#define THRESHU 80
#define THRESHL 30
#define MAX_SOURCE_SIZE (0x100000)
float kernx[5][5],kerny[5][5];
int sobelx[5][5]={{15,69,114,69,15},{35,155,255,155,35},{0,0,0,0,0},{-35,-155,-255,-155,-35},{-15,-69,-114,-69,-15}};
int sobely[5][5]={{15,35,0,-35,-15},{69,155,0,-155,-69},{114,255,0,-255,-114},{69,155,0,-155,-69},{15,35,0,-35,-15}};
cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue queue = NULL;
cl_program program = NULL;

/* set Local and Global worksize */
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

lws[0] = 32;
lws[1] = 16;
 
break;}

return 0;
}

/* Identifies only those pixels as edges that are within the Threshold */
void threshold(float *ampd,int i,int j,int n,int *flag)
{

flag[i*n+j]=1;
if(flag[i*n+j]!=1&&ampd[i*n+j]<THRESHL)
{
if(i!=0&&i!=n-1)
{
threshold(ampd,i+1,j,n,flag);
if(j!=n-1)
{
threshold(ampd,i+1,j+1,n,flag);
threshold(ampd,i,j+1,n,flag);
threshold(ampd,i-1,j+1,n,flag);
}
if(j!=0)
{
threshold(ampd,i+1,j-1,n,flag);
threshold(ampd,i,j-1,n,flag);
threshold(ampd,i-1,j-1,n,flag);

}
threshold(ampd,i-1,j,n,flag);
}
else if(i==0)
{
threshold(ampd,i+1,j,n,flag);
if(j!=n-1)
{
threshold(ampd,i+1,j+1,n,flag);
threshold(ampd,i,j+1,n,flag);

}
if(j!=0)
{
threshold(ampd,i+1,j-1,n,flag);
threshold(ampd,i,j-1,n,flag);


}
threshold(ampd,i-1,j,n,flag);
}
else if(i==n-1)
{
threshold(ampd,i+1,j,n,flag);
if(j!=n-1)
{

threshold(ampd,i,j+1,n,flag);
threshold(ampd,i-1,j+1,n,flag);
}
if(j!=0)
{

threshold(ampd,i,j-1,n,flag);
threshold(ampd,i-1,j-1,n,flag);

}
threshold(ampd,i-1,j,n,flag);
}
}
else if(flag[i*n+j]==1)
return;
else if(ampd[i*n+j]<THRESHL)
ampd[i*n+j]=255;
}

/* Quantization based on gradient values and direction */
void quant1(float *dy,float *dx,float *ampd,int n)
{

int i,j;
int *quant;
quant = (int *)malloc(n * n * sizeof(int));
for (i=0; i < n; i++) {
for (j=0; j < n; j++) {
float ang=atan2(dy[i*n+j],dx[i*n+j]);
float theta=ang*180/PI+180;
if((theta>=0&&theta<22.5)||(theta>=157.5&&theta<202.5)||(theta>=337.5&&theta<360))
quant[n*((i))+((j))]=0;
else if((theta>=22.5&&theta<67.5)||(theta>=202.5&&theta<247.5))
quant[n*((i))+((j))]=1;
else if((theta>=67.5&&theta<112.5)||(theta>=247.5&&theta<292.5))
quant[n*((i))+((j))]=2;
else if((theta>=112.5&&theta<157.5)||(theta>=292.5&&theta<337.5))
quant[n*((i))+((j))]=3;

if(i>0&&i<n-1&&j>0&&j<n-1)
{
if((quant[n*((i))+((j))]==0&&ampd[n*(i-1)+j]<ampd[n*(i)+j]&&ampd[n*(i+1)+j]<ampd[n*(i)+j]))
ampd[n*i+j]=255;
else if((quant[n*((i))+((j))]==1&&ampd[n*(i-1)+(j-1)]<ampd[n*(i)+j]&&ampd[n*(i+1)+(j+1)]<ampd[n*(i)+j]))
ampd[n*i+j]=255;
else if((quant[n*((i))+((j))]==2&&ampd[n*(i)+(j+1)]<ampd[n*(i)+j]&&ampd[n*(i)+(j-1)]<ampd[n*(i)+j]))
ampd[n*i+j]=255;
else if((quant[n*((i))+((j))]==3&&ampd[n*(i-1)+(j+1)]<ampd[n*(i)+j]&&ampd[n*(i+1)+(j-1)]<ampd[n*(i)+j]))
ampd[n*i+j]=255;
}
if(i==0||j==0||(j==n-1)||(i==n-1))
ampd[n*i+j]=0;

}}

}


int main()
{
cl_mem xmobj=NULL;
cl_mem yobj = NULL;
cl_mem y1obj = NULL;
cl_mem ampdobj = NULL;
cl_mem ipgmobj = NULL;
cl_kernel hpfl = NULL;
cl_kernel copy = NULL;


cl_platform_id platform_id = NULL;

cl_uint ret_num_devices;
cl_uint ret_num_platforms;

cl_int ret;

cl_event event;
cl_ulong start;
cl_ulong end;


float *y;
float *dx;
float *dy;

int *flag;
ppm_t ipgm;
pgm_t opgm;
int i, j;
int n;
FILE *fp;
const char fileName[] = "newced.cl";
size_t source_size;
char *source_str;

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

/* Read r */
readPPM(&ipgm, "egg1.ppm");


n = ipgm.width;
int size=5;
/* Computing Gaussian kernel  */
for(i=0;i<size;i++)
{
	for(j=0;j<size;j++)
	{
		if(i==0&&j==0)
		kernx[size/2][size/2]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA)));
		if(i==0&&j==1)
		{
		kernx[size/2-1][size/2]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA)));
		kernx[size/2+1][size/2]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA)));
		kernx[size/2][size/2-1]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA))) ;
		kernx[size/2][size/2+1]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA))) ;
		}
		if(i==1&&j==1)
		{
		kernx[size/2-1][size/2-1]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA))) ;
		kernx[size/2+1][size/2+1]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA))) ;
		kernx[size/2+1][size/2-1]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA))) ;
		kernx[size/2-1][size/2+1]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA))) ;
		}
		if(i==0&&j==2)
		{
		kernx[size/2-2][size/2]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA))) ;
		kernx[size/2+2][size/2]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA))) ;
		kernx[size/2][size/2-2]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA))) ;
		kernx[size/2][size/2+2]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA))) ;
		}
		if(i==1&&j==2)
		{
		kernx[size/2-1][size/2+2]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA))) ;
		kernx[size/2+1][size/2+2]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA))) ;
		kernx[size/2-1][size/2-2]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA))) ;
		kernx[size/2+1][size/2-2]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA))) ;
		kernx[size/2-2][size/2-1]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA))) ;
		kernx[size/2+2][size/2+1]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA))) ;
		kernx[size/2+2][size/2-1]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA))) ;
		kernx[size/2-2][size/2+1]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA))) ;
		}
		if(i==2&&j==2)
		{
		kernx[size/2-2][size/2-2]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA))) ;
		kernx[size/2+2][size/2+2]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA))) ;
		kernx[size/2+2][size/2-2]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA))) ;
		kernx[size/2-2][size/2+2]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA))) ;
		}
		
		
	}
}

y = (float *)malloc(n * n * sizeof(float));
dx = (float *)malloc(n * n * sizeof(float));
dy = (float *)malloc(n * n * sizeof(float));

flag = (int *)malloc(n * n * sizeof(int));
float* ampd;
ampd = (float*)malloc(n*n*sizeof(float));

/* Get platform/device*/
ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
if(ret!=CL_SUCCESS)
printf("1111\n");

ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id,&ret_num_devices);
if(ret!=CL_SUCCESS)
printf("1111\n");

/* Create OpenCL context */
context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
if(ret!=CL_SUCCESS)
printf("1111\n");
/* Create Command queue */
queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);

/* Create Buffer Objects */
xmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, 25*sizeof(cl_float), NULL,&ret);
yobj = clCreateBuffer(context, CL_MEM_READ_WRITE, n*n*sizeof(cl_float), NULL,&ret);
y1obj = clCreateBuffer(context, CL_MEM_READ_WRITE, n*n*sizeof(cl_float), NULL,&ret);
ampdobj = clCreateBuffer(context, CL_MEM_READ_WRITE, n*n*sizeof(cl_float), NULL,&ret);
ipgmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, 3*n*n*sizeof(unsigned char), NULL,&ret);



/* Transfer data to memory buffer */


ret = clEnqueueWriteBuffer(queue, y1obj, CL_TRUE, 0, n*n*sizeof(cl_float), NULL, 0,NULL, &event);
ret = clEnqueueWriteBuffer(queue, ampdobj, CL_TRUE, 0, n*n*sizeof(cl_float), NULL, 0,NULL, &event);
ret = clEnqueueWriteBuffer(queue, xmobj, CL_TRUE, 0, 25*sizeof(cl_float), kernx, 0,NULL, &event);
ret = clEnqueueWriteBuffer(queue, yobj, CL_TRUE, 0, n*n*sizeof(cl_float), y, 0,NULL, &event);
ret = clEnqueueWriteBuffer(queue, ipgmobj, CL_TRUE, 0, 3*n*n*sizeof(unsigned char), ipgm.buf, 0,NULL, &event);



clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start,NULL);
/* Create kernel program from source */
program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
if(!ret==CL_SUCCESS)
printf("12 %d\n",ret);
/* Build kernel program */
ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
if(!ret==CL_SUCCESS)
printf("15 %d\n",ret);


hpfl = clCreateKernel(program, "compute1", &ret);
if(!ret==CL_SUCCESS)
printf("1 %d\n",ret);

ret = clSetKernelArg(hpfl, 0, sizeof(cl_mem), (void *)&ipgmobj);
ret = clSetKernelArg(hpfl, 1, sizeof(cl_mem), (void *)&yobj);
ret = clSetKernelArg(hpfl, 2, sizeof(cl_int), (void *)&n);
setWorkSize(gws, lws,n, n);
ret = clEnqueueNDRangeKernel(queue, hpfl, 2, NULL, gws, lws, 0, NULL, &event);
if(!ret==CL_SUCCESS)
printf("1 %d\n",ret);
clWaitForEvents(1, &event);

/* Gaussian blur */
hpfl = clCreateKernel(program, "kern", &ret);
if(!ret==CL_SUCCESS)
printf("1 %d\n",ret);

/* Canny edge detection */
copy = clCreateKernel(program, "ced", &ret);
if(!ret==CL_SUCCESS)
printf("1 %d\n",ret);



ret = clSetKernelArg(hpfl, 0, sizeof(cl_mem), (void *)&yobj);
ret = clSetKernelArg(hpfl, 1, sizeof(cl_mem), (void *)&xmobj);
ret = clSetKernelArg(hpfl, 2, sizeof(cl_mem), (void *)&y1obj);
ret = clSetKernelArg(hpfl, 3, sizeof(cl_int), (void *)&n);
setWorkSize(gws, lws,n, n);
ret = clEnqueueNDRangeKernel(queue, hpfl, 2, NULL, gws, lws, 0, NULL, &event);
if(!ret==CL_SUCCESS)
printf("133 %d\n",ret);

clWaitForEvents(1, &event);

ret = clSetKernelArg(copy, 0, sizeof(cl_mem), (void *)&y1obj);
ret = clSetKernelArg(copy, 1, sizeof(cl_mem), (void *)&ampdobj);
ret = clSetKernelArg(copy, 2, sizeof(cl_int), (void *)&n);
setWorkSize(gws, lws,n, n);
ret = clEnqueueNDRangeKernel(queue, copy, 2, NULL, gws, lws, 0, NULL, &event);
if(!ret==CL_SUCCESS)
printf("133 %d\n",ret);

clWaitForEvents(1, &event);

ret = clEnqueueReadBuffer(queue, ampdobj, CL_TRUE, 0, n*n*sizeof(cl_float), ampd, 0,NULL, &event);
if(!ret==CL_SUCCESS)
printf("15 %d\n",ret);

clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
printf(" memory buffer write: %10.5f [ms]¥n", (end - start)/1000000.0);



for (i=0; i < n; i++) {
for (j=0; j < n; j++) {
if(flag[(n*i+j)]!=1&&ampd[n*i+j]<THRESHU)
threshold(ampd,i,j,n,flag);//flags a pixel as an edge or not an edge
else if(ampd[n*i+j]>THRESHU)
{
ampd[n*i+j]=0;flag[(n*i+j)]=1;
}}}

opgm.width = n;
opgm.height = n;
normalizeF2PGM(&opgm, ampd);

writePGM(&opgm,"output.pgm");

/* Finalizations */
free(ampd);
destroyPPM(&ipgm);
destroyPGM(&opgm);
ret = clFinish(queue);
ret = clReleaseKernel(hpfl);
free(y);
ret = clReleaseProgram(program);
ret = clReleaseMemObject(xmobj);
ret = clReleaseMemObject(yobj);
ret = clReleaseMemObject(y1obj);
ret = clReleaseMemObject(ampdobj);

ret = clReleaseCommandQueue(queue);
ret = clReleaseContext(context);


return 0;
}

