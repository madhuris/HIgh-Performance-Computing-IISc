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

#define AMP(a, b) (sqrt((a)*(a)+(b)*(b)))

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue queue = NULL;
cl_program program = NULL;

float kern[5][5];

/* Setting the Local and Global Work size */
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
lws[0] = 16;
lws[1] = 16;
 
break;}

return 0;
}





int main()
{

cl_mem xmobj=NULL;
cl_mem robj = NULL;
cl_mem gobj = NULL;
cl_mem bobj = NULL;
cl_mem rmobj = NULL;
cl_mem gmobj = NULL;
cl_mem bmobj = NULL;
cl_mem cobj = NULL;
cl_mem oobj = NULL;

cl_kernel hpfl = NULL;

cl_platform_id platform_id = NULL;

cl_uint ret_num_devices;

cl_uint ret_num_platforms;

cl_int ret;

cl_float *xm;
cl_float *r;
cl_float *g;
cl_float *b;

cl_event event;

cl_ulong start;
cl_ulong end;


ppm_t ipgm;//input image structure
ppm_t opgm;//output image structure

FILE *fp;
const char fileName[] = "gauss.cl";
size_t source_size;
char *source_str;

cl_int n;
cl_int m;

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

/* Read image */
readPPM(&ipgm, "lena.ppm");

n = ipgm.width;
m = (cl_int)(log((double)n)/log(2.0));

xm = (float *)malloc(n * n * sizeof(cl_float));


/* Gaussian Kernel */
int i,j,size=5;
for(i=0;i<size;i++)
{
	for(j=0;j<size;j++)
	{
		if(i==0&&j==0)
		kern[size/2][size/2]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA)));
		if(i==0&&j==1)
		{
		kern[size/2-1][size/2]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA)));
		kern[size/2+1][size/2]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA)));
		kern[size/2][size/2-1]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA)));
		kern[size/2][size/2+1]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA)));
		}
		if(i==1&&j==1)
		{
		kern[size/2-1][size/2-1]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA)));
		kern[size/2+1][size/2+1]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA)));
		kern[size/2+1][size/2-1]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA)));
		kern[size/2-1][size/2+1]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA)));
		}
		if(i==0&&j==2)
		{
		kern[size/2-2][size/2]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA)));
		kern[size/2+2][size/2]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA)));
		kern[size/2][size/2-2]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA)));
		kern[size/2][size/2+2]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA)));
		}
		if(i==1&&j==2)
		{
		kern[size/2-1][size/2+2]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA)));
		kern[size/2+1][size/2+2]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA)));
		kern[size/2-1][size/2-2]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA)));
		kern[size/2+1][size/2-2]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA)));
		kern[size/2-2][size/2-1]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA)));
		kern[size/2+2][size/2+1]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA)));
		kern[size/2+2][size/2-1]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA)));
		kern[size/2-2][size/2+1]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA)));
		}
		if(i==2&&j==2)
		{
		kern[size/2-2][size/2-2]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA)));
		kern[size/2+2][size/2+2]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA)));
		kern[size/2+2][size/2-2]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA)));
		kern[size/2-2][size/2+2]=(1.0/(2*PI*SIGMA*SIGMA))*pow(E,(((i)*(i)+(j)*(j))/(-2*SIGMA*SIGMA)));
		}
		
		
	}
}
r=(float *)malloc(n*n*sizeof(cl_float));
b=(float *)malloc(n*n*sizeof(cl_float));
g=(float *)malloc(n*n*sizeof(cl_float));

for (i=0; i < n; i++) {
for (j=0; j < n; j++) {


r[i*n+j] = (float)ipgm.buf[3*n*i+3*j];
g[i*n+j]=(float)ipgm.buf[3*n*i+3*j+1];
b[i*n+j]=(float)ipgm.buf[3*n*i+3*j+2];

}}

/* Get platform/device*/
ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id,&ret_num_devices);

/* Create OpenCL context */
context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

/* Create Command queue */
queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);

/* Create Buffer Objects */
xmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, 25*sizeof(cl_float), NULL,&ret); //Memory object for the 5*5 Gaussian Kernel
robj = clCreateBuffer(context, CL_MEM_READ_WRITE, n*n*sizeof(cl_float), NULL,&ret);
gobj = clCreateBuffer(context, CL_MEM_READ_WRITE, n*n*sizeof(cl_float), NULL,&ret);
bobj = clCreateBuffer(context, CL_MEM_READ_WRITE, n*n*sizeof(cl_float), NULL,&ret);
rmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, n*n*sizeof(cl_float), NULL,&ret);
gmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, n*n*sizeof(cl_float), NULL,&ret);
bmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, n*n*sizeof(cl_float), NULL,&ret);
cobj = clCreateBuffer(context, CL_MEM_READ_WRITE, n*n*3*sizeof(cl_char), NULL,&ret);
oobj = clCreateBuffer(context, CL_MEM_READ_WRITE, n*n*3*sizeof(cl_char), NULL,&ret);


/* Transfer data to memory buffer */
ret = clEnqueueWriteBuffer(queue, xmobj, CL_TRUE, 0, 25*sizeof(cl_float), kern, 0,NULL, &event);
ret = clEnqueueWriteBuffer(queue, robj, CL_TRUE, 0, n*n*sizeof(cl_float), r, 0,NULL, &event);
ret = clEnqueueWriteBuffer(queue, gobj, CL_TRUE, 0, n*n*sizeof(cl_float), g, 0,NULL, &event);
ret = clEnqueueWriteBuffer(queue, bobj, CL_TRUE, 0, n*n*sizeof(cl_float), b, 0,NULL, &event);
ret = clEnqueueWriteBuffer(queue, rmobj, CL_TRUE, 0, n*n*sizeof(cl_float), NULL, 0,NULL, &event);
ret = clEnqueueWriteBuffer(queue, gmobj, CL_TRUE, 0, n*n*sizeof(cl_float), NULL, 0,NULL, &event);
ret = clEnqueueWriteBuffer(queue, bmobj, CL_TRUE, 0, n*n*sizeof(cl_float), NULL, 0,NULL, &event);
ret = clEnqueueWriteBuffer(queue, cobj, CL_TRUE, 0, n*n*3*sizeof(cl_char), ipgm.buf, 0,NULL, &event);
ret = clEnqueueWriteBuffer(queue, oobj, CL_TRUE, 0, n*n*3*sizeof(cl_char), NULL, 0,NULL, &event);

/* Start Profiling */
/* Use this when OpenCL profiler is not in use */
clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start,
NULL);

/* Create kernel program from source */
program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
if(!ret==CL_SUCCESS)
printf("12 %d\n",ret);

/* Build kernel program */
ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
if(!ret==CL_SUCCESS)
printf("15 %d\n",ret);

char *buffer;
buffer=(char*)malloc(2000*sizeof(char));
clGetProgramBuildInfo(program,device_id,CL_PROGRAM_BUILD_LOG,2000*sizeof(char),buffer,NULL);
printf("%s",buffer);

hpfl = clCreateKernel(program, "gaussFilter", &ret);
if(!ret==CL_SUCCESS)
printf("1 %d\n",ret);


/* Apply gauss filter on red*/

ret = clSetKernelArg(hpfl, 0, sizeof(cl_mem), (void *)&robj);	//Red
ret = clSetKernelArg(hpfl, 1, sizeof(cl_mem), (void *)&xmobj);	//Gaussian kernel
ret = clSetKernelArg(hpfl,2, sizeof(cl_mem), (void *)&rmobj);	//Convolved Red
ret = clSetKernelArg(hpfl, 3, sizeof(cl_int), (void *)&n);
setWorkSize(gws, lws,n, n);
ret = clEnqueueNDRangeKernel(queue, hpfl, 2, NULL, gws, lws, 0, NULL,&event);
if(!ret==CL_SUCCESS)
printf("11 %d\n",ret);
clWaitForEvents(1, &event);

/* Apply gauss filter on green*/

ret = clSetKernelArg(hpfl, 0, sizeof(cl_mem), (void *)&gobj);	//Green
ret = clSetKernelArg(hpfl, 1, sizeof(cl_mem), (void *)&xmobj);	//Gaussian kernel
ret = clSetKernelArg(hpfl, 2, sizeof(cl_mem), (void *)&gmobj);	//Convolved green
ret = clSetKernelArg(hpfl, 3, sizeof(cl_int), (void *)&n);
setWorkSize(gws, lws,n, n);
ret = clEnqueueNDRangeKernel(queue, hpfl, 2, NULL, gws, lws, 0, NULL, &event);
if(!ret==CL_SUCCESS)
printf("111 %d\n",ret);
clWaitForEvents(1, &event);



/* Apply gauss filter on blue*/

ret = clSetKernelArg(hpfl, 0, sizeof(cl_mem), (void *)&bobj);	//Blue
ret = clSetKernelArg(hpfl, 1, sizeof(cl_mem), (void *)&xmobj);	//Gaussian Kernel
ret = clSetKernelArg(hpfl, 2, sizeof(cl_mem), (void *)&bmobj);	//Convolved blue
ret = clSetKernelArg(hpfl, 3, sizeof(cl_int), (void *)&n);
setWorkSize(gws, lws,n, n);
ret = clEnqueueNDRangeKernel(queue, hpfl, 2, NULL, gws, lws, 0, NULL, &event);
if(!ret==CL_SUCCESS)
printf("1 %d\n",ret);
clWaitForEvents(1, &event);

opgm.buf = (unsigned char*)malloc(n *3* n * sizeof(unsigned char));

/* Read Convolved Red, Blue and Green */
ret = clEnqueueReadBuffer(queue, rmobj, CL_TRUE, 0, n*n*sizeof(cl_float), r, 0,NULL, NULL);
if(!ret==CL_SUCCESS)
printf("15 %d\n",ret);

ret = clEnqueueReadBuffer(queue, gmobj, CL_TRUE, 0, n*n*sizeof(cl_float), g, 0,NULL, NULL);
if(!ret==CL_SUCCESS)
printf("15 %d\n",ret);

ret = clEnqueueReadBuffer(queue, bmobj, CL_TRUE, 0, n*n*sizeof(cl_float), b, 0,NULL, NULL);
if(!ret==CL_SUCCESS)
printf("15 %d\n",ret);

clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
printf(" memory buffer write: %10.5f [ms]¥n", (end - start)/1000000.0);


float* ampd;
ampd = (float*)malloc(n*n*3*sizeof(float));


for (i=0; i < n; i++) {
for (j=0; j < n; j++) {
if(i>2&&i<n-3&&j>2&&j<n-3) //Border Conditions
{
ampd[n*((3*i))+((3*j))] = r[i*n+j];
ampd[n*((3*i))+((3*j))+1] = g[i*n+j];
ampd[n*((3*i))+((3*j))+2] = b[i*n+j];
}
else
{
ampd[n*((3*i))+((3*j))] = 0;
ampd[n*((3*i))+((3*j))+1] = 0;
ampd[n*((3*i))+((3*j))+2] = 0;
}
}
}
opgm.width = n;
opgm.height = n;

//Write convolved image into output.ppm
normalizeF2PPM(&opgm, ampd);
free(ampd);


/* Finalizations*/
ret = clFlush(queue);
ret = clFinish(queue);
ret = clReleaseKernel(hpfl);

ret = clReleaseProgram(program);

ret = clReleaseMemObject(robj);
ret = clReleaseMemObject(gobj);
ret = clReleaseMemObject(bobj);

ret = clReleaseCommandQueue(queue);
ret = clReleaseContext(context);

destroyPPM(&ipgm);
destroyPPM(&opgm);

free(source_str);

free(xm);


return 0;
}

