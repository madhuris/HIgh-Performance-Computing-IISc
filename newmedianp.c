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

#define MAX_SOURCE_SIZE (0x100000)

#define AMP(a, b) (sqrt((a)*(a)+(b)*(b))) 
cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue queue = NULL;
cl_program program = NULL;

/* Set Global and Local Worksize */
int setWorkSize(size_t* gws, size_t* lws, cl_int x, cl_int y)
{
switch(y) {
case 1:
gws[0] = 1;
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



int main()
{
cl_mem xmobj = NULL;
cl_mem rmobj = NULL;
cl_mem wmobj = NULL;
cl_mem robj = NULL;
cl_mem gobj = NULL;
cl_mem bobj = NULL;
cl_mem cobj = NULL;
cl_mem oobj = NULL;
cl_kernel hpfl = NULL;
cl_kernel copy = NULL;
cl_kernel copy1 = NULL;

cl_platform_id platform_id = NULL;

cl_uint ret_num_devices;
cl_uint ret_num_platforms;

cl_int ret;

cl_float *r;
cl_float *g;
cl_float *b;

cl_event event;
cl_ulong start;
cl_ulong end;


ppm_t ipgm; // input image structure
ppm_t opgm; // output image structure
FILE *fp;
const char fileName[] = "newmedian2.cl";
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

b = (float *)malloc(n * n * sizeof(cl_float));
g = (float *)malloc(n * n * sizeof(cl_float));
r = (float *)malloc(n * n * sizeof(cl_float));

/* Get platform/device*/
ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id,&ret_num_devices);

/* Create OpenCL context */
context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

/* Create Command queue */
queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);

/* Create Buffer Objects */
xmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, n*n*sizeof(cl_float), NULL,&ret);
rmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, n*n*sizeof(cl_float), NULL,&ret);
wmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, n*n*sizeof(cl_float), NULL,&ret);
robj = clCreateBuffer(context, CL_MEM_READ_WRITE, n*n*sizeof(cl_float), NULL,&ret);
gobj = clCreateBuffer(context, CL_MEM_READ_WRITE, n*n*sizeof(cl_float), NULL,&ret);
bobj = clCreateBuffer(context, CL_MEM_READ_WRITE, n*n*sizeof(cl_float), NULL,&ret);
cobj = clCreateBuffer(context, CL_MEM_READ_WRITE, n*n*3*sizeof(cl_char), NULL,&ret);
oobj = clCreateBuffer(context, CL_MEM_READ_WRITE, n*n*3*sizeof(cl_float), NULL,&ret);


/* Transfer data to memory buffer */
ret = clEnqueueWriteBuffer(queue, xmobj, CL_TRUE, 0, n*n*sizeof(cl_float), NULL, 0,NULL, &event);
ret = clEnqueueWriteBuffer(queue, rmobj, CL_TRUE, 0, n*n*sizeof(cl_float), NULL, 0,NULL, &event);
ret = clEnqueueWriteBuffer(queue, wmobj, CL_TRUE, 0, n*n*sizeof(cl_float), NULL, 0,NULL, &event);
ret = clEnqueueWriteBuffer(queue, cobj, CL_TRUE, 0, n*n*3*sizeof(cl_char), ipgm.buf, 0,NULL, &event);
ret = clEnqueueWriteBuffer(queue, oobj, CL_TRUE, 0, n*n*3*sizeof(cl_float), NULL, 0,NULL, &event);
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

hpfl = clCreateKernel(program, "medianFilter", &ret);
if(!ret==CL_SUCCESS)
printf("1 %d\n",ret);

copy = clCreateKernel(program, "assign", &ret);
if(!ret==CL_SUCCESS)
printf("1 %d\n",ret);

copy1 = clCreateKernel(program, "assign1", &ret);
if(!ret==CL_SUCCESS)
printf("1 %d\n",ret);

//Assigning RGB values from image object
ret = clSetKernelArg(copy, 0, sizeof(cl_mem), (void *)&robj);
ret = clSetKernelArg(copy, 1, sizeof(cl_mem), (void *)&gobj);
ret = clSetKernelArg(copy, 2, sizeof(cl_mem), (void *)&bobj);
ret = clSetKernelArg(copy, 3, sizeof(cl_mem), (void *)&cobj);
ret = clSetKernelArg(copy, 4, sizeof(cl_int), (void *)&n);
setWorkSize(gws, lws,3*n, n);
ret = clEnqueueNDRangeKernel(queue, copy, 2, NULL, gws, lws, 0, NULL, &event);
if(!ret==CL_SUCCESS)
printf("133 %d\n",ret);

clWaitForEvents(1, &event);

/* Apply median filter on Red */

ret = clSetKernelArg(hpfl, 0, sizeof(cl_mem), (void *)&robj);	
ret = clSetKernelArg(hpfl, 1, sizeof(cl_mem), (void *)&wmobj);	
ret = clSetKernelArg(hpfl, 2, sizeof(cl_int), (void *)&n);
setWorkSize(gws, lws,n, n);
ret = clEnqueueNDRangeKernel(queue, hpfl, 2, NULL, gws, lws, 0, NULL,&event);
if(!ret==CL_SUCCESS)
printf("1 %d\n",ret);
clWaitForEvents(1, &event);

/* Apply median filter on Green */

ret = clSetKernelArg(hpfl, 0, sizeof(cl_mem), (void *)&gobj);
ret = clSetKernelArg(hpfl, 1, sizeof(cl_mem), (void *)&rmobj);
ret = clSetKernelArg(hpfl, 2, sizeof(cl_int), (void *)&n);
setWorkSize(gws, lws,n, n);
ret = clEnqueueNDRangeKernel(queue, hpfl, 2, NULL, gws, lws, 0, NULL, &event);
if(!ret==CL_SUCCESS)
printf("111 %d\n",ret);
clWaitForEvents(1, &event);



/* Apply median filter on Blue */

ret = clSetKernelArg(hpfl, 0, sizeof(cl_mem), (void *)&bobj);
ret = clSetKernelArg(hpfl, 1, sizeof(cl_mem), (void *)&xmobj);
ret = clSetKernelArg(hpfl, 2, sizeof(cl_int), (void *)&n);
setWorkSize(gws, lws,n, n);
ret = clEnqueueNDRangeKernel(queue, hpfl, 2, NULL, gws, lws, 0, NULL, &event);
if(!ret==CL_SUCCESS)
printf("1 %d\n",ret);
clWaitForEvents(1, &event);



/* Assign RGB values to Output image object */

ret = clSetKernelArg(copy1, 0, sizeof(cl_mem), (void *)&wmobj);
ret = clSetKernelArg(copy1, 1, sizeof(cl_mem), (void *)&rmobj);
ret = clSetKernelArg(copy1, 2, sizeof(cl_mem), (void *)&xmobj);
ret = clSetKernelArg(copy1, 3, sizeof(cl_mem), (void *)&oobj);
ret = clSetKernelArg(copy1, 4, sizeof(cl_int), (void *)&n);
setWorkSize(gws, lws,n, n);
ret = clEnqueueNDRangeKernel(queue, copy1, 2, NULL, gws, lws, 0, NULL, &event);
if(!ret==CL_SUCCESS)
printf("134 %d\n",ret);

clWaitForEvents(1, &event);

float *ampd;
ampd = (float*)malloc(n*n*3*sizeof(float));

/* Read image into ampd */
ret = clEnqueueReadBuffer(queue, oobj, CL_TRUE, 0, n*n*3*sizeof(cl_float), ampd, 0,NULL, NULL);
if(!ret==CL_SUCCESS)
printf("15 %d\n",ret);

clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
printf(" memory buffer write: %10.5f [ms]¥n", (end - start)/1000000.0);


opgm.width = n;
opgm.height = n;
opgm.buf = (unsigned char*)malloc(n *3* n * sizeof(unsigned char));

/* Write into output.ppm */
normalizeF2PPM(&opgm, ampd);
free(ampd);

/* Finalizations*/
ret = clFlush(queue);
ret = clFinish(queue);
ret = clReleaseKernel(hpfl);

ret = clReleaseProgram(program);
ret = clReleaseMemObject(xmobj);
ret = clReleaseMemObject(rmobj);
ret = clReleaseMemObject(wmobj);
ret = clReleaseMemObject(robj);
ret = clReleaseMemObject(gobj);
ret = clReleaseMemObject(bobj);

ret = clReleaseCommandQueue(queue);
ret = clReleaseContext(context);

destroyPPM(&ipgm);
destroyPPM(&opgm);

free(source_str);

return 0;
}

