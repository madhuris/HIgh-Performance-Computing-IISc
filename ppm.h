#ifndef _PPM_H_
#define _PPM_H_

#include <math.h>
#include <string.h>

#define PPM_MAGIC "P6"

#ifdef _WIN32
#define STRTOK_R(ptr, del, saveptr) strtok_s(ptr, del, saveptr)
#else
#define STRTOK_R(ptr, del, saveptr) strtok_r(ptr, del, saveptr)
#endif

typedef struct _ppm_t {
int width;
int height;
unsigned char *buf;
} ppm_t;

int readPPM(ppm_t* ppm, const char* filename)
{
char *token, *pc, *saveptr;
char *buf;
size_t bufsize;
char del[] = " \t\n";
unsigned char *dot;

long begin, end;
int filesize;
int i, w, h, luma, pixs;


FILE* fp;
if ((fp = fopen(filename, "rb"))==NULL) {
fprintf(stderr, "Failed to open file\n");
return -1;
}

fseek(fp, 0, SEEK_SET);
begin = ftell(fp);
fseek(fp, 0, SEEK_END);
end = ftell(fp);
filesize = (int)(end - begin);

buf = (char*)malloc(filesize * sizeof(char));
fseek(fp, 0, SEEK_SET);
bufsize = fread(buf, filesize * sizeof(char), 1, fp);

fclose(fp);

token = (char *)STRTOK_R(buf, del, &saveptr);
if (strncmp(token, PPM_MAGIC, 2) != 0) {

return -1;}

token = (char *)STRTOK_R(NULL, del, &saveptr);
if (token[0] == '#' ) {
token = (char *)STRTOK_R(NULL, "\n", &saveptr);
token = (char *)STRTOK_R(NULL, del, &saveptr);
}

w = strtoul(token, &pc, 10);
token = (char *)STRTOK_R(NULL, del, &saveptr);
h = strtoul(token, &pc, 10);
token = (char *)STRTOK_R(NULL, del, &saveptr);
luma = strtoul(token, &pc, 10);

token = pc + 1;
pixs = w * h;

ppm->buf = (unsigned char *)malloc(pixs*3 * sizeof(unsigned char));

dot = ppm->buf;

for (i=0; i< pixs*3; i++, dot++) {

*dot = *token++;}
 
ppm->width = w;
ppm->height = h;

return 0;
}

int writePPM(ppm_t* ppm, const char* filename)
{
int i, w, h, pixs;
FILE* fp;
unsigned char* dot;

w = ppm->width;
h = ppm->height;
pixs = w * h;

if ((fp = fopen(filename, "wb+")) ==NULL) {
fprintf(stderr, "Failed to open file\n");
return -1;
}

fprintf (fp, "%s\n%d %d\n255\n", PPM_MAGIC, w, h);



dot = ppm->buf;

for(i=0;i<pixs*3;i++,dot++){
putc((unsigned char)(*dot), fp);
//printf("%d\n",(int)*dot);
}


/*for (i=0; i<pixs*3; i++, dot++) {

putc((unsigned char)*dot, fp);}

for(i=pixs;i<pixs+1000;i++){
putc((unsigned char)(255), fp);}
*/
fclose(fp);


return 0;
}



int normalizeF2PPM(ppm_t* ppm, float* x)
{
FILE* fp;
long int i;int j, w, h;

w = ppm->width;
h = ppm->height;

ppm->buf = (unsigned char*)malloc(w *3* h * sizeof(unsigned char));
char *filename="output.ppm";
if ((fp = fopen(filename, "wb+")) ==NULL) {
fprintf(stderr, "Failed to open file\n");
return -1;
}
fprintf (fp, "%s\n%d %d\n255\n", PPM_MAGIC, w, h);


for (i=0; i < h; i++) {
for (j=0; j < w*3; j++) {

ppm->buf[3*i*h+j] = (unsigned char)(x[3*i*h+j]);
//printf("%d\n",(int)ppm->buf[i*h+j]);





}}

for(i=0;i<w*h*3;i++)
{
putc((unsigned char)(ppm->buf[i]), fp);
//printf("%d\n",(char)(ppm->buf[i]));

}
return 0;
}

int destroyPPM(ppm_t* ppm)
{
if (ppm->buf) {

free(ppm->buf);}

return 0;
}

#endif /* _PPM_H_ */

