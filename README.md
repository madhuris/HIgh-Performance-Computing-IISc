# HIgh-Performance-Computing-IISc
Parallelizing applications using OpenCL

>>To run 

	1. gcc -c -Wall -I /Path-to-include-directory-in-SDK/ filename.c -o objectfile.o

	2. gcc -lm -L /Path-to-lib-directory-in-SDK/ -l OpenCL objectfile.o -o executable

>>Output images

	Gaussian and Median 

	Output : output.ppm

	JPEG compression

	Output : x.jpg

	Canny Edge Detection

	Output : output.pgm

>>To change images

	Gaussian,Median and Canny Edge Detection : Change filename in the line readPPM()

>> Points to note
	
	Use square images only

		
