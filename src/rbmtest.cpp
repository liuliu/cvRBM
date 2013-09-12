#include <iostream>
#include "cv.h"
#include "highgui.h"
#include "mlrbm.h"

int main()
{
	CvRBM* rbm = new CvRBM( CvRBMParams( 5, 28*28, 1000, 500, 250, 30 ) );
	FILE* test = fopen( "t10k-images.idx3-ubyte", "r" );
	char bigendian[4];
	int a;
	fread( bigendian, 1, 4, test );
	a = (bigendian[0]<<24)+(bigendian[1]<<16)+(bigendian[2]<<8)+bigendian[3];
	unsigned int count;
	fread( bigendian, 1, 4, test );
	count = (bigendian[0]<<24)+(bigendian[1]<<16)+(bigendian[2]<<8)+bigendian[3];
	unsigned int width, height;
	fread( bigendian, 1, 4, test );
	width = (bigendian[0]<<24)+(bigendian[1]<<16)+(bigendian[2]<<8)+bigendian[3];
	fread( bigendian, 1, 4, test );
	height = (bigendian[0]<<24)+(bigendian[1]<<16)+(bigendian[2]<<8)+bigendian[3];
	printf("%d %d %d %d\n", a, count, width, height);
	unsigned char* image = (unsigned char*)malloc(width*height);
	CvMat* train_data = cvCreateMat( count/200, width*height, CV_64FC1 );
	double* data_vec = train_data->data.db;
	for ( int i = 0; i < count/200; i++ )
	{
		fread( image, 1, width*height, test );
		for ( int j = 0; j < width*height; j++ )
		{
			*data_vec = (double)image[j]/255.;
			data_vec++;
		}
	}
	rbm->train( train_data, NULL );
	rbm->save( "rbmdata" );
	CvMat* testcase = cvCreateMat( 1, width*height, CV_64FC1 );
	double* test_vec = testcase->data.db;
	data_vec = train_data->data.db+20*width*height;
	for ( int i = 0; i < width*height; i++ )
	{
		*test_vec = *data_vec;
		test_vec++;
		data_vec++;
	}
	IplImage* img = cvCreateImage( cvSize(width, height), 8, 1 );
	unsigned char* img_vec = (unsigned char*)img->imageData;
	test_vec = testcase->data.db;
	for ( int i = 0; i < width*height; i++ )
	{
		*img_vec = (unsigned char)(*test_vec*255.0);
		test_vec++;
		img_vec++;
	}
	cvNamedWindow( "original", 1 );
	cvShowImage( "original", img );
	CvMat* abst = rbm->abstract( testcase );
	CvMat* result = rbm->reconstruct( abst );
	IplImage* rimg = cvCreateImage( cvSize(width, height), 8, 1 );
	unsigned char* rimg_vec = (unsigned char*)rimg->imageData;
	double* result_vec = result->data.db;
	for ( int i = 0; i < width*height; i++ )
	{
		*rimg_vec = (unsigned char)(*result_vec*255.0);
		result_vec++;
		rimg_vec++;
	}
	cvNamedWindow( "result", 1 );
	cvShowImage( "result", rimg );
	cvWaitKey(0);
	cvDestroyWindow( "original" );
	cvDestroyWindow( "result" );
}
