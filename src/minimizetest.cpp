#include "mlminimize.h"
#include <iostream>
/*
class CV_EXPORTS Rosenbrock : public CvMinimize
{
	private:
		virtual bool function( CvMat** x, double& f, CvMat** df )
		{
			double* x_vec = x[0]->data.db;
			f = 0;
			for ( int i = 0; i < 1; i++ )
				f += 100*(x_vec[i+1]-x_vec[i]*x_vec[i])*(x_vec[i+1]-x_vec[i]*x_vec[i])+(1-x_vec[i])*(1-x_vec[i]);
			double* df_vec = df[0]->data.db;
			cvZero( df[0] );
			for ( int i = 0; i < 1; i++ )
				df_vec[i] = -400*x_vec[i]*(x_vec[i+1]-x_vec[i]*x_vec[i])-2*(1-x_vec[i]);
			for ( int i = 1; i < 2; i++ )
				df_vec[i] += 200*(x_vec[i]-x_vec[i-1]*x_vec[i-1]);
			return 1;
		}
	public:
		Rosenbrock( CvMinimizeParams* params )
		: CvMinimize(params) {}
};
*/
struct Rosenbrock
{
	bool operator()( CvMat** x, double& f, CvMat** df, const void* userdata )
	{
		int* count = (int*)userdata;
		(*count)++;
		double* x_vec = x[0]->data.db;
		f = 0;
		for ( int i = 0; i < 1; i++ )
			f += 100*(x_vec[i+1]-x_vec[i]*x_vec[i])*(x_vec[i+1]-x_vec[i]*x_vec[i])+(1-x_vec[i])*(1-x_vec[i]);
		double* df_vec = df[0]->data.db;
		cvZero( df[0] );
		for ( int i = 0; i < 1; i++ )
			df_vec[i] = -400*x_vec[i]*(x_vec[i+1]-x_vec[i]*x_vec[i])-2*(1-x_vec[i]);
		for ( int i = 1; i < 2; i++ )
			df_vec[i] += 200*(x_vec[i]-x_vec[i-1]*x_vec[i-1]);
		return 1;
	}
};

int main()
{
	//Rosenbrock* rb = new Rosenbrock( params );
	CvMat* X = cvCreateMat( 1, 2, CV_64FC1 );
	cvZero( X );
	int data = 0;
	CvMat** result = cvMinimize<Rosenbrock>( &X, CvMinimizeParams(), 25, &data );
	printf("count: %d\n", data);
	//CvMat** result = rb->minimize( &X, 25 );
	for ( int k = 0; k < 2; k++ )
		printf("%f\n", result[0]->data.db[k]);
}
