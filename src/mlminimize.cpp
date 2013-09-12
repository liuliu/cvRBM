#include "mlminimize.h"

CvMinimizeParams cvMinimizeParams( int _NUM, double _INT, double _EXT, int _ITMAX, double _RATIO, double _SIG, double _RHO )
{
	return CvMinimizeParams( _NUM, _INT, _EXT, _ITMAX, _RATIO, _SIG, _RHO );
}

CvMat** CvMinimize::minimize( CvMat** X,
			      int length,
			      double red )
{
	CvMat **df0, **df3, **dF0;
	CvMat **s;
	CvMat **X0, **Xn;

	CV_FUNCNAME( "CvMinimize::minimize" );

	__BEGIN__;

	dF0 = (CvMat**)cvAlloc( params.NUM*sizeof(CvMat*) );
	df0 = (CvMat**)cvAlloc( params.NUM*sizeof(CvMat*) );
	df3 = (CvMat**)cvAlloc( params.NUM*sizeof(CvMat*) );
	s = (CvMat**)cvAlloc( params.NUM*sizeof(CvMat*) );
	X0 = (CvMat**)cvAlloc( params.NUM*sizeof(CvMat*) );
	Xn = (CvMat**)cvAlloc( params.NUM*sizeof(CvMat*) );
	for ( int k = 0; k < params.NUM; k++ )
	{
		dF0[k] = cvCreateMat( X[k]->rows, X[k]->cols, CV_64FC1 );
		df0[k] = cvCreateMat( X[k]->rows, X[k]->cols, CV_64FC1 );
		df3[k] = cvCreateMat( X[k]->rows, X[k]->cols, CV_64FC1 );
		s[k] = cvCreateMat( X[k]->rows, X[k]->cols, CV_64FC1 );
		X0[k] = cvCreateMat( X[k]->rows, X[k]->cols, CV_64FC1 );
		Xn[k] = cvCreateMat( X[k]->rows, X[k]->cols, CV_64FC1 );

		cvZero( dF0[k] );
		cvZero( df0[k] );
		cvZero( df3[k] );
		cvZero( s[k] );
		cvZero( X0[k] );
		cvZero( Xn[k] );
	}

	double F0 = 0, f0 = 0, f1 = 0, f2 = 0, f3 = 0, f4 = 0;
	double x1 = 0, x2 = 0, x3 = 0, x4 = 0;
	double d0 = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0;
	double A = 0, B = 0;
	bool ls_failed = 0;

	function( X, f0, df0 );
	d0 = 0;
	for ( int k = 0; k < params.NUM; k++ )
	{
		cvSubRS( df0[k], cvScalar(0), s[k] );
		d0 += -cvDotProduct( s[k], s[k] );
	}
	x3 = red/(1.-d0);
	int i = 0;
	int l = ( length > 0 ) ? length : -length;
	int ls = ( length > 0 ) ? 1 : 0;
	int eh = ( length > 0 ) ? 0 : 1;
	while ( i < l )
	{
		i+=ls;
		for ( int k = 0; k < params.NUM; k++ )
		{
			cvCopy( X[k], X0[k] );
			cvCopy( df0[k], dF0[k] );
		}
		F0 = f0;
		int m = ( length > 0 ) ? params.ITMAX : l-i;
		if ( params.ITMAX < m )
			m = params.ITMAX;
		for ( ; ; )
		{
			x2 = 0;
			f2 = f0;
			d2 = d0;
			f3 = f0;
			for ( int k = 0; k < params.NUM; k++ )
				cvCopy( df0[k], df3[k] );
			while ( m > 0 )
			{
				m--;
				i+=eh;
				for ( int k = 0; k < params.NUM; k++ )
					cvScaleAdd( s[k], cvScalar(x3), X[k], Xn[k] );
				if (function( Xn, f3, df3 ))
					break;
				else
					x3 = (x2+x3)*.5;
			}
			if ( f3 < F0 )
			{
				for ( int k = 0; k < params.NUM; k++ )
				{
					cvCopy( Xn[k], X0[k] );
					cvCopy( df3[k], dF0[k] );
				}
				F0 = f3;
			}
			d3 = 0;
			for ( int k = 0; k < params.NUM; k++ )
				d3 += cvDotProduct( df3[k], s[k] );
			if ( (d3 > params.SIG*d0)||(f3 > f0+x3*params.RHO*d0)||(m <= 0) )
				break;
			x1 = x2;
			f1 = f2;
			d1 = d2;
			x2 = x3;
			f2 = f3;
			d2 = d3;
			A = 6.*(f1-f2)+3.*(d2+d1)*(x2-x1);
			B = 3.*(f2-f1)-(2.*d1+d2)*(x2-x1);
			x3 = B*B-A*d1*(x2-x1);
			if ( x3 < 0 )
				x3 = x2*params.EXT;
			else {
				x3 = x1-d1*(x2-x1)*(x2-x1)/(B+sqrt(x3));
				if ( x3 < 0 )
					x3 = x2*params.EXT;
				else {
					if ( x3 > x2*params.EXT )
						x3 = x2*params.EXT;
					else if ( x3 < x2+params.INT*(x2-x1) )
						x3 = x2+params.INT*(x2-x1);
				}
			}
		}
		while ( ((fabs(d3) > -params.SIG*d0)||(f3 > f0+x3*params.RHO*d0 ))&&(m > 0) )
		{
			if ( (d3 > 1e-8)||(f3 > f0+x3*params.RHO*d0) )
			{
				x4 = x3;
				f4 = f3;
				d4 = d3;
			} else {
				x2 = x3;
				f2 = f3;
				d2 = d3;
			}
			if ( f4 > f0 )
				x3 = x2-(.5*d2*(x4-x2)*(x4-x2))/(f4-f2-d2*(x4-x2));
			else {
				A = 6.*(f2-f4)/(x4-x2)+3.*(d4+d2);
				B = 3.*(f4-f2)-(2.*d2+d4)*(x4-x2);
				x3 = B*B-A*d2*(x4-x2)*(x4-x2);
				if ( x3 < 0 )
					x3 = (x2+x4)*.5;
				else
					x3 = x2+(sqrt(x3)-B)/A;
			}
			if ( x3 > x4-params.INT*(x4-x2) )
				x3 = x4-params.INT*(x4-x2);
			if ( x3 < x2+params.INT*(x4-x2) )
				x3 = x2+params.INT*(x4-x2);
			for ( int k = 0; k < params.NUM; k++ )
				cvScaleAdd( s[k], cvScalar(x3), X[k], Xn[k] );
			function( Xn, f3, df3 );
			if ( f3 < F0 )
			{
				for ( int k = 0; k < params.NUM; k++ )
				{
					cvCopy( Xn[k], X0[k] );
					cvCopy( df3[k], dF0[k] );
				}
				F0 = f3;
			}
			m--;
			i+=eh;
			d3 = 0;
			for ( int k = 0; k < params.NUM; k++ )
				d3 += cvDotProduct( df3[k], s[k] );
		}
		if ( (fabs(d3) < -params.SIG*d0)&&(f3 < f0+x3*params.RHO*d0) )
		{
			double df0_df3 = 0;
			double df3_df3 = 0;
			double df0_df0 = 0;
			for ( int k = 0; k < params.NUM; k++ )
			{
				df0_df3 += cvDotProduct( df0[k], df3[k] );
				df3_df3 += cvDotProduct( df3[k], df3[k] );
				df0_df0 += cvDotProduct( df0[k], df0[k] );
			}
			CvScalar scalar = cvScalar((df0_df3-df3_df3)/df0_df0);
			for ( int k = 0; k < params.NUM; k++ )
			{
				cvCopy( Xn[k], X[k] );
				cvScaleAdd( s[k], scalar, df3[k], s[k] );
				cvSubRS( s[k], cvScalar(0), s[k] );
				cvCopy( df3[k], df0[k] );
			}
			d3 = d0;
			d0 = 0;
			for ( int k = 0; k < params.NUM; k++ )
				d0 += cvDotProduct( df0[k], s[k] );
			if ( d0 > 0 )
			{
				d0 = 0;
				for ( int k = 0; k < params.NUM; k++ )
				{
					cvSubRS( df0[k], cvScalar(0), s[k] );
					d0 += -cvDotProduct( s[k], s[k] );
				}
			}
			x3 = x3*(params.RATIO < d3/(d0-1e-8) ? params.RATIO : d3/(d0-1e-8));
			ls_failed = 0;
		} else {
			for ( int k = 0; k < params.NUM; k++ )
			{
				cvCopy( X0[k], X[k] );
				cvCopy( dF0[k], df0[k] );
			}
			f0 = F0;
			if ( ls_failed )
				break;
			d0 = 0;
			for ( int k = 0; k < params.NUM; k++ )
			{
				cvSubRS( df0[k], cvScalar(0), s[k] );
				d0 += -cvDotProduct( s[k], s[k] );
			}
			x3 = red/(1.-d0);
			ls_failed = 1;
		}
	}

	for ( int k = 0; k < params.NUM; k++ )
	{
		cvReleaseMat( &s[k] );
		cvReleaseMat( &X0[k] );
		cvReleaseMat( &Xn[k] );
		cvReleaseMat( &dF0[k] );
		cvReleaseMat( &df0[k] );
		cvReleaseMat( &df3[k] );
	}
	cvFree( &s );
	cvFree( &X0 );
	cvFree( &Xn );
	cvFree( &dF0 );
	cvFree( &df0 );
	cvFree( &df3 );

	__END__;

	return X;
}
