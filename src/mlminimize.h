#ifndef GUARD_mlminimize_h
#define GUARD_mlminimize_h

#include <ml.h>

struct CV_EXPORTS CvMinimizeParams
{
	double INT;
	double EXT;
	int NUM;
	int ITMAX;
	double RATIO;
	double SIG;
	double RHO;
	CvMinimizeParams()
	: INT(.1), EXT(3.), NUM(1), ITMAX(20), RATIO(10.), SIG(.1), RHO(.05)
	{}
	CvMinimizeParams( int _NUM, double _INT, double _EXT, int _ITMAX, double _RATIO, double _SIG, double _RHO )
	: INT(_INT), EXT(_EXT), NUM(_NUM), ITMAX(_ITMAX), RATIO(_RATIO), SIG(_SIG), RHO(_RHO)
	{}
};

class CV_EXPORTS CvMinimize
{
	private:
		CvMinimizeParams params;
		virtual bool function( CvMat** x, double& f, CvMat** df ) = 0;
	public:
		CvMinimize( CvMinimizeParams _params )
		: params(_params)
		{}
		CvMat** minimize( CvMat** X, int length, double red = 1. );
};

template <class Func>
class CV_EXPORTS CvMinimizeTemplate : public CvMinimize
{
	private:
		const void* userdata;
		CvMinimizeParams* params;
		virtual bool function( CvMat** x, double& f, CvMat** df )
		{
			Func func;
			return func( x, f, df, userdata );
		}
	public:
		CvMinimizeTemplate( CvMinimizeParams _params, const void* _userdata )
		: CvMinimize(_params), userdata(_userdata) {}
};

CvMinimizeParams cvMinimizeParams( int _NUM, double _INT = .1, double _EXT = 3., int _ITMAX = 20, double _RATIO = 10., double _SIG = .1, double _RHO = .05 );

template <class Func>
CvMat** cvMinimize( CvMat** X, CvMinimizeParams params, int length, const void* userdata = NULL, double red = 1. )
{
	return CvMinimizeTemplate<Func>(params, userdata).minimize( X, length, red );
}

#endif
