#ifndef GUARD_mlrbm_h
#define GUARD_mlrbm_h

#include <ml.h>
#include <cstdarg>

#define CV_TYPE_NAME_ML_RBM      "opencv-ml-restricted-boltzmann-machine"

struct CV_EXPORTS CvRBMParams
{
	double epsilonw;
	double epsilonvb;
	double epsilonhb;
	double weightcost;
	double initialmomentum;
	double finalmomentum;
	int maxepoch;
	int tuneepoch;
	int maxbatches;
	int tunebatches;
	int linesearch;
	int deep;
	int* size;
	CvRBMParams( int _deep, ... )
	: epsilonw(0.1), epsilonvb(0.1), epsilonhb(0.1), weightcost(0.0002), initialmomentum(0.5), finalmomentum(0.9), maxepoch(50), tuneepoch(200), maxbatches(100), tunebatches(1000), linesearch(3), deep(_deep)
	{
		va_list vl;
		va_start( vl, _deep );
		size = (int*)cvAlloc( _deep*sizeof(int) );
		for ( int i = 0; i < _deep; i++ )
			size[i] = va_arg( vl, int );
		va_end( vl );
	}
	CvRBMParams( double _epsilonw, double _epsilonvb, double _epsilonhb, double _weightcost, double _initialmomentum, double _finalmomentum, int _maxepoch, int _tuneepoch, int _maxbatches, int _tunebatches, int _linesearch, int _deep, ... )
	: epsilonw(_epsilonw), epsilonvb(_epsilonvb), epsilonhb(_epsilonhb), weightcost(_weightcost), initialmomentum(_initialmomentum), finalmomentum(_finalmomentum), maxepoch(_maxepoch), tuneepoch(_tuneepoch), maxbatches(_maxbatches), tunebatches(_tunebatches), linesearch(_linesearch), deep(_deep)
	{
		va_list vl;
		va_start( vl, _deep );
		size = (int*)cvAlloc( _deep*sizeof(int) );
		for ( int i = 0; i < _deep; i++ )
			size[i] = va_arg( vl, int );
		va_end( vl );
	}
};

class CV_EXPORTS CvRBM : public CvStatModel
{
	private:
		CvRBMParams params;
		CvMat** cache;
		CvMat** weight;
		bool trainRBM( CvMat*& vishid, CvMat*& visbiases, CvMat*& hidbiases, CvMat** train_data, const int numtotal, const int numdims, const int numhid );
		bool fineTune( CvMat** train_data, const int numtotal );
	public:
		CvRBM( CvRBMParams _params );
		virtual ~CvRBM();
		virtual bool train( const CvMat* _train_data, const CvMat* responses );
		virtual void clear();
		virtual void write( CvFileStorage* fs, const char* name );
		virtual void read( CvFileStorage* fs, CvFileNode* root_node );
		CvMat* abstract( CvMat* sample, CvMat* response = 0, CvMat** w = 0, CvMat** probs = 0 );
		CvMat* reconstruct( CvMat* sample, CvMat* response = 0, CvMat** w = 0, CvMat** probs = 0 );
		int deep();
		int size( int i );
};

#endif
