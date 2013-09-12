#include "mlrbm.h"
#include "mlminimize.h"
#include <iostream>

CvRBM::CvRBM( CvRBMParams _params )
: params(_params), cache(0), weight(0)
{
}

CvRBM::~CvRBM()
{
	clear();
}

bool CvRBM::trainRBM( CvMat*& vishid, CvMat*& visbiases,
					  CvMat*& hidbiases, CvMat** train_data,
					  const int numtotal, const int numdims, const int numhid )
{
	bool result = false;
	CvMat* responses = 0;

	double* data_vec;

	static CvRNG rng_state = cvRNG(0xffffffff);

	vishid = cvCreateMat( numhid, numdims, CV_64FC1 );
	hidbiases = cvCreateMat( 1, numhid, CV_64FC1 );
	visbiases = cvCreateMat( 1, numdims, CV_64FC1 );
	cvRandArr( &rng_state, vishid, CV_RAND_NORMAL, cvRealScalar(0), cvRealScalar(.1) );
	cvZero( hidbiases );
	cvZero( visbiases );
	double* vishid_vec;
	double* vishid_sl;
	double* hidbiases_vec;
	double* visbiases_vec;

	CvMat* poshidprobs;
	CvMat* neghidprobs;
	CvMat* posprods;
	CvMat* negprods;
	CvMat* vishidinc;
	CvMat* hidbiasinc;
	CvMat* visbiasinc;

	poshidprobs = cvCreateMat( 1, numhid, CV_64FC1 );
	neghidprobs = cvCreateMat( 1, numhid, CV_64FC1 );
	posprods = cvCreateMat( numhid, numdims, CV_64FC1 );
	negprods = cvCreateMat( numhid, numdims, CV_64FC1 );
	vishidinc = cvCreateMat( numhid, numdims, CV_64FC1 );
	hidbiasinc = cvCreateMat( 1, numhid, CV_64FC1 );
	visbiasinc = cvCreateMat( 1, numdims, CV_64FC1 );

	cvZero( poshidprobs );
	cvZero( neghidprobs );
	cvZero( posprods );
	cvZero( negprods );
	cvZero( vishidinc );
	cvZero( hidbiasinc );
	cvZero( visbiasinc );
	double* poshidprobs_vec;
	double* neghidprobs_vec;
	double* posprods_vec;
	double* negprods_vec;
	double* vishidinc_vec;
	double* hidbiasinc_vec;
	double* visbiasinc_vec;

	CvMat* poshidstates;
	CvMat* negdata;

	poshidstates = cvCreateMat( 1, numhid, CV_64FC1 );
	negdata = cvCreateMat( 1, numdims, CV_64FC1 );
	double* poshidstates_vec;
	double* negdata_vec;

	CvMat* poshidact;
	CvMat* posvisact;
	CvMat* neghidact;
	CvMat* negvisact;
	poshidact = cvCreateMat( 1, numhid, CV_64FC1 );
	posvisact = cvCreateMat( 1, numdims, CV_64FC1 );
	neghidact = cvCreateMat( 1, numhid, CV_64FC1 );
	negvisact = cvCreateMat( 1, numdims, CV_64FC1 );
	double* poshidact_vec;
	double* posvisact_vec;
	double* neghidact_vec;
	double* negvisact_vec;

	double _maxepoch = 1./(params.maxepoch-1.);
	int numcases = params.maxbatches;

	CvMat* dataidx;
	dataidx = cvCreateMat( 1, numtotal, CV_32SC1 );
	int* dataidx_vec = dataidx->data.i;
	for ( int i = 0; i < numtotal; i++ )
		dataidx_vec[i] = i;

	for ( int epoch = 0; epoch < params.maxepoch; epoch++ )
	{
		double momentum = params.initialmomentum+(params.finalmomentum-params.initialmomentum)*epoch*_maxepoch;
		double _momentum = 1.-momentum;
		cvRandShuffle( dataidx, &rng_state );
		dataidx_vec = dataidx->data.i;
		double errsum = 0;

		for ( int batches = 0; batches < numtotal; batches+=numcases )
		{
			int numlower = batches;
			int numupper = ( numlower+numcases < numtotal ) ? numlower+numcases : numtotal;
			double _numcases = 1./(double)(numupper-numlower);
			double err = 0;

			cvZero( posprods );
			cvZero( negprods );

			cvZero( poshidact );
			cvZero( posvisact );
			cvZero( neghidact );
			cvZero( negvisact );

			for ( int s = numlower; s < numupper; s++ )
			{
				CvMat*& data = train_data[dataidx_vec[s]];
				cvRandArr( &rng_state, poshidstates, CV_RAND_UNI, cvRealScalar(0), cvRealScalar(1.) );
				/* START POSITIVE PHASE */
				poshidstates_vec = poshidstates->data.db;
				posprods_vec = posprods->data.db;
				poshidprobs_vec = poshidprobs->data.db;
				hidbiases_vec = hidbiases->data.db;
				vishid_vec = vishid->data.db;
				for ( int i = 0; i < numhid; i++ )
				{
					data_vec = data->data.db;
					double probsv = *hidbiases_vec;
					for ( int j = 0; j < numdims; j++ )
					{
						probsv += *data_vec * *vishid_vec;
						data_vec++;
						vishid_vec++;
					}
					*poshidprobs_vec = 1./(1.+exp(-probsv));
					data_vec = data->data.db;
					for ( int j = 0; j < numdims; j++ )
					{
						*posprods_vec += *data_vec * *poshidprobs_vec;
						data_vec++;
						posprods_vec++;
					}
					*poshidstates_vec = ( *poshidprobs_vec > *poshidstates_vec ) ? 1. : 0;
					poshidstates_vec++;
					hidbiases_vec++;
					poshidprobs_vec++;
				}
	
				cvAdd( poshidact, poshidprobs, poshidact );
				cvAdd( posvisact, data, posvisact );
				/* END OF POSITIVE PHASE */
		
				/* START NEGATIVE PHASE */
				negdata_vec = negdata->data.db;
				visbiases_vec = visbiases->data.db;
				vishid_vec = vishid->data.db;
				for ( int i = 0; i < numdims; i++ )
				{
					double negdatav = *visbiases_vec;
					vishid_sl = vishid_vec;
					poshidstates_vec = poshidstates->data.db;
					for ( int j = 0; j < numhid; j++ )
					{
						negdatav += *vishid_sl * *poshidstates_vec;
						poshidstates_vec++;
						vishid_sl += vishid->cols;
					}
					*negdata_vec = 1./(1.+exp(-negdatav));
					negdata_vec++;
					vishid_vec++;
					visbiases_vec++;
				}
				neghidprobs_vec = neghidprobs->data.db;
				negprods_vec = negprods->data.db;
				vishid_vec = vishid->data.db;
				hidbiases_vec = hidbiases->data.db;
				for ( int i = 0; i < numhid; i++ )
				{
					negdata_vec = negdata->data.db;
					double probsv = *hidbiases_vec;
					for (int j = 0; j < numdims; j++)
					{
						probsv += *negdata_vec * *vishid_vec;
						negdata_vec++;
						vishid_vec++;
					}
					*neghidprobs_vec = 1./(1.+exp(-probsv));
					negdata_vec = negdata->data.db;
					for ( int j = 0; j < numdims; j++ )
					{
						*negprods_vec += *negdata_vec * *neghidprobs_vec;
						negprods_vec++;
						negdata_vec++;
					}
					neghidprobs_vec++;
					hidbiases_vec++;
				}
				cvAdd( neghidact, neghidprobs, neghidact );
				cvAdd( negvisact, negdata, negvisact );
				/* END OF NEGATIVE PHASE */

				data_vec = data->data.db;
				negdata_vec = negdata->data.db;
				for ( int i = 0; i < numdims; i++ )
				{
					err+=(*data_vec - *negdata_vec)*(*data_vec - *negdata_vec);
					data_vec++;
					negdata_vec++;
				}
			}
			errsum+=err;

			/* UPDATE WEIGHTS AND BIASES */
			vishidinc_vec = vishidinc->data.db;
			vishid_vec = vishid->data.db;
			posprods_vec = posprods->data.db;
			negprods_vec = negprods->data.db;
			for ( int i = 0; i < numhid; i++ )
				for ( int j = 0; j < numdims; j++ )
				{
					*vishidinc_vec = momentum * *vishidinc_vec+params.epsilonw*((*posprods_vec - *negprods_vec)*_numcases-params.weightcost * *vishid_vec);
					*vishid_vec += *vishidinc_vec;
					vishidinc_vec++;
					vishid_vec++;
					posprods_vec++;
					negprods_vec++;
				}

			visbiasinc_vec = visbiasinc->data.db;
			visbiases_vec = visbiases->data.db;
			posvisact_vec = posvisact->data.db;
			negvisact_vec = negvisact->data.db;
			for ( int i = 0; i < numdims; i++ )
			{
				*visbiasinc_vec = momentum * *visbiasinc_vec+params.epsilonvb*_numcases*(*posvisact_vec - *negvisact_vec);
				*visbiases_vec += *visbiasinc_vec;
				visbiasinc_vec++;
				visbiases_vec++;
				posvisact_vec++;
				negvisact_vec++;
			}

			hidbiasinc_vec = hidbiasinc->data.db;
			hidbiases_vec = hidbiases->data.db;
			poshidact_vec = poshidact->data.db;
			neghidact_vec = neghidact->data.db;
			for ( int i = 0; i < numhid; i++ )
			{
				*hidbiasinc_vec = momentum * *hidbiasinc_vec+params.epsilonhb*_numcases*(*poshidact_vec - *neghidact_vec);
				*hidbiases_vec += *hidbiasinc_vec;
				hidbiasinc_vec++;
				hidbiases_vec++;
				poshidact_vec++;
				neghidact_vec++;
			}
			/* END OF UPDATES */
		}
		printf("error in epoch %d : %f\n" , epoch, errsum);
	}

	cvReleaseMat( &dataidx );
	
	cvReleaseMat( &poshidprobs );
	cvReleaseMat( &neghidprobs );
	cvReleaseMat( &posprods );
	cvReleaseMat( &negprods );
	cvReleaseMat( &vishidinc );
	cvReleaseMat( &hidbiasinc );
	cvReleaseMat( &visbiasinc );

	cvReleaseMat( &poshidstates );
	cvReleaseMat( &negdata );

	cvReleaseMat( &poshidact );
	cvReleaseMat( &posvisact );
	cvReleaseMat( &neghidact );
	cvReleaseMat( &negvisact );

	return result;
}

class CV_EXPORTS CvRBMBackPropagate : public CvMinimize
{
	private:
		CvRBM* rbm;
		CvMat** train_data;
		CvMat* dataidx;
		int lower;
		int upper;

		CvMat** ix;
		CvMat** wprobs;
		CvMat* hidprobs;
		CvMat* negdata;

		bool function( CvMat** X, double& f, CvMat** df )
		{
			int* dataidx_vec = dataidx->data.i;
			f = 0;
			double _numcases = 1./(upper-lower);
			double* data_vec;
			double* negdata_vec;
			double* vishid_vec;
			double* vishid_sl;
			int step;
			double* jx_vec;
			double* ix_vec;
			double* df_vec;
			double* wprobs_vec;
			for ( int i = 0; i < (rbm->deep()-1)*4; i++ )
				cvZero( df[i] );
			CvMat** ix_ptr;
			CvMat** df_ptr;
			CvMat** wprobs_ptr;
			CvMat** X_ptr;
			for ( int s = lower; s < upper; s++ )
			{
				CvMat*& data = train_data[dataidx_vec[s]];
				rbm->abstract( data, hidprobs, X, wprobs );
				rbm->reconstruct( hidprobs, negdata, X, wprobs+rbm->deep()-1 );
				wprobs[0] = data;
				wprobs[rbm->deep()-1] = hidprobs;
				wprobs[rbm->deep()*2-2] = negdata;

				data_vec = data->data.db;
				negdata_vec = negdata->data.db;
				for ( int i = 0; i < data->cols; i++ )
				{
					f += *data_vec * log(*negdata_vec+1e-8)+(1. - *data_vec)*log(1.+1e-8 - *negdata_vec);
					data_vec++;
					negdata_vec++;
				}

				ix_ptr = ix+rbm->deep()*2-2;
				df_ptr = df+(rbm->deep()-1)*4-1;
				cvSub( negdata, data, *ix_ptr );
				cvConvertScale( *ix_ptr, *ix_ptr, _numcases );
				cvAdd( *df_ptr, *ix_ptr, *df_ptr );

				ix_ptr--;
				df_ptr--;
				wprobs_ptr = wprobs+rbm->deep()*2-3;
				cvGEMM( ix_ptr[1], *wprobs_ptr, 1, *df_ptr , 1, *df_ptr, CV_GEMM_A_T );

				X_ptr = X+rbm->deep()*4-6;
				for ( int k = rbm->deep()*2-3; k > 0; k-- )
				{
					cvMatMul( ix_ptr[1], *X_ptr, *ix_ptr );
					ix_vec = (*ix_ptr)->data.db;
					wprobs_vec = (*wprobs_ptr)->data.db;
					for ( int i = 0; i < (*ix_ptr)->cols; i++ )
					{
						*ix_vec *= *wprobs_vec * (1. - *wprobs_vec);
						ix_vec++;
						wprobs_vec++;
					}
					df_ptr--;
					cvAdd( *df_ptr, *ix_ptr, *df_ptr );

					ix_ptr--;
					df_ptr--;
					wprobs_ptr--;
					cvGEMM( ix_ptr[1], *wprobs_ptr, 1, *df_ptr , 1, *df_ptr, CV_GEMM_A_T );
					X_ptr -= 2;
				}
			}

			f = -_numcases*f;
			printf("cross entropy error: %f\n", f);
			return 1;
		}
	public:
		CvRBMBackPropagate( CvRBM* _rbm, CvMinimizeParams _param )
		: CvMinimize(_param), rbm(_rbm)
		{
			ix = (CvMat**)cvAlloc( (rbm->deep()*2-1)*sizeof(CvMat*) );
			wprobs = (CvMat**)cvAlloc( (rbm->deep()*2-1)*sizeof(CvMat*) );
			hidprobs = cvCreateMat( 1, rbm->size(rbm->deep()-1), CV_64FC1 );
			negdata = cvCreateMat( 1, rbm->size(0), CV_64FC1 );
			for ( int i = 1; i < rbm->deep()-1; i++ )
			{
				wprobs[i] = cvCreateMat( 1, rbm->size(i), CV_64FC1 );
				wprobs[i+rbm->deep()-1] = cvCreateMat( 1, rbm->size(rbm->deep()-1-i), CV_64FC1 );
				ix[i] = cvCreateMat( 1, rbm->size(i), CV_64FC1 );
				ix[i+rbm->deep()-1] = cvCreateMat( 1, rbm->size(rbm->deep()-1-i), CV_64FC1 );
			}
			ix[0] = cvCreateMat( 1, rbm->size(0), CV_64FC1 );
			ix[rbm->deep()-1] = cvCreateMat( 1, rbm->size(rbm->deep()-1), CV_64FC1 );
			ix[rbm->deep()*2-2] = cvCreateMat( 1, rbm->size(0), CV_64FC1 );
		}
		CvMat** train( CvMat** X, CvMat** _train_data, CvMat* _dataidx, int _lower, int _upper, int length, double red = 1. )
		{
			train_data = _train_data;
			dataidx = _dataidx;
			lower = _lower;
			upper = _upper;
			return minimize( X, length, red );
		}
		virtual ~CvRBMBackPropagate()
		{
			cvReleaseMat( &hidprobs );
			cvReleaseMat( &negdata );
			for ( int i = 1; i < rbm->deep()-1; i++ )
			{
				cvReleaseMat( &wprobs[i] );
				cvReleaseMat( &wprobs[i+rbm->deep()-1] );
				cvReleaseMat( &ix[i] );
				cvReleaseMat( &ix[i+rbm->deep()-1] );
			}
			cvFree( &wprobs );
			cvReleaseMat( &ix[0] );
			cvReleaseMat( &ix[rbm->deep()-1] );
			cvReleaseMat( &ix[rbm->deep()*2-2] );
			cvFree( &ix );
		}
};

bool CvRBM::fineTune( CvMat** train_data, const int numtotal )
{
	bool result = false;
	CvMat* responses = 0;

	double* data_vec;
	double* negdata_vec;

	static CvRNG rng_state = cvRNG(0xffffffff);
	
	double _maxepoch = 1./(params.maxepoch-1.);
	int numcases = params.tunebatches;

	CvMat* dataidx;
	dataidx = cvCreateMat( 1, numtotal, CV_32SC1 );
	int* dataidx_vec = dataidx->data.i;
	for ( int i = 0; i < numtotal; i++ )
		dataidx_vec[i] = i;

	CvMat** wprobs;
	CvMat* hidprobs;
	CvMat* negdata;

	wprobs = (CvMat**)cvAlloc( (params.deep*2-1)*sizeof(CvMat*) );
	hidprobs = cvCreateMat( 1, params.size[params.deep-1], CV_64FC1 );
	negdata = cvCreateMat( 1, params.size[0], CV_64FC1 );
	for ( int i = 1; i < params.deep-1; i++ )
	{
		wprobs[i] = cvCreateMat( 1, params.size[i], CV_64FC1 );
		wprobs[i+params.deep-1] = cvCreateMat( 1, params.size[params.deep-1-i], CV_64FC1 );
	}

	CvRBMBackPropagate backprop( this, CvMinimizeParams( (params.deep-1)*4, .1, 3., 20, 10., .1, .05 ) );

	for ( int epoch = 0; epoch < params.tuneepoch; epoch++ )
	{
		cvRandShuffle( dataidx, &rng_state );
		dataidx_vec = dataidx->data.i;
		double errsum = 0;

		for ( int batches = 0; batches < numtotal; batches+=numcases )
		{
			int numlower = batches;
			int numupper = ( numlower+numcases < numtotal ) ? numlower+numcases : numtotal;
			double _numcases = 1./(double)(numupper-numlower);
			double err = 0;
			for ( int s = numlower; s < numupper; s++ )
			{
				CvMat*& data = train_data[dataidx_vec[s]];
				abstract( data, hidprobs, weight, wprobs );
				reconstruct( hidprobs, negdata, weight, wprobs+params.deep-1 );
				data_vec = data->data.db;
				negdata_vec = negdata->data.db;
				for ( int i = 0; i < data->cols; i++ )
				{
					err+=(*data_vec - *negdata_vec)*(*data_vec - *negdata_vec);
					data_vec++;
					negdata_vec++;
				}
			}
			printf("error in epoch %d, batch %d before tune: %f\n", epoch, batches, err);
			errsum += err;
			backprop.train( weight, train_data, dataidx, numlower, numupper, params.linesearch );
		}
		printf("error in epoch %d : %f\n" , epoch, errsum);
	}
	
	cvReleaseMat( &hidprobs );
	cvReleaseMat( &negdata );
	for ( int i = 1; i < params.deep-1; i++ )
	{
		cvReleaseMat( &wprobs[i] );
		cvReleaseMat( &wprobs[i+params.deep-1] );
	}
	cvFree( &wprobs );
	cvReleaseMat( &dataidx );
}

int CvRBM::deep()
{
	return params.deep;
}

int CvRBM::size( int i )
{
	return params.size[i];
}

bool CvRBM::train( const CvMat* raw_data, const CvMat* responses )
{
	CV_FUNCNAME( "CvRBM::train" );

	__BEGIN__;

	cache = (CvMat**)cvAlloc( (params.deep*2-1)*sizeof(CvMat*) );
	weight = (CvMat**)cvAlloc( (params.deep-1)*4*sizeof(CvMat*) );
	CvMat*** train_cache = (CvMat***)cvAlloc( sizeof(CvMat**) );
	for ( int s = 0; s < params.deep; s++ )
		train_cache[s] = (CvMat**)cvAlloc( raw_data->rows*sizeof(CvMat*) );
	for ( int s = 1; s < params.deep; s++ )
		for ( int i = 0; i < raw_data->rows; i++ )
			train_cache[s][i] = cvCreateMat( 1, params.size[s], CV_64FC1 );
	CvMat** train_data = (CvMat**)cvAlloc( raw_data->rows*sizeof(CvMat*) );
	for ( int i = 0;  i < raw_data->rows; i++ )
	{
		train_data[i] = cvCreateMatHeader( 1, raw_data->cols, CV_64FC1 );
		cvSetData( train_data[i], raw_data->data.db+i*raw_data->cols, raw_data->step );
		train_cache[0][i] = train_data[i];
	}

	CvMat** vishid = weight;
	CvMat** hidbiases = weight+1;
	CvMat** hidvis = weight+((params.deep-1)*4-2);
	CvMat** visbiases = weight+((params.deep-1)*4-1);
	for ( int s = 0; s < params.deep-1; s++ )
	{
		trainRBM( *vishid, *visbiases, *hidbiases, train_cache[s], raw_data->rows, params.size[s], params.size[s+1] );
		for ( int k = 0; k < raw_data->rows; k++ )
		{
			double* poshidprobs_vec = train_cache[s+1][k]->data.db;
			double* vishid_vec = (*vishid)->data.db;
			double* hidbiases_vec = (*hidbiases)->data.db;
			for ( int i = 0; i < (*vishid)->rows; i++ )
			{
				double* data_vec = train_cache[s][k]->data.db;
				double probsv = *hidbiases_vec;
				for ( int j = 0; j < (*vishid)->cols; j++ )
				{
					probsv += *data_vec * *vishid_vec;
					data_vec++;
					vishid_vec++;
				}
				*poshidprobs_vec = 1./(1.+exp(-probsv));
				poshidprobs_vec++;
				hidbiases_vec++;
			}
		}
		*hidvis = cvCreateMat( (*vishid)->cols, (*vishid)->rows, CV_64FC1 );
		cvTranspose( *vishid, *hidvis );
		vishid += 2;
		hidbiases += 2;
		hidvis -= 2;
		visbiases -= 2;
	}

	fineTune( train_data, raw_data->rows );

	for ( int s = 1; s < params.deep-1; s++ )
	{
		cache[s] = cvCreateMat( 1, params.size[s], CV_64FC1 );
		cache[s+params.deep-1] = cvCreateMat( 1, params.size[params.deep-1-s], CV_64FC1 );
	}

	for ( int s = 1; s < params.deep; s++ )
		for ( int i = 0; i < raw_data->rows; i++ )
			cvReleaseMat( &train_cache[s][i] );
	for ( int s = 0; s < params.deep; s++ )
		cvFree( &train_cache[s] );
	cvFree( &train_cache );

	__END__;

	return 1;
}

CvMat* CvRBM::abstract( CvMat* sample, CvMat* response, CvMat** w, CvMat** probs )
{
	CV_FUNCNAME( "CvRBM::abstract" );

	__BEGIN__;

	if ( w == 0 )
		w = weight;
	if ( probs == 0 )
		probs = cache;
	probs[0] = sample;
	probs[params.deep-1] = ( response == 0 ) ? cvCreateMat( 1, params.size[params.deep-1], CV_64FC1 ) : response;
	CvMat** vishid = w;
	CvMat** hidbiases = w+1;
	for ( int s = 0; s < params.deep-1; s++ )
	{
		double* poshidprobs_vec = probs[s+1]->data.db;
		double* vishid_vec = (*vishid)->data.db;
		double* hidbiases_vec = (*hidbiases)->data.db;
		for ( int i = 0; i < (*vishid)->rows; i++ )
		{
			double* data_vec = probs[s]->data.db;
			double probsv = *hidbiases_vec;
			for ( int j = 0; j < (*vishid)->cols; j++ )
			{
				probsv += *data_vec * *vishid_vec;
				data_vec++;
				vishid_vec++;
			}
			*poshidprobs_vec = 1./(1.+exp(-probsv));
			poshidprobs_vec++;
			hidbiases_vec++;
		}
		vishid += 2;
		hidbiases += 2;
	}

	__END__;

	return probs[params.deep-1];
}

CvMat* CvRBM::reconstruct( CvMat* sample, CvMat* response, CvMat** w, CvMat** probs )
{
	CV_FUNCNAME( "CvRBM::reconstruct" );

	__BEGIN__;

	if ( w == 0 )
		w = weight;
	if ( probs == 0 )
		probs = cache+params.deep-1;
	probs[0] = sample;
	probs[params.deep-1] = ( response == 0 ) ? cvCreateMat( 1, params.size[0], CV_64FC1 ) : response;
	CvMat** hidvis = w+2*(params.deep-1);
	CvMat** visbiases = w+(2*(params.deep-1)+1);
	for ( int s = 0; s < params.deep-1; s++ )
	{
		double* negdata_vec = probs[s+1]->data.db;
		double* hidvis_vec = (*hidvis)->data.db;
		double* visbiases_vec = (*visbiases)->data.db;
		for ( int i = 0; i < (*hidvis)->rows; i++ )
		{
			double negdatav = *visbiases_vec;
			double* poshidstates_vec = probs[s]->data.db;
			for ( int j = 0; j < (*hidvis)->cols; j++ )
			{
				negdatav += *hidvis_vec * *poshidstates_vec;
				poshidstates_vec++;
				hidvis_vec++;
			}
			*negdata_vec = 1./(1.+exp(-negdatav));
			negdata_vec++;
			visbiases_vec++;
		}
		
		hidvis += 2;
		visbiases += 2;
	}

	__END__;

	return probs[params.deep-1];
}

void CvRBM::clear()
{
	CV_FUNCNAME( "CvRBM::clear" );

	__BEGIN__;

	if ( cache != 0 )
		for ( int i = 1; i < params.deep-1; i++ )
		{
			cvReleaseMat( &cache[i] );
			cvReleaseMat( &cache[i+params.deep-1] );
		}
	cvFree( &cache );
	if ( weight != 0 )
		for ( int i = 0; i < (params.deep-1)*4; i++ )
			cvReleaseMat( &weight[i] );
	cvFree( &weight );

	__END__;
}

void CvRBM::write( CvFileStorage* fs, const char* name )
{
	CV_FUNCNAME( "CvRBM::write" );

	__BEGIN__;

	int i;

	cvStartWriteStruct( fs, name, CV_NODE_MAP, CV_TYPE_NAME_ML_RBM );

	CV_CALL( cvWriteInt( fs, "params_deep", params.deep ) );

	CV_CALL( cvStartWriteStruct( fs, "size", CV_NODE_SEQ ) );
	for ( i = 0; i < params.deep; i++ )
		CV_CALL( cvWriteInt( fs, NULL, params.size[i] ) );
	CV_CALL( cvEndWriteStruct( fs ) );

	CV_CALL( cvStartWriteStruct( fs, "weight", CV_NODE_SEQ ) );
	for ( i = 0; i< (params.deep-1)*4; i++ )
		CV_CALL( cvWrite( fs, NULL, weight[i] ) );
	CV_CALL( cvEndWriteStruct( fs ) );

	cvEndWriteStruct( fs );

	__END__;
}

void CvRBM::read( CvFileStorage* fs, CvFileNode* root_node )
{
	CV_FUNCNAME( "CvRBM::read" );

	__BEGIN__;

	int i;
	CvFileNode* node;
	CvSeq* seq;
	CvSeqReader reader;

	clear();

	CV_CALL( params.deep = cvReadIntByName( fs, root_node, "params_deep", -1 ) );
	if( params.deep <= 0 )
		CV_ERROR( CV_StsParseError, "No \"params_deep\" in RBM classifier" );

	cache = (CvMat**)cvAlloc( (params.deep*2-1)*sizeof(CvMat*) );
	weight = (CvMat**)cvAlloc( (params.deep-1)*4*sizeof(CvMat*) );

	cvFree( &params.size );
	params.size = (int*)cvAlloc( params.deep*sizeof(int) );

	CV_CALL( node = cvGetFileNodeByName( fs, root_node, "size" ));
	seq = node->data.seq;
	if( !CV_NODE_IS_SEQ(node->tag) || seq->total != params.deep )
		CV_ERROR( CV_StsBadArg, "" );
	CV_CALL( cvStartReadSeq( seq, &reader, 0 ));
	for ( i = 0; i < params.deep; i++ )
	{
		CV_CALL( params.size[i] = cvReadInt( (CvFileNode*)reader.ptr ));
		CV_NEXT_SEQ_ELEM( seq->elem_size, reader );
	}

	for ( i = 1; i < params.deep-1; i++ )
	{
		cache[i] = cvCreateMat( 1, params.size[i], CV_64FC1 );
		cache[i+params.deep-1] = cvCreateMat( 1, params.size[params.deep-1-i], CV_64FC1 );
	}

	CV_CALL( node = cvGetFileNodeByName( fs, root_node, "weight" ));
	seq = node->data.seq;
	if( !CV_NODE_IS_SEQ(node->tag) || seq->total != (params.deep-1)*4 )
		CV_ERROR( CV_StsBadArg, "" );
	CV_CALL( cvStartReadSeq( seq, &reader, 0 ));
	for ( i = 0; i< (params.deep-1)*4; i++ )
	{
		CV_CALL( weight[i] = (CvMat*)cvRead( fs, (CvFileNode*)reader.ptr ));
		CV_NEXT_SEQ_ELEM( seq->elem_size, reader );
	}

	__END__;
}
