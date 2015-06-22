#include <iostream>
#include <vector>
#include "mex.h"


using namespace std;


typedef struct {
	double y;
	double x;
} PointF;


typedef struct {
	int y;
	int x;
} Point;


typedef struct {
	PointF pts_a;
	PointF pts_b;
} FeatureF;


typedef struct {
	Point pts_a;
	Point pts_b;
} Feature;



inline double rand_double() {
	return (double)rand() / RAND_MAX;
}

inline int f2i(double num, int max_val) {
	int val = (int)(num + 0.5);
	val = val > 0 ? val : 0;
	val = val < max_val ? val : max_val;
	return val;
}

inline int sub2ind(int y, int x, int height) {
	return x * height + y;
}

void create_features(vector<FeatureF>& features, int n_features) {
	for (int i = 0; i < n_features; i++) {
		FeatureF f;
		f.pts_a.y = rand_double();
		f.pts_a.x = rand_double();
		f.pts_b.y = rand_double();
		f.pts_b.x = rand_double();
		features.push_back(f);
	}
}

void feature_f2i(const vector<FeatureF>& features_f, vector<Feature>& features, int height, int width) {
	for (unsigned int i = 0; i < features_f.size(); i++) {
		FeatureF feat_f = features_f[i];
		Feature feat;
		feat.pts_a.y = f2i(feat_f.pts_a.y * height, height - 1);
		feat.pts_a.x = f2i(feat_f.pts_a.x * width, width - 1);
		feat.pts_b.y = f2i(feat_f.pts_b.y * height, height - 1);
		feat.pts_b.x = f2i(feat_f.pts_b.x * width, width - 1);
		features.push_back(feat);
	}
}



class Fern {

public:
	unsigned int n_features;
	unsigned int n_classes;
	unsigned int n_indices;

	unsigned int width;
	unsigned int height;

	double u;					// priority

	vector<vector<double>> probs;		// log of probrabilities
	vector<vector<int>> n_in_c;			// number of samples in class c that make feature i equals 1

	vector<int> sum_n_in_c;				// number of samples in class c

	vector<FeatureF> features_f;
	vector<Feature> features;
	
	Fern(unsigned int _n_features, unsigned int _n_classes) 
		: n_features(_n_features), n_classes(_n_classes) {

		u = 1.0;

		n_indices = 1;
		for (unsigned int i = 0; i < n_features; i++) {
			n_indices <<= 1;
		}
		
		for (unsigned int i = 0; i < n_classes; i++) {
			vector<int> n_in_c_i(n_indices, 0);
			vector<double> probs_i(n_indices, 0.0);
			n_in_c.push_back(n_in_c_i);
			probs.push_back(probs_i);
		}
		
		sum_n_in_c = vector<int>(n_classes, 0);
		
		create_features(features_f, n_features);
	}

	void fit_img_size(unsigned int _height, unsigned int _width) {
		height = _height;
		width = _width;
		feature_f2i(features_f, features, height, width);
	}
	
	inline unsigned int get_prob_index(double* img) {
		unsigned int index = 0;
		for (unsigned int i = 0; i < n_features; i++) {
			index <<= 1;

			int ind_a = sub2ind(features[i].pts_a.y, features[i].pts_a.x, height);
			int ind_b = sub2ind(features[i].pts_b.y, features[i].pts_b.x, height);
			if (img[ind_a] < img[ind_b]) {
				index++;
			}
		}
		return index;
	}

	void count(double* img, int label) {
		unsigned int index = get_prob_index(img);
		n_in_c[label][index]++;
		sum_n_in_c[label]++;
	}

	void count_all(double* imgs, double* labels, int img_len, int n_samples) {
		double* img = imgs;
		double* p_label = labels;
		for (int i = 0; i < n_samples; i++) {
			count(img, (int)(*p_label));
			img = img + img_len;
			p_label++;
		}
	}

	void learn() {
		for (unsigned int i = 0; i < n_classes; i++) {
			double log_Z = log((double)sum_n_in_c[i] + u * n_indices);
			for (unsigned int j = 0; j < n_indices; j++) {
				probs[i][j] = log((double)n_in_c[i][j] + u) - log_Z;
			}
		}
	}

	// confs is the output which contains n_classes confidence scores predicted by the fern
	void predict(double* img, vector<double>& confs) {
		unsigned int index = get_prob_index(img);
		for (unsigned int i = 0; i < n_classes; i++) {
			confs.push_back(probs[i][index]);
		}
	}
};


class Ferns {
public:
	unsigned int n_ferns;
	unsigned int n_features;
	unsigned int n_classes;

	unsigned int height;
	unsigned int width;

	vector<Fern> ferns;

	Ferns(unsigned int _n_ferns = 30, unsigned int _n_features = 10, unsigned int _n_classes = 2) : 
		n_ferns(_n_ferns), n_features(_n_features), n_classes(_n_classes) {

		for (unsigned int i = 0; i < n_ferns; i++) {
			Fern fern(n_features, n_classes);
			ferns.push_back(fern);
		}
	}

	void fit_img_size(unsigned int _height, unsigned int _width) {
		height = _height;
		width = _width;

		for (unsigned int i = 0; i < n_ferns; i++) {
			ferns[i].fit_img_size(height, width);
		}
	}

	void learn(double* imgs, double* labels, int img_len, int n_samples) {
		for (unsigned int i = 0; i < n_ferns; i++) {
			ferns[i].count_all(imgs, labels, img_len, n_samples);
			ferns[i].learn();
		}
	}

	int predict(double* img, vector<double>& confs) {		
		confs = vector<double>(n_classes, 0);

		for (unsigned int i = 0; i < n_ferns; i++) {
			vector<double> fern_confs;
			ferns[i].predict(img, fern_confs);
			for (unsigned int j = 0; j < n_classes; j++) {
				confs[j] += fern_confs[j];
			}
		}

		int i_max = 0;
		double conf_max = confs[0];
		for (unsigned int i = 1; i < n_classes; i++) {
			if (confs[i] > conf_max) {
				conf_max = confs[i];
				i_max = i;
			}
		}

		return i_max;
	}
};



Ferns *ferns;
// input: options, height, width, imgs, labels
// options: learn - 1, predict - 2, clear - 0
// output: predictions
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {

	int opt = (int)mxGetScalar(prhs[0]);
	int height = (int)mxGetScalar(prhs[1]);
	int width = (int)mxGetScalar(prhs[2]);
	double* imgs = (double*)mxGetData(prhs[3]);
	int img_len = (int)mxGetM(prhs[3]);
	int n_samples = (int)mxGetN(prhs[3]);
	double* labels = NULL;
	if (nrhs > 4) {
		labels = mxGetPr(prhs[4]);
	}

	vector<double> confs;
	double *p;
	double *img;
	switch (opt) {
	case 1: 
		srand(0);
	
		ferns = new Ferns(40, 12, 10);
		ferns->fit_img_size(height, width);
		ferns->learn(imgs, labels, img_len, n_samples);
		break;
	case 2:
		plhs[0] = mxCreateDoubleMatrix(n_samples, 1, mxREAL);
		p = mxGetPr(plhs[0]);
		img = imgs;
		for (int i = 0; i < n_samples; i++) {
			int prediction = ferns->predict(img, confs);
			img += img_len;
			*p++ = prediction;
		}
		break;
	case 0:
		delete ferns;
		break;
	default:
		break;
	}
}