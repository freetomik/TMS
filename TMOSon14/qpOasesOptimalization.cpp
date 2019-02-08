

#include <fstream>

#include "qpOasesOptimization.h"

std::vector<cv::Mat> optimizationWithOases(qpOASES::int_t height, qpOASES::int_t width,
	cv::Mat detailImage, cv::Mat weight1, cv::Mat weight2,
	std::vector<cv::Mat> baseChannels, std::vector<cv::Mat> detailChannels)
{

	USING_NAMESPACE_QPOASES
	/*
		Deklarace proměnných
	*/
	int_t w = width;
	int_t h = height;
	std::vector<cv::Mat> sAndT;
    cv::Mat s;
	s = cv::Mat::ones(height, width, CV_32F);
	cv::Mat t;
    t = cv::Mat::zeros(height, width, CV_32F);
	// float r1 = 200/255.0, r2 = 500/255.0;
	float r1 = 200, r2 = 500;

	/*
		Optimization
	*/

	int_t mainSize = h*w*2; // počet parametrů
	/*
		Creating D matrix
	*/
	// long memoryMB = (mainSize*mainSize*4) / (1024*1024);
	// std::cout << "allocating " << memoryMB << " MB" << '\n';
	long memoryKB = (mainSize*mainSize*4) / (1024);
	std::cout << "allocating " << memoryKB << " KB" << '\n';
	// TODO don't allocate this
	cv::Mat DMatrix = cv::Mat::zeros(mainSize, mainSize, CV_32F);

	// TODO construct sparse matrix from: Row indices, Indices to first entry of columns and Vector of entries 
	
	// (TODO copy borders (one pixel))

	for (int j = 0; j < h; j++) {
		for (int i = 0; i < w; i++) {

			float r1w1 = r1*weight1.at<float>(j, i);
			float r2w2 = r2*weight2.at<float>(j, i);

			float D2 = ((float)(pow(detailImage.at<float>(j, i), 2)));

			// central differences: f(x+1) - f(x-1)

			// s part
// -D_i^2*s_i^2 + ...
			
			// diagonal values
			DMatrix.at<float>(w*j + i, w*j + i) += -D2 *2; // D_i^2*s_i^2
			// coefficients of quadratic terms are doubled for qp

// ... + r_1*w_1 * (s_{i-1}^2 - 2*s_{i-1}*s_{i+1} + s_{i+1}^2  +  s_{j-1}^2 - 2*s_{j-1}*s_{j+1} + s_{j+1}^2) + ...

			// 0.2 is some reduction factor
			if (i > 0) {
				DMatrix.at<float>(w*j + i-1, w*j + i-1) += r1w1 *2 ; // r_1*w_1*s_{i-1}^2
				// coefficients of quadratic terms are doubled for qp
			}
			if (i > 0 && i < w-1) {
				DMatrix.at<float>(w*j + i+1, w*j + i-1) += -2*r1w1 *0.2; // -2*r_1*w_1*s_{i-1}*s_{i+1}
				DMatrix.at<float>(w*j + i-1, w*j + i+1) += -2*r1w1 *0.2; // -2*r_1*w_1*s_{i-1}*s_{i+1}
			}
			if (i < w-1) {
				DMatrix.at<float>(w*j + i+1, w*j + i+1) += r1w1 *2 ; // r_1*w_1*s_{i-1}^2
				// coefficients of quadratic terms are doubled for qp
			}
			
			if (j > 0) {
				DMatrix.at<float>(w*(j-1) + i, w*(j-1) + i) += r1w1 *2 ; // r_1*w_1*s_{j-1}^2
				// coefficients of quadratic terms are doubled for qp
			}
			if (j > 0 && j < h-1) {
				DMatrix.at<float>(w*(j+1) + i, w*(j-1) + i) += -2*r1w1 *0.2; // -2*r_1*w_1*s_{j-1}*s_{j+1}
				DMatrix.at<float>(w*(j-1) + i, w*(j+1) + i) += -2*r1w1 *0.2; // -2*r_1*w_1*s_{j-1}*s_{j+1}
			}
			if (j < h-1) {
				DMatrix.at<float>(w*(j+1) + i, w*(j+1) + i) += r1w1 *2 ; // r_1*w_1*s_{j+1}^2
				// coefficients of quadratic terms are doubled for qp
			}
			
			// t part
// ... + r_2*w_2 * (t_{i-1}^2 - 2*t_{i-1}*t_{i+1} + t_{i+1}^2  +  t_{j-1}^2 - 2*t_{j-1}*t_{j+1} + t_{j+1}^2)
			
			if (i > 0) {
				DMatrix.at<float>(w*h + w*j + i-1, w*h + w*j + i-1) += r2w2 *2 *0.2; // r_2*w_2*t_{i-1}^2
				// coefficients of quadratic terms are doubled for qp
			}
			if (i > 0 && i < w-1) {
				DMatrix.at<float>(w*h + w*j + i+1, w*h + w*j + i-1) += -2*r2w2 *0.2; // -2*r_2*w_2*t_{i-1}*t_{i+1}
				DMatrix.at<float>(w*h + w*j + i-1, w*h + w*j + i+1) += -2*r2w2 *0.2; // -2*r_2*w_2*t_{i-1}*t_{i+1}
			}
			if (i < w-1) {
				DMatrix.at<float>(w*h + w*j + i+1, w*h + w*j + i+1) += r2w2 *2 *0.2; // r_2*w_2*t_{i-1}^2
				// coefficients of quadratic terms are doubled for qp
			}
			
			if (j > 0) {
				DMatrix.at<float>(w*h + w*(j-1) + i, w*h + w*(j-1) + i) += r2w2 *2 *0.2; // r_2*w_2*t_{j-1}^2
				// coefficients of quadratic terms are doubled for qp
			}
			if (j > 0 && j < h-1) {
				DMatrix.at<float>(w*h + w*(j+1) + i, w*h + w*(j-1) + i) += -2*r2w2 *0.2; // -2*r_2*w_2*t_{j-1}*t_{j+1}
				DMatrix.at<float>(w*h + w*(j-1) + i, w*h + w*(j+1) + i) += -2*r2w2 *0.2; // -2*r_2*w_2*t_{j-1}*t_{j+1}
			}
			if (j < h-1) {
				DMatrix.at<float>(w*h + w*(j+1) + i, w*h + w*(j+1) + i) += r2w2 *2 *0.2; // r_2*w_2*t_{j+1}^2
				// coefficients of quadratic terms are doubled for qp
			}

		}
	}
	
	
	std::cout << "after creating DMatrix" << '\n';
	
	// long H_size = mainSize*mainSize * sizeof(real_t) / (1024*1024);
	// std::cout << "allocating " << H_size << "MB" << '\n';
	long H_size = mainSize*mainSize * sizeof(real_t) / (1024);
	std::cout << "allocating " << H_size << "KB" << '\n';
	// FIXME use floats for less memory consumption
	real_t *H = new real_t[mainSize*mainSize];
	// SymSparseMat *H = SymSparseMat(mainSize, mainSize);
	// SymSparseMat *H = new SymSparseMat();
	real_t *A = new real_t[h*w*mainSize*3]; // potřebujeme pro každý pixel jednu podmínku
	// SparseMatrix *A = new SparseMatrix();
	real_t *g = new real_t[mainSize]; // to je ponecháno nule, nemá žádný vliv
	real_t *lb = new real_t[mainSize]; // dolní hranice (počet hodnot jako počet pixelů)
	real_t *ub = new real_t[mainSize]; // horní hranice (počet hodnot jako počet pixelů)
	real_t *lbA = new real_t[h*w*3]; // dolní hranice A (počet pixelů, pro každý pixel jedna dolní hranice)
	real_t *ubA = new real_t[h*w*3]; // horní hranice A (počet pixelů, pro každý pixel jedna horní hranice)

	// DMatrix = 2*DMatrix; // to už je v prvcích uděláno
	/*
		Vkládám do Hessovy matice
	*/
	// DMatrix = DMatrix*2;
	for (int j = 0; j < mainSize; j++) {
		for (int i = 0; i < mainSize; i++) {
			H[j * mainSize + i] =  DMatrix.at<float>(j, i);
		}
	}
	DMatrix.release();
	/*
		Nastavuji dolní + horní hranice + nulové hodnoty g
	*/
	for (int a = 0; a < mainSize; a++) {
		g[a] = 0;		

		if (a < mainSize/2) {
			lb[a] = -5000;
			ub[a] = 5000;
		} else {
			lb[a] = -2000;
			ub[a] = 2000;
		}
		
	}

	/*
		Přednastavené A-čkovské hranice (pro sichr)
	*/
	for (int i = 0; i < h*w*mainSize; i++) {
		A[i] = 0.0;
	}
	
  	int counterA = 0; // počitadlo podmínek, protože se chci posouvat ob jeden řádek, tak tam je mainSiye*counterA
  	for (int j = 0; j < h; j++) {
		for (int i = 0; i < w; i++) {
			/*
				Nastavení A (podmínka)
			*/
			/*
				Nastavení odpovídá -B_i <= t_i + s_iD_i <= 1 - B_i
				posunuji se vždy o řádek, proto mainSize*counterA
				u hranic se to nedělá kvůli tomu, že pro každou podmínku je jenom jedna hranice dolní/horní
			*/
			A[mainSize*counterA + j * w + i] = (float)(detailChannels[0].at<float>(j, i)/255.0);
			A[mainSize*counterA + w*h + j * w + i] = 1.0;
			lbA[j*w + i] = -(float)(baseChannels[0].at<float>(j, i)/255.0); // dolní hranice A
			ubA[j*w + i] = 1.0 - (float)(baseChannels[0].at<float>(j, i)/255.0); // horní hranice A
			counterA++;

			A[mainSize*counterA + j * w + i] = (float)(detailChannels[1].at<float>(j, i)/255.0);
			A[mainSize*counterA + w*h + j * w + i] = 1.0;
			// FIXME 2x + + ?
			lbA[1*h*w + + j*w + i] = -(float)(baseChannels[1].at<float>(j, i)/255.0); // dolní hranice A
			ubA[1*h*w + + j*w + i] = 1.0 - (float)(baseChannels[1].at<float>(j, i)/255.0); // horní hranice A
			counterA++;

			A[mainSize*counterA + j * w + i] = (float)(detailChannels[2].at<float>(j, i)/255.0);
			A[mainSize*counterA + w*h + j * w + i] = 1.0;
			lbA[2*h*w + j*w + i] = -(float)(baseChannels[2].at<float>(j, i)/255.0); // dolní hranice A
			ubA[2*h*w + j*w + i] = 1.0 - (float)(baseChannels[2].at<float>(j, i)/255.0); // horní hranice A
			counterA++;
		}
	}


	QProblem example( mainSize, h*w*3); // počet parametrů, počet podmínek

	Options options;
	example.setOptions( options );
	int_t nWSR = 50000;
	// example.init(H,g,A,lb, ub,lbA,ubA, nWSR);
	// TODO check return value
	example.init(H,g,A,lb, ub,lbA,ubA, nWSR);


	
	real_t *xOpt = new real_t[mainSize];
	example.getPrimalSolution( xOpt ); 

	/*
		Vložení do vytvořených matic
	*/
	int lineCounter = 0;
	int parametersCounterS = 0;
	int parametersCounterT = 0;
	for (int i = 0; i < mainSize; i++) {
			/*
				Zjištění, jestli je to parametr s či t, s parametry jsou jako první
			*/
			if (lineCounter < w*h) {
				s.at<float>(parametersCounterS / w, parametersCounterS % w) = xOpt[parametersCounterS];
				parametersCounterS++;				 			    
			} else {				
				t.at<float>(parametersCounterT / w, parametersCounterT % w) = xOpt[w*h + parametersCounterT];
				parametersCounterT++; 
			}
			lineCounter++;
	}

	/*
		Uvolním paměť
	*/

	delete H;
	delete xOpt;
	delete A;
	delete lb;
	delete ub;
	delete g;
	delete lbA;
	delete ubA;

	sAndT.push_back(s);
	sAndT.push_back(t);
	return sAndT;
}
