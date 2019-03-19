#include <malloc.h>
#include <omp.h>
#include <math.h>

//cp /mnt/c/Users/HP/Documents/iit_acad/col380/Lab2/col380_lab2_suite/
//g++ -fopenmp -lm lab2_io.c lab2_omp.c main_omp.c -o pca
//./pca testcase/testcase_2_2 90

// sort in decreasing order
void revSort(int N, double* W, double* W_n){
	int i, j; 
	for(i = 0; i < N; i++)
		W_n[i] = W[i];
	for (i = 0; i < N-1; i++){    
		for (j = 0; j < N-i-1; j++){  
	    	if (W_n[j] < W_n[j+1]){
	    		double temp = W_n[j];
	    		W_n[j] = W_n[j+1];
	    		W_n[j+1] = temp;
	    	}
	    }
	}
}

//get new M*N matrix;
double** newMatrix(int M, int N){
	double** W;
	W = (double **)malloc(M*sizeof(double*));
#pragma omp parallel for //num_threads(4)
	for(int i = 0; i < M; i++)
	    W[i]= (double*)malloc(N*sizeof(double));
	return W;
}

void freeMatrix(int M, int N, double** W){
	for(int i = 0; i < M; i++)
	    free(W[i]);
	free(W);
}

//initialise N*N matrix as identity
void identity(int N, double** W){
#pragma omp parallel for //num_threads(4)
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			W[i][j] = 0.0;
			if(i == j)
				W[i][j] = 1.0;
		}
	}
}

//initialise N*N matrix as identity
void identityLinear(int N, double* W){
#pragma omp parallel for //num_threads(4)
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			W[i*N + j] = 0.0;
			if(i == j)
				W[i*N + j] = 1.0;
		}
	}
}

//initialise M*N matrix as zero
void zeros(int M, int N, double** W){
#pragma omp parallel for //num_threads(4)
	for(int i = 0; i < M; i++){
		for(int j = 0; j < N; j++){
			W[i][j] = 0.0;
		}
	}
}

void zerosLinear(int M, int N, double* W){
#pragma omp parallel for //num_threads(4)
	for(int i = 0; i < M*N; i++){
		W[i] = 0.0;
	}
}

// copy M*N matrix
void copyM(int M, int N, double** W, double** W_n){
#pragma omp parallel for //num_threads(4)
	for(int i = 0; i < M; i++){
		for(int j = 0; j < N; j++){
			W_n[i][j] = W[i][j];
		}
	}
}

void copyMToLinear(int M, int N, double** W, double* W_n){
#pragma omp parallel for //num_threads(4)
	for(int i = 0; i < M; i++){
		for(int j = 0; j < N; j++){
			W_n[i*N + j] = W[i][j];
		}
	}
}

void copyLinearToM(int M, int N, double* W, double** W_n){
#pragma omp parallel for //num_threads(4)
	for(int i = 0; i < M; i++){
		for(int j = 0; j < N; j++){
			W_n[i][j] = W[i*N + j];
		}
	}
}

void copyLinearToLinear(int M, int N, double* W, double* W_n){
#pragma omp parallel for //num_threads(4)
	for(int i = 0; i < M*N; i++){
		W_n[i] = W[i];
	}
}

// print M*N matrix
void printM(int M, int N, double** W){
	for(int i = 0; i < M; i++){
		for(int j = 0; j < N; j++){
			printf("%f ", W[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

// transpose of M*N matrix W
void transpose(int M, int N, double** W, double** W_T){
#pragma omp parallel for //num_threads(4) schedule(static)
	for(int i = 0; i < M; i++){
		for(int j = 0; j < N; j++){
			W_T[j][i] = W[i][j];
		}
	}
}

// multiple M*N matrix with N*R matrix
void multiply(int M, int N, int R, double** W1, double** W2, double** W_M){
#pragma omp parallel for num_threads(4) schedule(static)
	for(int i = 0; i < M; i++){
		for(int j = 0; j < R; j++){
			double sum = 0.0;
			for(int k = 0; k < N; k++){
				sum += W1[i][k] * W2[k][j];
			}
			W_M[i][j] = sum;
		}
	}
}

// multiple M*N matrix with N*R matrix in 1D array
void multiplyLinear(int M, int N, int R, double* W1, double* W2, double* W_M){
#pragma omp parallel for //num_threads(4) schedule(static)
	for(int i = 0; i < M; i++){
		for(int j = 0; j < R; j++){
			double sum = 0.0;
			for(int k = 0; k < N; k++){
				sum += W1[i*N + k] * W2[k*R + j];
			}
			W_M[i*R + j] = sum;
		}
	}
}

double nabs(double a){
	if(a > 0)
		return a;
	return -a;
}

// QR factors of N*N matrix
// returns weird answers some of te times
void QRfactors(int N, double** W, double** Q, double** R){

	double** V = newMatrix(N, N);
	transpose(N, N, W, V);

	for(int i = 0; i < N; i++){
		R[i][i] = 0.0;
		double norm = 0.0;
		for(int k = 0; k < N; k++){
			norm += V[i][k]*V[i][k];
		}
		norm = sqrt(norm);
		R[i][i] = norm;
#pragma omp parallel for num_threads(2) schedule(static) shared(Q, V)
		for(int k = 0; k < N; k++)
			Q[k][i] = V[i][k]/norm;

		for(int j = i+1; j < N; j++){
			R[i][j] = 0.0;
			for(int k = 0; k < N; k++)
				R[i][j] += Q[k][i]*V[j][k];
			for(int k = 0; k < N; k++)
				V[j][k] = V[j][k] - R[i][j]*Q[k][i];
		}
	}
}

void QRfactorsLinear(int N, double* W, double* Q, double* R){

	double* V = (double*)malloc(N*N*sizeof(double));
	copyLinearToLinear(N, N, W, V);

	for(int i = 0; i < N; i++){
		R[i*N + i] = 0.0;
		double norm = 0.0;
		for(int k = 0; k < N; k++){
			norm += V[k*N + i]*V[k*N + i];
		}
		norm = sqrt(norm);
		R[i*N + i] = norm;
		for(int k = 0; k < N; k++)
			Q[k*N + i] = V[k*N + i]/R[i*N + i];
		for(int j = i+1; j < N; j++){
			R[i*N + j] = 0.0;
			for(int k = 0; k < N; k++)
				R[i*N + j] += Q[k*N + i]*V[k*N + j];
			for(int k = 0; k < N; k++)
				V[k*N + j] = V[k*N + j] - R[i*N + j]*Q[k*N + i];
		}
	}
}

// /*
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */
int iter_count;
void SVD(int M, int N, float* D, float** U, float** SIGMA, float** V_T)
{
	int i,j,k,t;

	double **D2;	//M*N matrix
	D2 = newMatrix(M, N);

	for(i = 0; i < M; i++){
		for(j = 0; j < N; j++)
			D2[i][j] = (double)D[i*N + j];
	}

	/* No need for taking transpose, replace U by V, V by U, sigma by sigma^T in the final output */

	double **D1;	//M*N matrix
	D1 = newMatrix(M, N);
	copyM(M, N, D2, D1);

	double **D_T;	//N*M matrix
	D_T = newMatrix(N, M);
	
	transpose(M, N, D1, D_T);

	double **DT_D;	//N*N matrix
	DT_D = newMatrix(N, N);

	multiply(N, M, N, D_T, D1, DT_D);	

	/********* QR Algorithm *********/
	/********************************/
	double **D_i = newMatrix(N, N);	//N*N matrix
	double **D_new = newMatrix(N, N);	//N*N matrix
	double **E_i = newMatrix(N, N);  //N*N matrix
	double **E_new = newMatrix(N, N);  //N*N matrix
	double **Q = newMatrix(N, N);  //N*N matrix
	double **R = newMatrix(N, N);  //N*N matrix
	
	copyM(N, N, DT_D, D_i);
	identity(N, E_i);
	zeros(N, N, Q);
	zeros(N, N, R);

	int CONG_LIMIT = 10000;
	double TOLERANCE = 0.0000000001;
	iter_count = 0;

	while(iter_count < CONG_LIMIT){

		iter_count++;
		QRfactors(N, D_i, Q, R);

		multiply(N, N, N, R, Q, D_new);
		multiply(N, N, N, E_i, Q, E_new);

		double maxDiff = 0.0;
		double diff;

		for(i = 0; i < N; i++){
			for(j = 0; j < N; j++){
				diff = E_new[i][j] - E_i[i][j];
				if(diff > maxDiff)
					maxDiff = diff;
				if(maxDiff > TOLERANCE)
					break;
			}
			if(maxDiff > TOLERANCE)
				break;
		}

		if(maxDiff < TOLERANCE)
			break;

		double** temp;
		temp = E_new;
		E_new = E_i;
		E_i = temp;

		temp = D_new;
		D_new = D_i;
		D_i = temp;
		//printf("%d\n", iter_count);
	}

	freeMatrix(N, N, E_new);
	freeMatrix(N, N, Q);
	freeMatrix(N, N, R);

	/**********QR Finish************/

	double* eigen_values = (double*)malloc(N*sizeof(double));
	double* sing_values = (double*)malloc(N*sizeof(double));
	double* sing_values_sorted = (double*)malloc(N*sizeof(double));

	for(i = 0; i < N; i++){
		eigen_values[i] = D_i[i][i];
		sing_values[i] = sqrt(eigen_values[i]);
	}

	revSort(N, sing_values, sing_values_sorted);

	double** sig = newMatrix(M, N);
	double** sig_inv = newMatrix(N, M);
	zeros(M, N, sig);
	zeros(N, M, sig_inv);

	int numEigenValues;
	if( N > M)
		numEigenValues = M;
	else
		numEigenValues = N;

	for(i = 0; i < numEigenValues; i++){
		sig[i][i] = sing_values_sorted[i];
		sig_inv[i][i] = 1/sing_values_sorted[i];
	}

	double** V = newMatrix(N, N);
	zeros(N, N, V);
	for(i = 0; i < numEigenValues; i++){
		double temp = sing_values_sorted[i];
		int index = j;
		for(j = 0; j < N; j++){
			if(sing_values[j] == temp){
				index = j;
				break;
			}
		}
		for(k = 0; k < N; k++)
			V[k][i] = E_i[k][index];
	}

	double** V_T1 = newMatrix(N, N);
	transpose(N, N, V, V_T1);

	double** DV = newMatrix(M, N);
	multiply(M, N, N, D1, V, DV);

	double** U1 = newMatrix(M, M);
	multiply(M, N, M, DV, sig_inv, U1);

	for(i = 0; i < N; i++){
		for(j = 0; j < N; j++){
			U[0][i*N + j] = (float)V[i][j];
		}
	}

	for(i = 0; i < M; i++){
		for(j = 0; j < M; j++){
			V_T[0][i*M + j] = (float)U1[j][i];
		}
	}

	for(i = 0; i < N; i++){
		SIGMA[0][i] = (float)sig[i][i];
	}

	//printf("SVD done\n\n");

}

// /* *A = double(m(m*n*(flar* )))
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */
void PCA(int retention, int M, int N, float* D, float* U, float* SIGMA, float** D_HAT, int *K)
{
	int i,j,k;

	double **D2;	//M*N matrix
	D2 = newMatrix(M, N);

	for(i = 0; i < M; i++){
		for(j = 0; j < N; j++)
			D2[i][j] = (double)D[i*N + j];
	}

	double sum_eigenValues = 0.0;
	for(i = 0; i < N; i++)
		sum_eigenValues += (double)(SIGMA[i]*SIGMA[i]);

	double sumRet = 0.0;
	for(i = 0; i < N; i++){
		sumRet += (SIGMA[i]*SIGMA[i])/sum_eigenValues;
		if(sumRet * 100 >= retention)
			break;
	}   
	K[0] = i + 1;
	//printf("K = %d\n\n", K[0]);
	double** W = newMatrix(N, K[0]);
	double** D_HAT1 = newMatrix(M, K[0]);
	D_HAT[0] = (float*)malloc(sizeof(float)*M*K[0]);

	for(i = 0; i < N; i++){
		for(j = 0; j < K[0]; j++){
			W[i][j] = (double)U[i*N + j];
		}
	}

	//printM(N, K[0], W);
	
	multiply(M, N, K[0], D2, W, D_HAT1);

	//printM(M, K[0], D_HAT1);

	//printf("\n%f\n%d\n", sumRet, iter_count);

	for(i = 0; i < M; i++){
		for(j = 0; j < K[0]; j++){
			D_HAT[0][i*K[0] + j] = (float)D_HAT1[i][j];
		}
	}
}
