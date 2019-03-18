#include <malloc.h>
#include <omp.h>
#include <math.h>

//cp /mnt/c/Users/HP/Documents/iit_acad/col380/Lab2/col380_lab2_suite/
//g++ -fopenmp -lm lab2_io.c lab2_omp.c main_omp.c -o pca
//./pca testcase/testcase_2_2 90

// sort in decreasing order
void revSort(int N, float* W, float* W_n){
	int i, j; 
	for(i = 0; i < N; i++)
		W_n[i] = W[i];
	for (i = 0; i < N-1; i++){    
		for (j = 0; j < N-i-1; j++){  
	    	if (W_n[j] < W_n[j+1]){
	    		float temp = W_n[j];
	    		W_n[j] = W_n[j+1];
	    		W_n[j+1] = temp;
	    	}
	    }
	}
}

//get new M*N matrix;
float** newMatrix(int M, int N){
	float** W;
	W = (float **)malloc(M*sizeof(float*));
#pragma omp parallel for num_threads(4)
	for(int i = 0; i < M; i++)
	    W[i]= (float*)malloc(N*sizeof(float));
	return W;
}

void freeMatrix(int M, int N, float** W){
	for(int i = 0; i < M; i++)
	    free(W[i]);
	free(W);
}

//initialise N*N matrix as identity
void identity(int N, float** W){
#pragma omp parallel for num_threads(4)
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			W[i][j] = 0.0;
			if(i == j)
				W[i][j] = 1.0;
		}
	}
}

//initialise N*N matrix as identity
void identityLinear(int N, float* W){
#pragma omp parallel for num_threads(4)
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			W[i*N + j] = 0.0;
			if(i == j)
				W[i*N + j] = 1.0;
		}
	}
}

//initialise M*N matrix as zero
void zeros(int M, int N, float** W){
#pragma omp parallel for num_threads(4)
	for(int i = 0; i < M; i++){
		for(int j = 0; j < N; j++){
			W[i][j] = 0.0;
		}
	}
}

void zerosLinear(int M, int N, float* W){
#pragma omp parallel for num_threads(4)
	for(int i = 0; i < M*N; i++){
		W[i] = 0.0;
	}
}

// copy M*N matrix
void copyM(int M, int N, float** W, float** W_n){
#pragma omp parallel for num_threads(4)
	for(int i = 0; i < M; i++){
		for(int j = 0; j < N; j++){
			W_n[i][j] = W[i][j];
		}
	}
}

void copyMToLinear(int M, int N, float** W, float* W_n){
#pragma omp parallel for num_threads(4)
	for(int i = 0; i < M; i++){
		for(int j = 0; j < N; j++){
			W_n[i*N + j] = W[i][j];
		}
	}
}

void copyLinearToM(int M, int N, float* W, float** W_n){
#pragma omp parallel for num_threads(4)
	for(int i = 0; i < M; i++){
		for(int j = 0; j < N; j++){
			W_n[i][j] = W[i*N + j];
		}
	}
}

void copyLinearToLinear(int M, int N, float* W, float* W_n){
#pragma omp parallel for num_threads(4)
	for(int i = 0; i < M*N; i++){
		W_n[i] = W[i];
	}
}

// print M*N matrix
void printM(int M, int N, float** W){
	for(int i = 0; i < M; i++){
		for(int j = 0; j < N; j++){
			printf("%f ", W[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

// transpose of M*N matrix W
void transpose(int M, int N, float** W, float** W_T){
#pragma omp parallel for num_threads(4) schedule(static)
	for(int i = 0; i < M; i++){
		for(int j = 0; j < N; j++){
			W_T[j][i] = W[i][j];
		}
	}
}

// multiple M*N matrix with N*R matrix
void multiply(int M, int N, int R, float** W1, float** W2, float** W_M){
#pragma omp parallel for num_threads(4) schedule(static)
	for(int i = 0; i < M; i++){
		for(int j = 0; j < R; j++){
			float sum = 0.0;
			for(int k = 0; k < N; k++){
				sum += W1[i][k] * W2[k][j];
			}
			W_M[i][j] = sum;
		}
	}
}

// multiple M*N matrix with N*R matrix in 1D array
void multiplyLinear(int M, int N, int R, float* W1, float* W2, float* W_M){
#pragma omp parallel for num_threads(4) schedule(static)
	for(int i = 0; i < M; i++){
		for(int j = 0; j < R; j++){
			float sum = 0.0;
			for(int k = 0; k < N; k++){
				sum += W1[i*N + k] * W2[k*R + j];
			}
			W_M[i*R + j] = sum;
		}
	}
}

float nabs(float a){
	if(a > 0)
		return a;
	return -a;
}

// QR factors of N*N matrix
// returns weird answers some of te times
void QRfactors(int N, float** W, float** Q, float** R){

	float** V = newMatrix(N, N);
	transpose(N, N, W, V);

	for(int i = 0; i < N; i++){
		R[i][i] = 0.0;
		float norm = 0.0;
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

void QRfactorsLinear(int N, float* W, float* Q, float* R){

	float* V = (float*)malloc(N*N*sizeof(float));
	copyLinearToLinear(N, N, W, V);

	for(int i = 0; i < N; i++){
		R[i*N + i] = 0.0;
		float norm = 0.0;
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
void SVD(int M, int N, float* D, float** U, float** SIGMA, float** V_T)
{
	int i,j,k,t;

	float **D2;	//M*N matrix
	D2 = newMatrix(M, N);

	for(i = 0; i < M; i++){
		for(j = 0; j < N; j++)
			D2[i][j] = D[i*N + j];
	}

	t = N;
	N = M;
	M = t;

	float **D1;	//M*N matrix
	D1 = newMatrix(M, N);
	transpose(N, M, D2, D1);

	/*printM(M, N, D1);*/

	float **D_T;	//N*M matrix
	D_T = newMatrix(N, M);
	
	transpose(M, N, D1, D_T);

	float **DT_D;	//N*N matrix
	DT_D = newMatrix(N, N);

	multiply(N, M, N, D_T, D1, DT_D);	

	/********* QR Algorithm *********/
	/********************************/
	float **D_i = newMatrix(N, N);	//N*N matrix
	float **D_new = newMatrix(N, N);	//N*N matrix
	float **E_i = newMatrix(N, N);  //N*N matrix
	float **E_new = newMatrix(N, N);  //N*N matrix
	float **Q = newMatrix(N, N);  //N*N matrix
	float **R = newMatrix(N, N);  //N*N matrix
	
	copyM(N, N, DT_D, D_i);
	identity(N, E_i);
	zeros(N, N, Q);
	zeros(N, N, R);

	int CONG_LIMIT = 100;
	float TOLERANCE = 0.0001;
	int iter_count = 0;

	while(iter_count < CONG_LIMIT){

		iter_count++;
		QRfactors(N, D_i, Q, R);

		multiply(N, N, N, R, Q, D_new);
		multiply(N, N, N, E_i, Q, E_new);

		float maxDiff = 0.0;
		float diff;
		for(i = 0; i < N; i++){
			diff = nabs(D_new[i][i] - D_i[i][i]);	
			if(diff > maxDiff)
				maxDiff = diff;
			if(maxDiff > TOLERANCE)
				break;
		}

		//if(maxDiff < TOLERANCE)
		//	break;

		float** temp;
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

	float* eigen_values = (float*)malloc(N*sizeof(float));
	float* sing_values = (float*)malloc(N*sizeof(float));
	float* sing_values_sorted = (float*)malloc(N*sizeof(float));

	for(i = 0; i < N; i++){
		eigen_values[i] = D_i[i][i];
		sing_values[i] = sqrt(eigen_values[i]);
	}

	revSort(N, sing_values, sing_values_sorted);

	float** sig = newMatrix(M, N);
	float** sig_inv = newMatrix(N, M);
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

	float** V = newMatrix(N, N);
	zeros(N, N, V);
	for(i = 0; i < numEigenValues; i++){
		float temp = sing_values_sorted[i];
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

	float** V_T1 = newMatrix(N, N);
	transpose(N, N, V, V_T1);

	float** DV = newMatrix(M, N);
	multiply(M, N, N, D1, V, DV);

	float** U1 = newMatrix(M, M);
	multiply(M, N, M, DV, sig_inv, U1);

/*	printM(M, M, U1);
	printM(M, N, sig);*/

	for(i = 0; i < M; i++){
		for(j = 0; j < M; j++){
			U[0][i*M + j] = U1[i][j];
		}
	}

	for(i = 0; i < N; i++){
		for(j = 0; j < N; j++){
			V_T[0][i*N + j] = V_T1[i][j];
		}
	}

	for(i = 0; i < M; i++){
		SIGMA[0][i] = sig[i][i];
	}

	//printf("SVD done\n\n");

}

// /* *A = float(m(m*n*(flar* )))
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */
void PCA(int retention, int M, int N, float* D, float* U, float* SIGMA, float** D_HAT, int *K)
{
	int i,j,k;

	float **D2;	//M*N matrix
	D2 = newMatrix(M, N);

	for(i = 0; i < M; i++){
		for(j = 0; j < N; j++)
			D2[i][j] = D[i*N + j];
	}

	float sum_eigenValues = 0.0;
	for(i = 0; i < N; i++)
		sum_eigenValues += (SIGMA[i]*SIGMA[i]);

	float sumRet = 0.0;
	for(i = 0; i < N; i++){
		sumRet += (SIGMA[i]*SIGMA[i])/sum_eigenValues;
		if(sumRet * 100 >= retention)
			break;
	}   
	K[0] = i + 1;
	printf("K = %d\n\n", K[0]);
	float** W = newMatrix(N, K[0]);
	float** D_HAT1 = newMatrix(M, K[0]);
	D_HAT[0] = (float*)malloc(sizeof(float)*M*K[0]);

	for(i = 0; i < N; i++){
		for(j = 0; j < K[0]; j++){
			W[i][j] = U[i*N + j];
		}
	}

	printM(N, K[0], W);
	
	multiply(M, N, K[0], D2, W, D_HAT1);

	printM(M, K[0], D_HAT1);

	for(i = 0; i < M; i++){
		for(j = 0; j < K[0]; j++){
			D_HAT[0][i*K[0] + j] = D_HAT1[i][j];
		}
	}
}
