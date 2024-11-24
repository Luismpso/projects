#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void digitalFilter(float* y, float* x, int N, float* a, float* b){
    //second order
    for (int k=0 ;k<=N; k++){
        if (k == 0) {
            y[k] = (b[0]*x[k]) / a[0];
        } else if (k == 1) {
            y[k] = (- a[1]*y[k-1] + b[0]*x[k] + b[1]*x[k-1]) / a[0];
        } else {
            y[k] = (- a[1]*y[k-1] - a[2]*y[k-2] + b[0]*x[k] + b[1]*x[k-1] + b[2]*x[k-2]) / a[0];
        }
    }
}
int main(){
    int N = 50001;
    float * x = (float*)malloc(N*sizeof(float));

    FILE* infile = fopen("emg.txt", "r");
    if (infile == NULL) {
        perror("Erro ao abrir o ficheiro");
        exit(1);
    }
    for (int i = 0; i < N; i++) {
        if (fscanf(infile, "%f", &x[i]) != 1) {
            printf("%d",i);
            perror("Erro ao ler valor do ficheiro");
            exit(1);
        }
    }
    fclose(infile);

    float* y = (float*)malloc(N*sizeof(float));
    if (y == NULL) {
        perror("Erro ao alocar memória para y");
        exit(1);
    }
    //filter coeficient
    float a[] = { 1.0, -0.6202041, 0.24040821};
    float b[] = {0.15505103, 0.31010205, 0.15505103};
    digitalFilter(y, x, N, a, b);

    FILE* outfile = fopen("emgfilterd.txt", "w");
    if (outfile == NULL) {
        perror("Erro ao abrir o ficheiro para escrita");
        exit(1);
    }
    for (int i = 0; i < N; i++) {
        fprintf(outfile, "%0.2f\n", y[i]);
    }
    fclose(outfile);

    free(x);
    free(y);
    return 0;
}

