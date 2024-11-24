#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(){
    int i = 0;
    double d;
    char b[128];
    int pi = (int)d;
    double pf = d - (int)d;; 

    printf("Introduza o decimal: ");
    scanf("%lf", &d);

    if (pi == 0) {
        b[i++] = '0';
    } 
    else {
        while (pi > 0) {
            b[i++] = (pi % 2) + '0';
            pi = pi / 2;
        }
    }

    if (pf > 0) {
        b[i++] = '.';
        while (pf > 0) {
            pf *= 2;
            if (pf >= 1.0) {
                b[i++] = '1';
                pf -= 1.0;
            } else {
                b[i++] = '0';
            }
        }
    }

    b[i] = '\0';
    printf("Binário: %s\n", b);
    return 0;
}