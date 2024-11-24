//Making a Box - Hard
#include <stdio.h>
int makeBox(int n){
    int i = 0;
    int j = 0;
    if (n == 1){
        printf("#\n");
        return 0;
    }
    while (i<n){
        printf("#");
        i++;
    }
    printf("\n");
    while (j<n-2){
        printf("#");
        i = 0;
        while (i<n-2){
            printf(" ");
            i++;
        }
        printf("#\n");
        j++;
    }
    i = 0;
    while (i<n){
        printf("#");
        i++;
    }
    printf("\n");
    return 0;
}
int main(){
    makeBox(0);
    makeBox(1);
    makeBox(2);
    makeBox(3);
    makeBox(4);
    makeBox(5);
    return 0;
}