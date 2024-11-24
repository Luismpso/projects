//Tweaking Letters - Hard
#include <stdio.h>
#include <string.h>
int tweakLetters(char* s , int tweaks[], int n){
    int i = 0;
    while (i<n){
        s[i] = s[i] + tweaks[i];
        i++;
    }
}
int main(){
    char s[] = "abc";
    int tweaks[] = {1, -1, 1};
    int n = strlen(s);
    tweakLetters(s, tweaks, n);
    printf("Result: %s\n", s);
    return 0;
}