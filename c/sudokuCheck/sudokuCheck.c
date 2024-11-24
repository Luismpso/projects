#include <math.h>
#include <stdio.h>
int main(){
    int r = 0;
    int i ,j ,k, l, x, y;
    int rsudoku [9][9] = {
        {9, 5, 4, 8, 1, 6, 3, 7, 2}, 
        {7, 8, 6, 2, 5, 3, 1, 4, 9}, 
        {1, 2, 3, 7, 9, 4, 6, 5, 8}, 
        {3, 1, 8, 9, 7, 2, 4, 6, 5}, 
        {2, 7, 9, 4, 6, 5, 8, 1, 3}, 
        {4, 6, 5, 3, 8, 1, 9, 2, 7}, 
        {8, 4, 7, 1, 2, 9, 5, 3, 6}, 
        {5, 3, 2, 6, 4, 8, 7, 9, 1}, 
        {6, 9, 1, 5, 3, 7, 2, 8, 4}};
    for (i = 0; i < 9; i++){
        for (j = 0; j < 9; j++){
            // Check the row for duplicates
            for (k = i + 1; k < 9; k++){ //not optmize
                if (rsudoku [i][j] == rsudoku [k][j]){
                    r = 1; 
                }
            }
            // Check the column for duplicates
            for (l = j + 1; l < 9; l++){
                if (rsudoku [i][j] == rsudoku [i][l]){
                    r = 1;
                }
            }
        }
    }
    // Check for duplicates in 3x3 subgrids
    for (i = 0; i < 9; i += 3){
        for (j = 0; j < 9; j += 3){
            for (k = 0; k < 3; k++){
                for (l = 0; l < 3; l++){
                    for (x = k + 1; x < 3; x++){
                        for (y = l + 1; y < 3; y++){
                            if (rsudoku [i+k][j+l] == rsudoku [i+x][j+y]){
                                r = 1;
                            }
                        }
                    }
                }
            }
        }
    }
    if (r == 0){
        printf("The Sudoku is valid.\n");
        return 0;
    }
    else{
        printf("The Sudoku is not valid.\n");
        return 1;
    }
}