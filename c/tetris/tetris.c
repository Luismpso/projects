#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <conio.h>
#include <windows.h>

// Constantes do jogo
#define HEIGHT 20
#define WIDTH 10
#define BASE_SPEED 50

// Caracteres para desenho
#define BLOCK 219     // █
#define GRID_DOT 250  // ·
#define BORDER_H 205  // ═
#define BORDER_V 186  // ║
#define CORNER_TL 201 // ╔
#define CORNER_TR 187 // ╗
#define CORNER_BL 200 // ╚
#define CORNER_BR 188 // ╝

// Cores do Windows Console
enum Colors {
    BLACK = 0, BLUE = 1, GREEN = 2, CYAN = 3, RED = 4, MAGENTA = 5, BROWN = 6, LIGHTGRAY = 7,
    DARKGRAY = 8, LIGHTBLUE = 9, LIGHTGREEN = 10, LIGHTCYAN = 11, LIGHTRED = 12, LIGHTMAGENTA = 13, YELLOW = 14, WHITE = 15
};

// Peças
int shapes[7][4][4] = {
    {{0,0,0,0}, {1,1,1,1}, {0,0,0,0}, {0,0,0,0}}, // I
    {{0,0,0,0}, {0,1,1,0}, {0,1,1,0}, {0,0,0,0}}, // O
    {{0,0,0,0}, {0,1,0,0}, {1,1,1,0}, {0,0,0,0}}, // T
    {{0,0,0,0}, {0,0,1,0}, {1,1,1,0}, {0,0,0,0}}, // L
    {{0,0,0,0}, {1,0,0,0}, {1,1,1,0}, {0,0,0,0}}, // J
    {{0,0,0,0}, {0,1,1,0}, {1,1,0,0}, {0,0,0,0}}, // S
    {{0,0,0,0}, {1,1,0,0}, {0,1,1,0}, {0,0,0,0}}  // Z
};

int shapeColors[7] = { LIGHTCYAN, YELLOW, MAGENTA, BROWN, BLUE, GREEN, RED };

// Estado
int field[HEIGHT][WIDTH];
int currentPiece[4][4];
int nextPiece[4][4];
int holdPiece[4][4];
int currentID, nextID, holdID = -1;
int pieceX, pieceY, ghostY;
int score = 0, level = 1, linesClearedTotal = 0;
int gameOver = 0, canHold = 1;
int gravityThreshold = 20; 

// Engine Console
void setCursorPosition(int x, int y) {
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    COORD coord = { (short)x, (short)y };
    SetConsoleCursorPosition(hOut, coord);
}

void setColor(int color) {
    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), color);
}

void hideCursor() {
   HANDLE consoleHandle = GetStdHandle(STD_OUTPUT_HANDLE);
   CONSOLE_CURSOR_INFO info;
   info.dwSize = 100; 
   info.bVisible = FALSE;
   SetConsoleCursorInfo(consoleHandle, &info);
}

// Lógica do Jogo
void copyPiece(int src[4][4], int dest[4][4]) {
    for(int i=0; i<4; i++) for(int j=0; j<4; j++) dest[i][j] = src[i][j];
}

int checkCollision(int pX, int pY, int piece[4][4]) {
    for(int i=0; i<4; i++) {
        for(int j=0; j<4; j++) {
            if(piece[i][j]) {
                int x = pX + j;
                int y = pY + i;
                if (x < 0 || x >= WIDTH || y >= HEIGHT) return 1;
                if (y >= 0 && field[y][x]) return 1;
            }
        }
    }
    return 0;
}

void updateGhostPiece() {
    ghostY = pieceY;
    while (!checkCollision(pieceX, ghostY + 1, currentPiece)) ghostY++;
}

void spawnPiece() {
    if (nextID == -1) nextID = rand() % 7;
    currentID = nextID;
    copyPiece(shapes[currentID], currentPiece);
    nextID = rand() % 7;
    copyPiece(shapes[nextID], nextPiece);
    pieceX = WIDTH / 2 - 2; pieceY = 0; canHold = 1;
    if (checkCollision(pieceX, pieceY, currentPiece)) gameOver = 1;
    updateGhostPiece();
}

void holdCurrentPiece() {
    if (!canHold) return;
    if (holdID == -1) {
        holdID = currentID;
        copyPiece(shapes[holdID], holdPiece);
        spawnPiece();
    } else {
        int tempID = currentID; currentID = holdID; holdID = tempID;
        copyPiece(shapes[currentID], currentPiece);
        copyPiece(shapes[holdID], holdPiece);
        pieceX = WIDTH / 2 - 2; pieceY = 0;
    }
    canHold = 0;
    updateGhostPiece();
}

void rotatePiece() {
    int temp[4][4]; copyPiece(currentPiece, temp);
    for(int i=0; i<4; i++) for(int j=0; j<4; j++) currentPiece[j][3-i] = temp[i][j];
    if (checkCollision(pieceX, pieceY, currentPiece)) {
        if (!checkCollision(pieceX + 1, pieceY, currentPiece)) pieceX++;
        else if (!checkCollision(pieceX - 1, pieceY, currentPiece)) pieceX--;
        else if (!checkCollision(pieceX, pieceY - 1, currentPiece)) pieceY--;
        else copyPiece(temp, currentPiece);
    }
    updateGhostPiece();
}

// Animação de limpeza de linha
void animateLineClear(int lines[], int count) {
    if (count == 0) return;
    
    // Piscar branco
    for(int f=0; f<3; f++) {
        for (int k=0; k<count; k++) {
            int y = lines[k];
            setCursorPosition(20, y+1); // 20 é o offset X padrão
            setColor(f % 2 == 0 ? WHITE : DARKGRAY);
            for(int j=0; j<WIDTH; j++) printf("%c%c", BLOCK, BLOCK);
        }
        Sleep(60);
    }
}

void lockPiece() {
    for(int i=0; i<4; i++) 
        for(int j=0; j<4; j++) 
            if(currentPiece[i][j] && pieceY + i >= 0) 
                field[pieceY + i][pieceX + j] = currentID + 1;
    
    // Verificar Linhas
    int linesCleared[4];
    int lineCount = 0;
    
    for(int i=0; i<HEIGHT; i++) {
        int full = 1;
        for(int j=0; j<WIDTH; j++) if(field[i][j] == 0) full = 0;
        if(full) linesCleared[lineCount++] = i;
    }

    if (lineCount > 0) {
        animateLineClear(linesCleared, lineCount); // Toca animação
        
        // Remove linhas (lógica)
        for(int k=0; k<lineCount; k++) {
            int lineY = linesCleared[k];
            for(int y=lineY; y>0; y--)
                for(int x=0; x<WIDTH; x++)
                    field[y][x] = field[y-1][x];
            for(int x=0; x<WIDTH; x++) field[0][x] = 0;
        }

        // Score e Nível
        linesClearedTotal += lineCount;
        int points[] = {0, 40, 100, 300, 1200}; 
        score += points[lineCount] * level;
        if (linesClearedTotal >= level * 10) {
            level++;
            if (gravityThreshold > 2) gravityThreshold -= 2;
        }
    }
    spawnPiece();
}

void hardDrop() {
    while (!checkCollision(pieceX, pieceY + 1, currentPiece)) pieceY++;
    score += (ghostY - pieceY) * 2; 
    lockPiece();
}

// Interface Gráfica

void drawBox(int x, int y, int w, int h, char* title) {
    setColor(WHITE);
    setCursorPosition(x, y); printf("%c", CORNER_TL);
    for(int i=0; i<w*2; i++) printf("%c", BORDER_H); printf("%c", CORNER_TR);
    
    setCursorPosition(x + (w - strlen(title)/2), y); printf(" %s ", title); // Titulo no topo

    for(int i=0; i<h; i++) {
        setCursorPosition(x, y+1+i); printf("%c", BORDER_V);
        setCursorPosition(x+w*2+1, y+1+i); printf("%c", BORDER_V);
    }
    setCursorPosition(x, y+h+1); printf("%c", CORNER_BL);
    for(int i=0; i<w*2; i++) printf("%c", BORDER_H); printf("%c", CORNER_BR);
}

void drawInterface() {
    system("cls");
    int offsetX = 20; 

    // Tabuleiro Principal
    drawBox(offsetX-1, 0, WIDTH, HEIGHT, "TETRIS");

    // Next Piece
    drawBox(offsetX + WIDTH*2 + 4, 3, 6, 4, "NEXT");

    // Hold Piece
    drawBox(offsetX - 16, 3, 6, 4, "HOLD");
    
    // Info
    setCursorPosition(offsetX - 16, 10); printf("CONTROLES:");
    setCursorPosition(offsetX - 16, 11); printf("WASD / Setas");
    setCursorPosition(offsetX - 16, 12); printf("Space: Drop");
    setCursorPosition(offsetX - 16, 13); printf("C: Guardar");
}

void drawGame() {
    int offsetX = 20;

    // 1. Tabuleiro
    for(int i=0; i<HEIGHT; i++) {
        setCursorPosition(offsetX, i+1);
        for(int j=0; j<WIDTH; j++) {
            int isCurrent = (i >= pieceY && i < pieceY + 4 && j >= pieceX && j < pieceX + 4 && currentPiece[i - pieceY][j - pieceX]);
            int isGhost = (!isCurrent && i >= ghostY && i < ghostY + 4 && j >= pieceX && j < pieceX + 4 && currentPiece[i - ghostY][j - pieceX]);

            if (isCurrent) {
                setColor(shapeColors[currentID]);
                printf("%c%c", BLOCK, BLOCK); // Bloco Sólido
            } 
            else if (field[i][j] > 0) {
                setColor(shapeColors[field[i][j]-1]);
                printf("%c%c", BLOCK, BLOCK); // Bloco Sólido
            }
            else if (isGhost) {
                setColor(DARKGRAY);
                printf("[]"); // Ghost fica melhor vazado ou ::
            }
            else {
                setColor(DARKGRAY); // Cor da grid
                if (j % 2 == 0) printf("%c ", GRID_DOT); // Grid de pontos
                else printf(" %c", GRID_DOT);
            }
        }
    }

    // 2. Next
    int nextX = offsetX + WIDTH*2 + 4 + 3; // Ajuste fino pra centralizar
    for(int i=0; i<4; i++) {
        setCursorPosition(nextX, 4+i);
        for(int j=0; j<4; j++) {
            if (nextPiece[i][j]) { setColor(shapeColors[nextID]); printf("%c%c", BLOCK, BLOCK); }
            else printf("  ");
        }
    }

    // 3. Hold
    int holdX = offsetX - 16 + 3;
    for(int i=0; i<4; i++) {
        setCursorPosition(holdX, 4+i);
        for(int j=0; j<4; j++) {
            if (holdID != -1 && holdPiece[i][j]) { setColor(shapeColors[holdID]); printf("%c%c", BLOCK, BLOCK); }
            else printf("  ");
        }
    }

    // 4. Score
    setColor(WHITE);
    setCursorPosition(offsetX, HEIGHT + 2);
    printf("SCORE: %-8d NIVEL: %d", score, level);
}

int main() {
    SetConsoleOutputCP(437); // CRUCIAL para os blocos funcionarem
    srand((unsigned int)time(0));
    hideCursor();
    
    // Inicialização
    for(int i=0; i<HEIGHT; i++) for(int j=0; j<WIDTH; j++) field[i][j] = 0;
    spawnPiece();
    drawInterface();

    int gravityCounter = 0;

    while(!gameOver) {
        drawGame();
        
        if(_kbhit()) {
            char c = _getch();
            if(c == -32) { 
                c = _getch(); 
                switch(c) {
                    case 75: if(!checkCollision(pieceX-1, pieceY, currentPiece)) { pieceX--; updateGhostPiece(); } break;
                    case 77: if(!checkCollision(pieceX+1, pieceY, currentPiece)) { pieceX++; updateGhostPiece(); } break;
                    case 80: if(!checkCollision(pieceX, pieceY+1, currentPiece)) pieceY++; break;
                    case 72: rotatePiece(); break;
                }
            } else {
                if(c >= 'a' && c <= 'z') c -= 32; 
                switch(c) {
                    case 'A': if(!checkCollision(pieceX-1, pieceY, currentPiece)) { pieceX--; updateGhostPiece(); } break;
                    case 'D': if(!checkCollision(pieceX+1, pieceY, currentPiece)) { pieceX++; updateGhostPiece(); } break;
                    case 'S': if(!checkCollision(pieceX, pieceY+1, currentPiece)) pieceY++; break;
                    case 'W': rotatePiece(); break;
                    case 'C': holdCurrentPiece(); drawInterface(); break;
                    case ' ': hardDrop(); gravityCounter = 0; break;
                    case 'Q': gameOver = 1; break;
                }
            }
        }
        
        if (gravityCounter++ > gravityThreshold) {
            if(!checkCollision(pieceX, pieceY+1, currentPiece)) pieceY++;
            else lockPiece();
            gravityCounter = 0;
        }
        Sleep(BASE_SPEED);
    }

    setCursorPosition(0, HEIGHT + 5);
    setColor(RED);
    printf("GAME OVER! Score: %d\n", score);
    system("pause");
    return 0;
}