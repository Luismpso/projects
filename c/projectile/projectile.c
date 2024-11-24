#include <math.h>
#include <stdio.h>
int makeGraph(double r, double g,double vo, double yo){
    int l = 20;
    int c = 50;
    double vox = vo * cos(r);
    double voy = vo * sin(r);
    double t = (-voy-(sqrt((voy*voy)+(2*g*yo))))/(-g);
	double d = vox*t;
	double h = ((voy*voy)/(2*g))+yo;
    for (int j = l; j >= 0 ; j--){
        for (int i = 0; i < c ; i++){
            if (i == 0){
                printf("|");
            }
            else if (j == 0){
                printf("_");
            }
            else {
                double x = (i*d)/c; //problem with angles greater than or equal to 90 degrees (but it isnt important)
                double y = (tan(r)*x)-((g*x*x)/(2*vox*vox))+yo;
                if (round(j*h/l) == round(y*l/h)){
                    printf("x");
                }
                else printf(" ");
            }
        }
        printf("\n");
    }
    return 0;
}
int main(){
    int type;
    double yo, vo, a, g;
	printf("Selecione um numero para o tipo de movimento:\n");
    printf("1 - Movimento Horizontal\n");
    printf("2 - Movimento Obliquo\n");
    printf("-> ");
    scanf("%d", &type);
	if (type == 1){
        printf("Introduza os respetivos dados:\n");
        printf("******************************\n");
        printf("Altura inicial: ");
        scanf("%lf", &yo);
        printf("Velocidade inicial (m/s): ");
        scanf("%lf", &vo);
        printf("Aceleracao gravitica(m/s*s): (mxm/s): ");
        scanf("%lf", &g);
        double r = 0;
        double voy = vo*sin(r);
	    double vox = vo*cos(r);
	    double t = (-voy-(sqrt((voy*voy)+(2*g*yo))))/(-g);
	    double d = vox*t;
	    double ts = voy/g;
	    double h = ((voy*voy)/(2*g))+yo;
	    double vyi = vo+g*t;
	    double vi = sqrt(vox*vox+vyi*vyi);
        printf("******************************\n");
        printf("Alcance: %lf (m)\n",d);
        printf("Tempo de voo: %lf (m)\n",t);
        printf("Vetor velocidade impacto: (%d) ex + (%d) ey (m/s)\n",vo, vyi);
        printf("Velocidade impacto: %lf (m/s)\n",vi);
        printf("******************************\n");
        makeGraph(0,g,vo,yo);
        return 0;
    }
	if(type == 2){
        printf("Introduza os respetivos dados:\n");
        printf("******************************\n");
        printf("Angulo inicial (graus): ");
        scanf("%lf", &a);
        printf("Altura inicial: ");
        scanf("%lf", &yo);
        printf("Velocidade inicial (m/s): ");
        scanf("%lf", &vo);
        printf("Aceleracao gravitica(m/s*s): (mxm/s): ");
        scanf("%lf", &g);
        double r = ((a*M_PI)/180);
	    double voy = vo*sin(r);
	    double vox = vo*cos(r);
	    double t = (-voy-(sqrt((voy*voy)+(2*g*yo))))/(-g);
	    double d = vox*t;
	    double ts = voy/g;
	    double h = ((voy*voy)/(2*g))+yo;
	    double vyi = vo+g*t;
	    double vi = sqrt(vox*vox+vyi*vyi);
        printf("******************************\n");
        printf("Alcance: %lf (m)\n",d);
        printf("Tempo de subida: %lf (m)\n",ts);
        printf("Tempo de voo: %lf (m)\n",t);
        printf("Altura maxima: %lf (m)\n",h);
        printf("Vetor velocidade impacto: (%lf) ex + (%lf) ey (m/s)\n",vox, vyi);
        printf("Velocidade impacto: %lf (m/s)\n",vi);
        printf("******************************\n");
        makeGraph(r,g,vo,yo);
        return 0;
    }
    else{
        return 1;
    }
}