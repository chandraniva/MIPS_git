
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "rndxor.c"
#define LLx   2048
#define LLy  1024
#define sqLL    (LLx*LLy)
#define T 100000

 
 uint64_t sd[2]={8142396154175,9018612629};
 char  s[sqLL], v[sqLL]; int nbr[sqLL][4];
 double beta=.6,q; 
 int nn,Lx,Ly,N,sqL,ENS;
 
void nbr2d() {  int i,j,k; 
for(i=0;i<Lx; i++){ 
for(j=0;j<Ly; j++){
k =  j*Lx +i; nbr[k][0]=  j*Lx+ ((i+1)%Lx)  ; nbr[k][1]=  ((j+1)%Ly)*Lx+i; 
nbr[k][2]=  ( (i-1+Lx)%Lx) +j*Lx; nbr[k][3]=  i + Lx*((j-1+Ly)%Ly); }}
}


double op(){int i,j,k,n;double mg=0; 
for(i=0;i<Lx; i++){ n=0;for(j=0;j<Ly; j++)n+=s[j*Lx +i];  mg+= fabs(1.*n/Ly -0.5);}  return(mg);}


double H(int i){char k; double eng=0;
if(s[i]){for(k=0;k<4;k++) eng -=s[nbr[i][k]];}
return eng;
}

void update_Ising()
{ int i,j,k,a,j1; char  ei, ef; char v1;
for(j1=0;j1<sqL;j1++){	
i=  rndx()*sqL;
if (s[i]) {  j= nbr[i][rand()%4]; 
		if (s[j]==0) {ei = H(i);s[j]=1;s[i]=0;ef= H(j);
		 if(ef>ei && rndx()< 1- exp(- beta* (ef-ei))) s[i]=1,s[j]=0;}
	        
	     }
}}

void update()
{ int i,j,k,a,j1; char  ei, ef; char v1;
for(j1=0;j1<sqL;j1++){	
i=  rndx()*sqL;
if (s[i]) {  j= nbr[i][v[i]]; 
		if (s[j]==0) {ei = H(i);s[j]=1;s[i]=0;ef= H(j);
		 if(ef>ei && rndx()< 1- exp(- beta* (ef-ei))) s[i]=1,s[j]=0;}
	        
	     }  if( rndx()<q) v[i] =   (1  + v[i] + rand()%3)%4; 
}}




void init() { int k=0,i; for(i=0;i<sqL;i++) s[i]=0, v[i]=  rndx()*4; 
 while(k<N) {i= rndx()*sqL; if  (!s[i]) s[i]=1, k++;}}


int main(int argc, char *argv[])
{ int i,j,k,t,e; double mag=0,temp,p, op2=0,op4=0;
double T_min, T_max, dT;char fl[50];FILE *pt;
Lx = atoi(argv[1]);  Ly = atoi(argv[2]);  q = atof(argv[3]);  T_min=atof(argv[4]); T_max=atof(argv[5]);  dT=atof(argv[6]); ENS = atoi(argv[7]);   nbr2d();  srand(8354178); sqL=Lx*Ly; N= sqL/2;  sprintf(fl, "%sL%dx%d_q%.2f_e%d.dat", argv[8], Lx, Ly,q,ENS); ENS= pow(10,ENS);pt=fopen(fl,"w");fclose(pt);
for (temp=T_min; temp<=T_max; temp +=dT)
{beta= 1./temp;init();for(t=1; t<T;t++) update();
mag=0;op2=0;op4=0;for(e=0;e<ENS;e++) update(),p=op(), mag+=p, p*=p, op2+=p, op4+=p*p;
pt=fopen(fl,"a");
fprintf(pt,"%lf %lf  %lf %lf \n", temp, mag/ENS, op2/ENS,  op4/ENS );
fclose(pt);
//printf("%lf %lf  %lf %lf \n", temp, mag/Lx/ENS, op2/Lx/ENS,  op4/Lx/ENS );
}
}

