clear all
clc
clf()

%tDi=(alphat*perm*tInicial)/(phi*mu*ct*rw*rw)
%n=2
%tDi=tDi*1.2
%tD(1)=tDi/1.2

%while tDi <= tDf
%  tD(n)=tDi
%  pDLinha(n)=DerivLog(tD(n-1),pD(n-1),tD(n),pD(n),tD(n+1),pD(n+1))
%  tDi=tDi*1.2
%  n=n+1
%end
%pDLinha(1)=pDLinha(2)

% ====================================================
function y=Ei(x)
% ====================================================

if ((x>0.0)&(x<1.0)) then
  a0=-0.577216;
  a1= 0.999992;
  a2=-0.249911;
  a3= 0.055200;
  a4=-0.009760;
  a5= 0.001079;
  y=a0+a1*x+a2*x*x+a3*x*x*x+a4*x*x*x*x+a5*x*x*x*x*x-log(abs(x));
else
  if (x<10.0) then
    a1=2.334733;
    a2=0.250621;
    b1=3.330657;
    b2=1.681534;
    y=((x*x+a1*x+a2)/(x*x+b1*x+b2))/(x*exp(x));
  else
    if (x<100.0)
      a1=4.03640;
      a2=1.15198;
      b1=5.03637;
      b2=4.19160;
      y=((x*x+a1*x+a2)/(x*x+b1*x+b2))/(x*exp(x));
    else
      y=0.0;
    end
  end
end
end
% ====================================================

% ====================================================
function y=LinhaFonte(r,t)
    y=0.5*Ei((r*r)/(4.0*t));
end
% ====================================================

% ====================================================
function y=DerivLog(x1,y1,xc,yc,x2,y2)
  d1=((y2-yc)/log(x2/xc))*(log(xc/x1)/log(x2/x1));
  d2=((yc-y1)/log(xc/x1))*(log(x2/xc)/log(x2/x1));
  y=(d1+d2);
end
% ====================================================
% Regime Pseudo-Permanente
% ====================================================
function y=FT_PWD_PSSFR(rde,u)
rd=1;
%a = (besselk(1,rde*sqrt(u))*besseli(1,sqrt(u))/besseli(1,rde*sqrt(u))) - besselk(1,sqrt(u));
%y = - (besselk(1,rde*sqrt(u))*besseli(0,rd*sqrt(u)))/(u*sqrt(u)*besseli(1,rde*sqrt(u))*a) + (besselk(0,rd*sqrt(u))/(u*sqrt(u)*a));
y =  (besselk(1,rde*sqrt(u))*besseli(0,rd*sqrt(u))+besselk(0,rd*sqrt(u))*besseli(1,rde*sqrt(u)))/(u*sqrt(u)* (-besseli(1,sqrt(u))*besselk(1,rde*sqrt(u))+besselk(1,sqrt(u))*besseli(1,rde*sqrt(u))));
end
% ====================================================
% Regime Pseudo-Permanente para Naturalmente Fraturado
% ====================================================
function y=FT_PWD_PSSFR_NFR(rde,u)
rd=1;
w=0.05;
l=0.001;
fu=(l+w*(1-w)*u)/(l+(1-w)*u);
CD=0;
S=0;
%a = (besselk(1,rde*sqrt(u))*besseli(1,sqrt(u))/besseli(1,rde*sqrt(u))) - besselk(1,sqrt(u));
%y = - (besselk(1,rde*sqrt(u))*besseli(0,rd*sqrt(u)))/(u*sqrt(u)*besseli(1,rde*sqrt(u))*a) + (besselk(0,rd*sqrt(u))/(u*sqrt(u)*a));
y =  (besselk(1,rde*sqrt(u*fu))*besseli(0,rd*sqrt(u*fu))+besselk(0,rd*sqrt(u*fu))*besseli(1,rde*sqrt(u*fu)))/(u*sqrt(u*fu)* (-besseli(1,sqrt(u*fu))*besselk(1,rde*sqrt(u*fu))+besselk(1,sqrt(u*fu))*besseli(1,rde*sqrt(u*fu))));
end
% ====================================================
% Regime Permanente
% ====================================================
function y=FT_PWD_SSFR(rde,u)
rd=1;
b = (besselk(0,rde*sqrt(u))*besseli(1,sqrt(u))/besseli(0,rde*sqrt(u))) + besselk(1,sqrt(u));
y = - (besselk(0,rde*sqrt(u))*besseli(0,rd*sqrt(u))/u*sqrt(u)*besseli(0,rde*sqrt(u))*b) + (besselk(0,rd*sqrt(u))/(u*sqrt(u)*b));
end
% ====================================================
% Regime Permanente para Naturalmente Fraturado
% ====================================================
function y=FT_PWD_SSFR_NFR(rde,u)
rd=1;
w=0.01;
l=0.00001;
fu=(l+w*(1-w)*u)/(l+(1-w)*u);
CD=0;
S=0;
b = (besselk(0,rde*sqrt(u*fu))*besseli(1,sqrt(u*fu)))/besseli(0,rde*sqrt(u*fu)) + besselk(1,sqrt(u*fu));
y = - (besselk(0,rde*sqrt(u*fu))*besseli(0,rd*sqrt(u*fu))/u*sqrt(u*fu)*besseli(0,rde*sqrt(u*fu))*b) + (besselk(0,rd*sqrt(u*fu))/(u*sqrt(u*fu)*b));
end
% ====================================================
%Solução Homogenea Pura
% ====================================================
function y=FT_PWD_RF(u)
y=besselk(0,sqrt(u))/u*sqrt(u)*besselk(1,sqrt(u));
end
% ====================================================
%Falha Selante
% ====================================================
function y=FT_PWD_SF(LD,u)
y=(besselk(0,sqrt(u))/u*sqrt(u)*besselk(1,sqrt(u)))+(besselk(0,2*LD*sqrt(u))/u*sqrt(u)*besselk(1,sqrt(u)));
end
% ====================================================
%  Reservatório Homogeneo
% ====================================================
function y=FT_PWD_HOM(u)
CD=0;
S=0;
y=besselk(0,sqrt(u))/((u*sqrt(u))*besselk(1,sqrt(u)));
end
% ====================================================
%  Reservatório Naturalmente Fraturado
% ====================================================
function y=FT_PWD_NFR(LD,u)
w=0.01;
l=0.00001;
fu=(l+w*(1-w)*u)/(l+(1-w)*u);
CD=0;
S=0;
y=besselk(0,sqrt(u*fu))/((u*sqrt(u*fu))*besselk(1,sqrt(u*fu)));
end
% ====================================================
% Reservatorio Naturalmente Fraturado com Falha Selante
% ====================================================
function y=FT_PWD_NFR_SF(LD,u)
w=0.05;
l=0.001;
fu=(l+w*(1-w)*u)/(l+(1-w)*u);
CD=0;
S=0;
y=besselk(0,sqrt(u*fu))/((u*sqrt(u*fu))*besselk(1,sqrt(u*fu)))+ besselk(0,2*LD*sqrt(u*fu))/((u*sqrt(u*fu))*besselk(1,sqrt(u*fu)));
end
% ====================================================
%  Reservatório Homogeneo com Estocagem e SKIN
% ====================================================
function y=FT_PWD_HOM_CD_S(u)
CD=10;
S=2;
%%%%%%%%%%%%%%%%%%%%%%
% Esse é o meu G BARRA
y1=besselk(0,sqrt(u))/((u*sqrt(u))*besselk(1,sqrt(u)))+S/u;
%%%%%%%%%%%%%%%%%%%%%%
y = y1/(1+(u*u*CD*y1));
end
% ====================================================
% Reservatorio Naturalmente Fraturado com Estocagem e Skin
% ====================================================
function y=FT_PWD_NFR_CD_S(u)
CD=10;
S=2;
w=0.01;
l=0.00001;
fu=(l+w*(1-w)*u)/(l+(1-w)*u);
y1=besselk(0,sqrt(u*fu))/((u*sqrt(u*fu))*besselk(1,sqrt(u*fu)))+S/u;
y=y1/(1+(u*u*CD*y1));
end
% ====================================================
% Regime Pseudo-Permanente com Estocagem e Skin
% ====================================================
function y=FT_PWD_PSSFR_CD_S(rde,u)
rd=1;
CD=10;
S=2;
y1 =  (besselk(1,rde*sqrt(u))*besseli(0,rd*sqrt(u))+besselk(0,rd*sqrt(u))*besseli(1,rde*sqrt(u)))/(u*sqrt(u)* (-besseli(1,sqrt(u))*besselk(1,rde*sqrt(u))+besselk(1,sqrt(u))*besseli(1,rde*sqrt(u))))+S/u;
y=y1/(1+(u*u*CD*y1));
end
% =========================================================================
% Regime Pseudo-Permanente para Naturalmente Fraturado com Estocagem e Skin
% =========================================================================
function y=FT_PWD_PSSFR_NFR_CD_S(rde,u)
rd=1;
w=0.05;
l=0.001;
fu=(l+w*(1-w)*u)/(l+(1-w)*u);
CD=10;
S=2;
y1 =  (besselk(1,rde*sqrt(u*fu))*besseli(0,rd*sqrt(u*fu))+besselk(0,rd*sqrt(u*fu))*besseli(1,rde*sqrt(u*fu)))/(u*sqrt(u*fu)* (-besseli(1,sqrt(u*fu))*besselk(1,rde*sqrt(u*fu))+besselk(1,sqrt(u*fu))*besseli(1,rde*sqrt(u*fu))))+ S/u;
y=y1/(1+(u*u*CD*y1));
end
% ====================================================
% Regime Permanente com Estocagem e Skin
% ====================================================
function y=FT_PWD_SSFR_CD_S(rde,u)
rd=1;
CD=10;
S=2;
b = (besselk(0,rde*sqrt(u))*besseli(1,sqrt(u))/besseli(0,rde*sqrt(u))) + besselk(1,sqrt(u));
y1 = - (besselk(0,rde*sqrt(u))*besseli(0,rd*sqrt(u))/u*sqrt(u)*besseli(0,rde*sqrt(u))*b) + (besselk(0,rd*sqrt(u))/(u*sqrt(u)*b))+S/u;
y=y1/(1+(u*u*CD*y1));
end
% ====================================================
% Regime Permanente para Naturalmente Fraturado com Estocagem e Skin
% ====================================================
function y=FT_PWD_SSFR_NFR_CD_S(rde,u)
rd=1;
w=0.01;
l=0.00001;
fu=(l+w*(1-w)*u)/(l+(1-w)*u);
CD=10;
S=2;
b = (besselk(0,rde*sqrt(u*fu))*besseli(1,sqrt(u*fu)))/besseli(0,rde*sqrt(u*fu)) + besselk(1,sqrt(u*fu));
y1 = - (besselk(0,rde*sqrt(u*fu))*besseli(0,rd*sqrt(u*fu))/u*sqrt(u*fu)*besseli(0,rde*sqrt(u*fu))*b) + (besselk(0,rd*sqrt(u*fu))/(u*sqrt(u*fu)*b))+S/u;
y=y1/(1+(u*u*CD*y1));
end
% ====================================================
%Falha Selante com Estocagem e Skin
% ====================================================
function y=FT_PWD_SF_CD_S(LD,u)
CD=10;
S=2;
y1=((besselk(0,sqrt(u))/u*sqrt(u)*besselk(1,sqrt(u)))+(besselk(0,2*LD*sqrt(u))/u*sqrt(u)*besselk(1,sqrt(u))))+ S/u;
y=y1/(1+(u*u*CD*y1));
end
% ==========================================================================
% Reservatorio Naturalmente Fraturado com Falha Selante com Estocagem e Skin
% ==========================================================================
function y=FT_PWD_NFR_SF_CD_S(LD,u)
w=0.005;
l=0.0001;
fu=(l+w*(1-w)*u)/(l+(1-w)*u);
CD=0;
S=2;
y1=(besselk(0,sqrt(u*fu))/((u*sqrt(u*fu))*besselk(1,sqrt(u*fu)))+ besselk(0,2*LD*sqrt(u*fu))/((u*sqrt(u*fu))*besselk(1,sqrt(u*fu))))+S/u;
y=y1/(1+(u*u*CD*y1));
end
% ====================================================





% ====================================================
function y=minimo(i1, i2)
% ====================================================
if (i1<i2)
    y=i1;
else
    y=i2;
end
end
% ====================================================
function y=CALCULA_PWD(td,LD)
%Rotina para inversão numérica de uma função no espaço de Laplace pelo algoritimo de Stehfest
% ====================================================
NP_STEHFEST=12;
n=NP_STEHFEST;

for i=1:n
    somat = 0;
    inis = floor((i + 1)/2);
    fins = minimo(i,floor(n/2));
    a0=(-1)^(i+(n/2));
%    disp([i,inis,fins])

    for k=inis:fins
        a1=k^(1+(n/2));
        a2a=2*k;
        a2b=factorial((a2a));
        a3a=((n/2)-k);
        a3b=factorial((a3a));
        a4a=factorial(k);
        a4b=a4a^2;
        a5a=i-k;
        a5b=factorial((a5a));
        a6a=((2*k)-i);
        a6b=factorial((a6a));
        somat=somat+(((a1*a2b)/(a3b*a4b*a5b*a6b)));
    end
    v(i) = a0 * somat;
 end
st = 0;
for m = 1:NP_STEHFEST
    u = m * log(2.0) / td;
%=================================
% AQUI ENTRA A FUNÇÃO QUE EU QUERO
%=================================

% 1-Homogeneo - Radial Infinito
%  st = st + v(m) * FT_PWD_HOM(u);
% 2-Homogeneo - Radial Infinito com Estocagem e Skin
%  st = st + v(m) * FT_PWD_HOM_CD_S(u);
% 3-Naturalmente Fraturado - Radial Infinito
%  st = st + v(m) * FT_PWD_NFR(10,u);
% 4-Naturalmente Fraturado - Radial Infinito com Estocagem e Skin
  st = st + v(m) * FT_PWD_NFR_CD_S(u);
% 5-Homogeneo - Falha Selante
%  st = st + v(m) * FT_PWD_SF(LD,u);
% 6-Homogeneo - Falha Selante com Estocagem e Skin
%  st = st + v(m) * FT_PWD_SF_CD_S(LD,u);
% 7-Naturalmente Fraturado - Falha Selante
%  st = st + v(m) * FT_PWD_NFR_SF(LD,u);
% 8-Naturalmente Fraturado - Falha Selante com Estocagem e Skin
%  st = st + v(m) * FT_PWD_NFR_SF_CD_S(LD,u);
% 9-Homogêneo - Com Manutenção de Pressão
%  st = st + v(m) * FT_PWD_SSFR(1000,u)
% 10-Homogêneo - Com Manutenção de Pressão com Estocagem e Skin
%  st = st + v(m) * FT_PWD_SSFR_CD_S(1000,u);
% 11-Naturalmente Fraturado - Com Manutenção de Pressão
%  st = st + v(m) * FT_PWD_SSFR_NFR(1000,u)
% 12-Naturalmente Fraturado - Com Manutenção de Pressão com Estocagem e Skin
%  st = st + v(m) * FT_PWD_SSFR_NFR_CD_S(1000,u);
% 13-Homogêneo - Sem Manutenção de Pressão
%  st = st + v(m) * FT_PWD_PSSFR(1000,u);
% 14-Homogêneo - Sem Manutenção de Pressão com Estocagem e Skin
%  st = st + v(m) * FT_PWD_PSSFR_CD_S(1000,u);
% 15-Naturalmente Fraturado - Sem Manutenção de Pressão
%  st = st + v(m) * FT_PWD_PSSFR_NFR(1000,u);
% 16-Naturalmente Fraturado - Sem Manutenção de Pressão com Estocagem e Skin
%  st = st + v(m) * FT_PWD_PSSFR_NFR_CD_S(1000,u);

end

y = log(2.0) * st / td;
end
% ====================================================

% Dados Dimensionais
Qs      =100:100:1000.0;
mu      =1.0;
Bo      =1.0;
phi     =0.3;
ct      =200E-06;
pi      =300.0;
rw      =0.1;
tFluxo  =120.0;
tInicial=1e-5;
perms   =500:500:10000;
esp     =100.0;
d       =10;
LD      =d./rw;

% BASTA COLOCAR RUIDO != 0 PARA TER AS MESMAS SOLUÇÕES COM RUÍDO
RUIDOS=0;%:0.1:0.5;

rde = 2000;
alphap=19.03;
alphat=3.484e-4;

wells_pwf = [];
wells_deltap_deriv = [];
N_TIMES = 51

for RUIDO=RUIDOS
    for q=Qs
        for perm=perms
            tDi=(alphat*perm*tInicial)/(phi*mu*ct*rw*rw);
            tDf=(alphat*perm*tFluxo)/(phi*mu*ct*rw*rw);
            disp([tDi, tDf])

            n=1;
            tD(1)=tDi;

            while (tDi <= tDf) & (n <= N_TIMES)
                pD(n)=CALCULA_PWD(tDi,LD);
                tD(n)=tDi;
                tDi=tDi*1.3;
                n=n+1;
            end

            for np=1:n-1
                t(np)=(phi*mu*ct*rw*rw*tD(np))/(alphat*perm);
                deltaP(np)=((alphap*q*Bo*mu*pD(np))/(perm*esp))+0.05*RUIDO*sin((2*rand()-1)*t(np));
                pwf(np)=pi-deltaP(np);
            end
            wells_pwf = [wells_pwf, pwf'];

            for np=2:n-2
                deltaPLinha(np)=DerivLog(t(np-1),deltaP(np-1),t(np),deltaP(np),t(np+1),deltaP(np+1));
            end
            deltaPLinha(1)=deltaPLinha(2);
            deltaPLinha(n-1)=deltaPLinha(n-2);
            wells_deltap_deriv = [wells_deltap_deriv, deltaPLinha'];
        end
    end
end

csvwrite('pwf_hom.csv', wells_pwf)
csvwrite('deltap_hom.csv', wells_deltap_deriv)
%csvwrite('times.csv', times)