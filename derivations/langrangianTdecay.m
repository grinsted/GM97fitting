

syms k c rho positive
syms Tm kz t z
syms Ts omega positive
syms dkdz
syms kt

dkdz=0

seasonal = exp(i*(omega*t-kz*z))
T=Tm+Ts*exp(-kt*t)*seasonal

%equation
% rho*c* DT/Dt = d(k*dT/dz)/dz

dTdz = diff(T,z)
DTdt = k*diff(dTdz,z) + dkdz*dTdz  %RHS

%---------- find kz ------------
eqn = (diff(T,t)-DTdt)/seasonal 
eqn= simplify(eqn)
kt_expr = solve(eqn,kt)

%---------- find DTsDt -----
%DTsdt = DTdt/seasonal
%DTsdt = simplify(DTsdt)

simplify(subs(DTsdt,kt,kt_expr))

%Ts_new = Ts_old * exp(kz*(- k*kz + dkdz*1i))

%how to determin kz?

