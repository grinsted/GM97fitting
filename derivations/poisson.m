syms a b positive
syms phi0 positive
syms A n positive
n=3
syms nu
syms szz 

% syms K G
% a= 1/G
% b= 1/K

%b=(3*a - 6*a*nu)/(2*nu + 2)

s = [0,0,0;0,0,0;0,0,szz];
p = -trace(s)/3;
I=eye(3);
tau = s+p*I;

sigma_e=sqrt(0.5*a*sum((tau(:).*tau(:)))+b*p^2/3)

phi0 = A * sigma_e^(n-1);

e = phi0 * (a*tau - 2*b*p*I/3)

nu = simplify(-e(1,1)/e(3,3))


%------------

eta_E = szz/e(3,3) %as in mellor and smith 1966

%nicholas constraint
% b= 9*a/2;
% nu_limit = simplify((3*a - 2*b)/(6*a + 2*b))

%eta_E = simplify(szz/(2*e(3,3)))