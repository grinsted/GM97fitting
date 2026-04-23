% 
% 
% 
% 
% SOLVING THE INVERSE FOR EZZ
% 

syms a b A n positive
syms szz negative
syms ezz em negative
syms e1 e2
syms k positive
syms c positive

n=3
e = diag([e1,e2,ezz])

ed = e - trace(e)*eye(3)/3
e_e2 = sum(ed(:).^2)/(2*a) + 3*trace(e)^2/(4*b) %can be simplified here..
k = A^(-1/n)
ccc = k * e_e2^((1-n)/(2*n))
sigma = ccc * ( ed/a + 3*trace(e)*eye(3)/(2*b) )
sigma_zz = simplify(sigma(3,3))

%note: the order maybe unreliable:
factors = factor(sigma_zz)
denominator = 1/factors(end)
nominator = prod(factors(1:end-1))

%test it
assert(simplify(sigma_zz*denominator/nominator)==1)

%NEW equation (move a and b to other side):

%szz*a*b*A^(1/n) = nominator*a*b*A^(1/n)/denominator
syms Aszzab3 %= (szz*a*b*A^(1/n))^3

nominator = nominator*a*b*A^(1/n)

%introduce subexpressions
nominator = collect(nominator,ezz)

syms r positive 
syms p 
rexp = (3*a)/2 + (2*b)/3
pexp = (3*a*e1)/2 + (3*a*e2)/2 - (b*e1)/3 - (b*e2)/3
nominator = subs(nominator,rexp,r)
nominator = subs(nominator, pexp,p)


denominator3 = collect((denominator)^3,ezz)

%ans =
%((9*a + 4*b)/(6*A^(1/3)*a*b))*ezz + (9*a*e1 + 9*a*e2 - 2*b*e1 - 2*b*e2)/(6*A^(1/3)*a*b)



big_poly = collect(Aszzab3*denominator3 - nominator^3,ezz)

collect(simplify(big_poly/r^3),ezz)


%solve(szz^3 == sigma_zz^3,ezz)

