


syms a b ed p sd B gammae n m  positive
syms epszz em negative

m = 1-1/n

epszz = em
eps = [0 0 0; 0 0 0; 0 0 epszz]
%em = sum(diag(eps))
e = eps - diag([1 1 1])*em/3
gammae = sqrt(2*sum(sum(e.*e)))

ed = sqrt(gammae^2/a+em^2/b)

%ed = abs(em)*sqrt(4/(3*a) + 1/b) %manual simplification 

emeqn = b*B^(1/n)*ed^m*p

emsol = simplify(solve(emeqn-em,em,'IgnoreAnalyticConstraints',true))
%three solutions. 
%This one chosen because this is negative when p is positive
%emsol = -3^(1/2 - n/2)*B*a^(1/2 - n/2)*b^(n/2 + 1/2)*p^n*(3*a + 4*b)^(n/2 - 1/2)

%manual simplifcation:
emsol = -b*B*p^n * (3*a  / (b*(3*a+4*b)))^(1/2 - n/2)

peqn = (1/b) * B^(-1/n) * ed ^((1-n)/n) * em
psolve = simplify(solve(p - peqn,p,'IgnoreAnalyticConstraints',true))

1/0

%--------------- now with horiz strain

clearvars
syms a b ed em p sd B gammae n m positive
syms d epsxy 
syms epszz negative

m = 1-1/n

epsxy = d*epszz
eps = [0 epsxy 0; epsxy 0 0; 0 0 epszz]
em = sum(diag(eps))
e = eps - eye(3)*em/3
gammae = sqrt(2*sum(sum(e.*e)))




ed = sqrt(gammae^2/a+em^2/b)

ed = simplify(ed)

%FURTHER SIMPLIFICATION:
%define: r = (12*b*c^2 + 3*a + 4*b)/(3*a*b)
syms r
ed = sqrt(r*epszz^2)


emeqn = b*B^(1/n)*ed^m*p

pretty(simplify(emeqn))

% emsol = simplify(solve(emeqn-em,em,'IgnoreAnalyticConstraints',true))
% emsol = emsol(3) %the negative one 
%


emsol = -B*b^n*p^n*(r^(n - 1))^(1/2)


r = (12*b*d^2 + 3*a + 4*b)/(3*a*b)
r_c = (+ 3*a + 4*b)/(3*a*b)

ratio = (-B*b^n*p^n*(r^(n - 1))^(1/2)) / (-B*b^n*p^n*(r_c^(n - 1))^(1/2))


ratio2 = (   ((12*b*d^2 + 3*a + 4*b)/(3*a*b))/((3*a + 4*b)/(3*a*b))  ) ^((n - 1)/2)


%----------------------------
syms epsxy r_V2
syms epszzc negative

n=3
solve(r_V2 - ((12*b*(epsxy^2/r_V2*epszzc^2)) /(3*a + 4*b) + 1)^(n-1),r_V2)
