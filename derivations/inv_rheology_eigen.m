syms e s se a b tau real
syms A Alin n positive
syms s1 s2 s3 real
syms e1 e2 e3 real

s = [s1 0 0;0 s2 0;0 0 s3];

p = -trace(s)/3;
tau = s+eye(3)*p;

sE = sqrt(a*sum(tau(:).*tau(:))/sym(2) + b*p^2/sym(3));

n=3
phi = A*sE^(n-1);%+Alin

e = [e1 0 0;0 e2 0; 0 0 e3];

assume(s3<0)
assume(e3<0)

soln = solve(phi*diag(s)==diag(e),[e3,s1,s2], 'Real', true, 'IgnoreAnalyticConstraints', true, 'ReturnConditions', true)
