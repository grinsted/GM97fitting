%THIS ONE does not work


%KNOWNS:
syms A B n a b positive
syms exx exy exz eyy eyz ezz real

%UNKNOWNS:
syms phi


e = [exx exy exz; exy eyy eyz; exz eyz ezz];


sigma = e/phi;
p = -trace(sigma)/3;
tau = sigma+eye(3)*p;

sE2 = a*sum(tau(:).*tau(:))/2 + (b*p^2)/3

phi_rhs = A*sE2^((n-1)/2)+B;
%syms q positive
%phi_rhs = subs(phi_rhs,(n/2 - 1/2),q)
%assume(q>0)

r = solve(1/phi ==1/phi_rhs, phi, "Real",true, "ReturnConditions",true)

%FINAL STEPS: 
%re-express sE2 in terms of invariants of e
% insert in into 
% * sE = sE2^(1/2);
% * phi = A*sE^(n-1);
% * sigma = e/phi