%KNOWNS:
syms A n positive
syms exx exy exz eyy eyz ezz real

%UNKNOWN:
syms tautau real positive % We want to solve for "tau:tau", so that we can express it in terms of strain rates


e = [exx exy exz; exy eyy eyz; exz eyz ezz];

%GLENs flow law:
tauE = (sum(tautau)/2)^(1/2);
phi = A*tauE^(n-1);
tau = e/phi;

%solve for tau:tau
r = solve(tautau == sum(tau(:).*tau(:)), tautau, "Real",true, "ReturnConditions",true)
