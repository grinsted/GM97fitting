syms e s se tau real
syms a b positive
syms A Alin n positive
syms sxx sxy sxz syy syz szz real
syms exx exy exz eyy eyz ezz real

s = [sxx sxy sxz; sxy syy syz; sxz syz szz];

p = -trace(s)/3;
tau = s+eye(3)*p;


sE = sqrt(a*sum(tau(:).*tau(:))/2 + b*p^2/3);

syms tautau pp real positive
sE = sqrt(a*sum(tautau)/2 + (b*pp)/3);
phi = A*sE^(n-1);%+Alin

e = [exx exy exz; exy eyy eyz; exz eyz ezz];

if true
    s = e/phi
    p = -trace(s)/3;
    tau = s+eye(3)*p;

    eqns = [sum(tau(:).*tau(:))==tautau, p^2 == pp]
    r = solve(eqns, [tautau,pp], "Real",true,"ReturnConditions",true,"IgnoreAnalyticConstraints",true)
else

    %
    %This doesnt work:
    %solve(e==phi*s,[sxx sxy sxz syy syz szz])



    %-----------------------------------
    V = @(X) X(:);
    iV = @(X) reshape(X,3,3);
    %
    P = a*eye(9) + (2*b/9-a/3)*V(eye(3))'*V(eye(3));
    %

    %now our equation looks like this - still cannot solve:
    %solve(phi*P*V(s)==V(e),[sxx sxy sxz syy syz szz])

    s = iV((P\V(e))/phi);
    p = -trace(s)/3;
    tau = s+eye(3)*p;

    eqns = [sum(tau(:).*tau(:))==tautau, p^2 ==pp].^(1-n)
    r= solve(eqns, [tautau,pp])

    %now our equation looks like this: (still doesnt solve)
    %solve(phi*V(s) == ve_over_p, [sxx sxy sxz syy syz szz])


end