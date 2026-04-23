%THIS ONE FINALLY WORKS


%KNOWNS:
syms A B n a b positive
syms exx exy exz eyy eyz ezz real
syms e1 e2 e3 real

%UNKNOWNS:
syms sE2 positive



%NICHOLAS FUNCTIONS
V = @(X) X(:);
iV = @(X) reshape(X,3,3);
P = a*eye(9) + (2*b/9-a/3)*V(eye(3))*V(eye(3))';


e = [exx exy exz; exy eyy eyz; exz eyz ezz];

if true
    %solve for sigma_E^2: ---works
    phi = A*sE2^((n-1)/2);

    sigma = iV((P\V(e))/phi);
    p = -trace(sigma)/3;
    tau = sigma+eye(3)*p;
    sE2_rhs = a*sum(tau(:).^2)/2 + (b*p^2)/3;

    %phi_rhs = A*sE2_rhs^((n-1)/2);

    r = solve(sE2 == sE2_rhs, sE2, "Real",true, "ReturnConditions",true);
    r
    
else
    %solve for phi instead   - this fails!
    syms phi positive

    sigma = iV((P\V(e))/phi);
    p = -trace(sigma)/3;
    tau = sigma+eye(3)*p;
    sE2_rhs = a*sum(tau(:).^2)/2 + (b*p^2)/3;
    phi_rhs = A*sE2_rhs^((n-1)/2);

    r = solve(phi == phi_rhs, phi, "Real",true, "ReturnConditions",true);
    r
end
