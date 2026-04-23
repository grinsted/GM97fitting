


syms rhoi rho DrhoDt dt rho0
syms c offset

solve(diff(rho,t)==c*(rhoi-rho)*rho,rho)