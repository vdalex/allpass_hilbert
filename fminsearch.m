[params, fval, exitflag, result] = fminsearch(
												'phcost', params, 
												optimset('MaxIter',4000, 'MaxFunEvals', 5000, 'TolFun',1e-12, 'TolX',1e-8, 'Display','on'),
												freqopt, 
												phdesired, 
												pnorm
											 )  % find optimal parameters




Octave:

: x = fminsearch (fun, x0)
: x = fminsearch (fun, x0, options)
: [x, fval] = fminsearch (…)

Find a value of x which minimizes the function fun.

The search begins at the point x0 and iterates using the Nelder & Mead Simplex algorithm (a derivative-free method). This algorithm is better-suited to functions which have discontinuities or for which a gradient-based search such as fminunc fails.

Options for the search are provided in the parameter options using the function optimset. 
Currently, fminsearch accepts the options: "TolX", "MaxFunEvals", "MaxIter", "Display". For a description of these options, see optimset.

    On exit, the function returns x, the minimum point, and fval, the function value thereof.

    Example usages:

    fminsearch (@(x) (x(1)-5).^2+(x(2)-8).^4, [0;0])

    fminsearch (inline ("(x(1)-5).^2+(x(2)-8).^4", "x"), [0;0])

    See also: fminbnd, fminunc, optimset. 

MATLAB:

x = fminsearch(fun,x0)
x = fminsearch(fun,x0,options)
x = fminsearch(problem)
[x,fval] = fminsearch(___)
[x,fval,exitflag] = fminsearch(___)
[x,fval,exitflag,output] = fminsearch(___)

x = fminsearch(fun,x0) starts at the point x0 and attempts to find a local minimum x of the function described in fun.
x = fminsearch(fun,x0,options) minimizes with the optimization options specified in the structure options. Use optimset to set these options.
x = fminsearch(problem) finds the minimum for problem, where problem is a structure.
[x,fval] = fminsearch(___), for any previous input syntax, returns in fval the value of the objective function fun at the solution x.
[x,fval,exitflag] = fminsearch(___) additionally returns a value exitflag that describes the exit condition.
[x,fval,exitflag,output] = fminsearch(___) additionally returns a structure output with information about the optimization process.

