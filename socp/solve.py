import cvxpy as cp
import numpy as np
import argparse


def solve(A, b, c, d, F, g, f, solver):
	"""
	Solves an SOCP of the form
		p* = min f^\top x
		s.t.||A_i x+b||<=c_i^\top x+d (element-wise)
			Fx=g
	Where x in R^n,
	A is a list of l matrices in R^{n_ixn},
	b is a list of l vectors in R^{n_i}
	F in R^{kxn}
	n is the dimension of the problem,
	l is the number of SOC constraints,
	n_i is the dimension of the ith SOC constraint
	k is the number of linear constraints.
	SOLVER is the string representing the solver to be used.

	Returns (p*, x*, tuple(lambda* for SOC and linear constraints in order))
	"""
	n_soc_constraints, dim = len(A), A[0].shape[1]
	x = cp.Variable(dim)
	constraints = [
		cp.SOC(c[i].T @ x + d[i], A[i] @ x + b[i]) for i in range(n_soc_constraints)
	] + [F @ x == g]
	prob = cp.Problem(cp.Minimize(f.T @ x), constraints)
	prob.solve()#solver=solver)
	return prob.value, x.value, tuple(con.dual_value for con in constraints)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Solve an SOCP')
	parser.add_argument('-d', '--dim', type=int, default=10)
	parser.add_argument('-sc', '--soc_constraints_dims', type=int, nargs='+', default=-1)
	parser.add_argument('-lc', '--linear_constraints', type=int, default=5)
	parser.add_argument('-s', '--solver', type=str, default='ECOS')
	parser.add_argument('--seed', type=int, default=1)
	args = parser.parse_args()
	if isinstance(args.soc_constraints_dims, int):
		soc_constraints_dims = [5, 5, 5]
	m = len(soc_constraints_dims)
	n = args.dim
	l = args.linear_constraints
	n_i = soc_constraints_dims
	np.random.seed(args.seed)
	f = np.random.randn(n)
	A = []
	b = []
	c = []
	d = []
	x0 = np.random.randn(n)
	for i in range(m):
	    A.append(np.random.randn(n_i[i], n))
	    b.append(np.random.randn(n_i[i]))
	    c.append(np.random.randn(n))
	    d.append(np.linalg.norm(A[i] @ x0 + b, 2) - c[i].T @ x0)
	F = np.random.randn(l, n)
	g = F @ x0

	p, x, lambdas = solve(A, b, c, d, F, g, f, args.solver)
	print(f'Solver used {args.solver}\nOptimal value {p}\nOptimal variable {x}')
