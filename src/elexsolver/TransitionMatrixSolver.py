import cvxpy as cp


class TransitionMatrixSolver:
    def __init__(self):
        self.transition_matrix = None

    @staticmethod
    def __get_constraint(X, strict):
        if strict:
            return [cp.sum(X, axis=1) == 1]
        return [cp.sum(X, axis=1) <= 1.1, cp.sum(X, axis=1) >= 0.9]

    def __solve(self, A, B, strict):
        transition_matrix = cp.Variable((A.shape[1], B.shape[1]))
        loss_function = cp.norm(A @ transition_matrix - B, "fro")
        objective = cp.Minimize(loss_function)
        constraint = TransitionMatrixSolver.__get_constraint(transition_matrix, strict)
        problem = cp.Problem(objective, constraint)
        problem.solve()
        return transition_matrix.value

    def fit(self, A, B, strict=False):
        transition_matrix = self.__solve(A, B, strict)
        self.transition_matrix = transition_matrix

    def predict(self, A):
        return A @ self.transition_matrix
