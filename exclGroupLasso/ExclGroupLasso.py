import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge


class reweightEG():
    
    """
    Exclusive group Lasso
    
    
    -------------------------
    Solves the following problem with reweighted features and weights
    
    min_w  || Xw - y ||_2^2 + alpha * sum_g ||w_g||_1^2
    
    where g \in G represents a set of indices of features

    -------------------------
    Optimisation procedure:

    Auxiliary diagonal matrix F: F_ii = sum_{j \in g} idx_group[j, i] ||w_g||_1 / w[i]
    OR equivalently G_ii = 1 / F_ii

    Compute F with w
    Compute X_tilde = X * sqrt(F)^-1
    Compute w_tilde = argmin_w_tilde f(X_tilde * w_tilde, y) + lambda ||w_tilde||_2^2
    Compute w = sqrt(F)^-1 * w_tilde
    
    -------------------------
    Parameters:
    
    :param alpha: float
        regularisation parameter in exclusive group Lasso
    
    :param idx_group: array-like, shape (n_group, n_feature)
        indicator matrix of group allocation
    
    :param n_group: int
        number of groups, must be specified if idx_group is not predefined, when n_group random groups wiil be created
        
    :param crit: float
        criteria to stop optimisation
        
    :param n_iter: int
        maximum number of iterations
        
    :param verbose: binary int
        if verbose = 1, summary of each optimisation iteration will be printed 
        
    -------------------------
    Return (as attributes):
    
    :return coef: array-like, shape (n_feature, )
        estimated weights/coefficients of exclusive group Lasso

    :return idx: list, length n_selected_feature
        indices of selected features

    :return converged: boolean
        boolean variable that indicates whether the optimisation has converged
        
    """

    def __init__(self, alpha=1, idx_group=None, n_group=None, crit=5*10**-4, n_iter=10**4, verbose=0):

        self.alpha = alpha
        self.idx_group = idx_group
        self.n_group = n_group
        self.loss_func = None
        self.crit = crit
        self.n_iter = n_iter
        self.verbose = verbose

        self.coef = None
        self.idx = None
        self.converged = False

        if n_iter < 1:
            raise ValueError('At lease one iteration is required.')

        if idx_group is None and n_group is None:
            raise KeyError('n_group must be specified if idx_group is None.')

    def _compute_G(self, w, feat_group):
        """
        Compute auxiliary matrix G

        :param w: array-like, shape (n_feature, )
            estimated weights/coefficients from the previous iteration

        :param feat_group: [??]

        :return G_diag: array-like, shape (n_feature, )
            diagonal of auxiliary matrix G
        """

        w = np.ravel(w)

        n_group = len(self.idx_group_new)
        n_feature = w.shape[0]

        G_diag = np.zeros(n_feature)
        w_group_norm = np.empty(n_group)
        for group_counter in range(n_group):
            w_group = w[self.idx_group_new[group_counter]]
            w_group_norm[group_counter] = np.linalg.norm(w_group, ord=1)

        w_group_norm[np.where(w_group_norm == 0)[0]] = 10 ** -9

        w_abs = np.abs(w)
        for feature_counter in range(n_feature):
            G_diag[feature_counter] = np.sqrt(w_abs[feature_counter] / w_group_norm[feat_group[feature_counter]])

        return G_diag

    def _compute_X_tran(self, X, G_diag):
        """
        Compute transformed feature matrix X_tilde

        :param X: array-like, shape (n_sample, n_feature)
            input features

        :param G_diag: array-like, shape (n_feature, )
            diagonal of auxiliary matrix G

        :return: array-like, shape (n_sample, n_feature)
            transformed feature matrix X_tilde
        """

        return np.dot(X, np.diag(G_diag))

    def _compute_w_tran(self, X_tran, y):
        """
        Compute transformed weight vector w_tran

        :param X_tran: array-like, shape (n_sample, n_feature)
            transformed feature matrix X_tilde

        :param y: array-like, shape (n_sample, )
            input labels

        :return: array-like, shape (n_feature, )
            transformed weight vector
        """

        w = 0
        if self.loss_func == 'hinge':
            clf = LinearSVC(fit_intercept=False, C=self.alpha)
            clf.fit(X_tran, y)
            w = clf.coef_
        elif self.loss_func == 'square':

            clf = Ridge(alpha=self.alpha, fit_intercept=False, tol=10 ** -9)
            clf.fit(X_tran, y)
            w = clf.coef_

        return np.ravel(w)

    def _create_rand_group(self, n_feature):
        """
        Create randomly allocated groups if idx_group is not specified

        :param n_feature: int
            number of features

        :return: array-like, shape (n_group, n_feature) [??]
            indicator matrix of random group allocation
        """

        self.idx_group = np.zeros((self.n_group, n_feature))
        idx = np.random.permutation(n_feature)
        idx = np.array_split(idx, self.n_group)

        for sub_counter, sub_idx in enumerate(idx):
            self.idx_group[sub_counter, sub_idx] = 1

    def _l12_norm(self, X, y):
        """
        Fit exclusive group Lasso


        Parameters:
        -------------------------
        :param X: array-like, shape (n_sample, n_feature)
            input features

        :param y: array-like, shape (n_feature, )
            input labels

        Return (as attributes)
        -------------------------
        :return coef: array-like, shape (n_feature, )
            estimated weights/coefficients

        :return idx: list, length n_selected_feature
            indices of selected features, a cut-off threshold of 10**-3 is used, can be modified to other thresholds

        :return converged: boolean
            boolean variable indicating the convergence of exclusive group Lasso
        """

        n_sample, n_feature = X.shape

        if len(np.unique(y)) == 2:
            self.loss_func = 'hinge'
        else:
            self.loss_func = 'square'

        if self.idx_group is None:
            self._create_rand_group(n_feature)

        self.idx_group_new = []
        feat_group = {}
        for group_counter in range(self.idx_group.shape[0]):
            temp = np.nonzero(self.idx_group[group_counter, :])[0]
            self.idx_group_new.append(temp)
            for idx_feature in temp:
                feat_group[idx_feature] = group_counter

        w = np.ones(n_feature) / n_feature
        G_diag = self._compute_G(w, feat_group)
        X_tran = self._compute_X_tran(X, G_diag)
        w_tran = self._compute_w_tran(X_tran, y)

        counter = 0
        while True:
            counter += 1

            w_pre = w.copy()
            w = np.multiply(w_tran, G_diag)

            G_diag = self._compute_G(w, feat_group)
            X_tran = self._compute_X_tran(X, G_diag)
            w_tran = self._compute_w_tran(X_tran, y)

            temp = np.linalg.norm(w_pre - w)

            if self.verbose == 1:
                print('iteration: %d, criteria: %.4f.' % (counter, temp))

            if temp <= self.crit or counter >= self.n_iter:
                break

        self.coef = w
        self.idx = np.where(np.abs(w) > 10 ** -3)[0]
        self.coef[np.where(np.abs(w) <= 10 ** -3)] = 0

        if counter < self.n_iter:
            self.converged = True

    def fit(self, X, y):
        """
        Fit exclusive group Lasso

        :param X: array-like, shape (n_sample, n_feature)
            input features

        :param y: array-like, shape (n_sample, )
            input labels
        """

        self._l12_norm(X, y)

    def predict(self, X):
        """
        Predict with fitted model

        :param X: array-like, shape (n_sample, n_feature)
            input features

        :return: array-like, shape (n_sample, )
            predicted labels
        """

        if self.loss_func == 'hinge':
            return np.ravel(np.sign(np.dot(X, self.coef)))
        else:
            return np.ravel(np.dot(X, self.coef))