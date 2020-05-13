#!/usr/bin/env python
# coding: utf-8


import numpy as np

from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder


class GSVotingClassifier(object):
    """Soft Voting/Majority Rule classifier for unfitted estimators.

    Objects of this class instantiate a series of base estimators and perform
    a grid-search on their hyperparameters. The estimators with highest f-beta
    scores are chosen to compose the final comitee.

    Note that due to the grid search procedure, the fitting stage may take a
    long time.

    Parameters
    ----------
    estimators : list of (str, estimator, parameters) tuples
        Invoking the ``fit`` method on the ``VotingClassifier`` will fit clones
        of those original estimators that will be stored in the class attribute
        ``self.estimators_``. An estimator can be set to ``'drop'``
        using ``set_params``.
    voting_type : str, {'hard', 'soft'} (default='hard')
        If 'hard', uses predicted class labels for majority rule voting.
        Else if 'soft', predicts the class label based on the argmax of
        the sums of the predicted probabilities, which is recommended for
        an ensemble of well-calibrated classifiers.
    weights : array-like, shape (n_classifiers,), optional (default=`None`)
        Sequence of weights (`float` or `int`) to weight the occurrences of
        predicted class labels (`hard` voting) or class probabilities
        before averaging (`soft` voting). Uses uniform weights if `None`.
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for ``fit``.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors
    cv : int or None, optional (default=None)
    top_n : int or None, optional (default=5)
    """
    def __init__(self, estimators, voting_type="soft", weights=None,
                 n_jobs=None, cv=None, top_n=5):

        if not estimators:
            raise ValueError("No estimators given.")
        if not isinstance(estimators, list):
            estimators = list(estimators)

        self.estimators_ = estimators
        self.voting_type = voting_type
        self.weights = weights
        self.n_jobs = n_jobs
        self.cv = cv
        self.top_n = top_n

        self.voting_estimator_ = None
        self.le_ = None
        self.classes_ = None

    def fit(self, X, y):
        """Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
        """
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError("Multilabel and multi-output"
                                      " classification is not supported.")

        if self.voting_type not in ("soft", "hard"):
            raise ValueError("Voting type must be 'soft' or 'hard'; got (voting=%r)"
                             % self.voting_type)

        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_

        scores = [None] * len(self.estimators_)
        best_clf = {}

        for i, (clf_name, clf, params) in enumerate(self.estimators_):
            gs = GridSearchCV(clf, params, cv=self.cv)
            gs.fit(X, y)
            scores[i] = (clf_name, gs.best_score_)
            best_clf[clf_name] = gs.best_estimator_

        top_n_clf = sorted(scores, key=lambda tup: tup[1], reverse=True)[:self.top_n]
        top_n_clf = [tup[0] for tup in top_n_clf]

        self.estimators_ = [(k, v) for k, v in best_clf.items() if k in top_n_clf]
        self.voting_estimator_ = VotingClassifier(self.estimators_,
                                                  voting=self.voting_type,
                                                  weights=self.weights,
                                                  n_jobs=self.n_jobs)
        self.voting_estimator_.fit(X, y)
        return self

    def predict(self, X, return_all=False):
        """Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples.
        return_all : boolean, optional
            Whether to return the predictions of all estimators
            (if set to `True`), or just the voting results.

        Returns
        -------
        predictions
            g, or all estimators.
            If `return_all=False`:
                array-like, shape (n_samples,)
                Predicted class labels for the voting estimator.
            If `return_all=True`:
                dict{estimator_name => array-like shape (n_samples,)}
                A dict with the estimators names as keys and their predictions
                as values
        """
        if not return_all:
            return self.voting_estimator_.predict(X)

        votes = {"voting": self.voting_estimator_.predict(X)}
        for clf_name, clf in self.estimators_:
            votes[clf_name] = clf.predict(X)

        return votes
