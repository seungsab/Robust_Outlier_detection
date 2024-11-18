from sklearn.decomposition import PCA
from sklearn import mixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from scipy.stats.distributions import chi2


class pca_online():

    def __init__(self, component_use='major', gmm_run=1, n_sample_max=1e4, method_Ncomp='eigengap', method_thresh = 'percentile', cutoff=0.95, n_stall=5, alpha=0.99, n_comp_min=1, n_comp_max=3, cv_types=["spherical", "tied", "diag", "full"]):
        # BASELINE MODEL
        self.pca, self.gmm = [], []         # PCA model and GMM for Block-linearization

        # SAMPLE INVENTORY
        # All samples including normal and abnormal samples
        self.X, self.Y = [], []

        # New samples of being evaluated
        self.Xnew, self.Ynew = [], []

        # Ressiduals and indice of anomaly for all samples
        self.resi0_Q, self.resi0_T2 = [], []
        self.N_Xinit = []                   # Initial number of training samples

        # ANOMALY PARMAETER
        # History of "# Comp." and "Threshold",  residuals of all samples
        self.ncomp, self.threshold, self.resi_Q, self.resi_T2 = [], [], [], []
        self.pca_scaler = StandardScaler()  # Standardization
        self.method_Ncomp, self.method_thresh = method_Ncomp, method_thresh

        # Maximal number of samples, significance level, cutoff for optimal components
        self.n_sample_max, self.alpha, self.cutoff = n_sample_max, alpha, cutoff
        # tolerance for alarming, Counts of consecutive violations
        self.n_stall, self.n_stall_count = n_stall, 0
        self.component_use = component_use
        

        # GMM PARAMETERS
        self.gmm_scaler = StandardScaler()  # Standardization
        self.gmm_n_comp_min = n_comp_min  # Minimal # of Gaussian components
        self.gmm_n_comp_max = n_comp_max  # Maximal # of Gaussian components
        self.gmm_cv_types = cv_types     # Covaraince type
        self.gmm_run = gmm_run

    def fit_initial_pca(self, Xinit):
        # Construct PCA using all samples
        # Append new sample into existing samples
        self.X, self.N_Xinit = Xinit, Xinit.shape[0]

        # Fit PCA sklearn's function
        Xinit_scaled = self.pca_scaler.fit_transform(Xinit)
        self.pca = PCA().fit(Xinit_scaled)

        # Select Best PC components
        n_comp = self.select_best_PCA_comp(self.pca.explained_variance_)

        # Compute Q-statistics (SPE)
        Q = self.compute_outlier_score(Xinit_scaled, n_comp, score_type='SPE')
        threshold = self.compute_threshold(Q)

        # Evaluate Anomalies
        self.Y = self.evaluate_anomaly(Q, threshold)
        self.fit_best_GMM()

        # Append results
        self.ncomp = [n_comp for _ in range(self.N_Xinit)]  # n_comp
        self.resi_Q = Q  # Residuals
        self.threshold = [threshold for _ in range(self.N_Xinit)]  # threshold

    def fit_online_pca(self, Xnew):
        # Check anomality using outlier score from PCA built by previous step
        Xnew_scaled = self.pca_scaler.transform(Xnew)
        Qnew = self.compute_outlier_score(
            Xnew_scaled, self.ncomp[-1], score_type='SPE')

        # Evaluate abnomality
        self.X = np.concatenate([self.X, Xnew], axis=0)
        if Qnew <= self.threshold[-1]:  # normal sample
            self.n_stall_count = 0
            self.Y = np.hstack((self.Y, 0))

            # Fitting PCA based on GMM results
            # Fit PCA sklearn's function
            if not self.gmm_run:
                X1 = self.X[np.where(self.Y == 0)[0]]
                X1_scaled = self.pca_scaler.fit_transform(X1)
                self.pca = PCA().fit(X1_scaled)

                # Select Best PC components
                n_comp = self.select_best_PCA_comp(
                    self.pca.explained_variance_)

                # Compute Q-statistics (SPE)
                Q_thresh = self.compute_outlier_score(
                    X1_scaled, n_comp, score_type='SPE')
                threshold = self.compute_threshold(Q_thresh)

            else:
                # Baseline Model Update
                # Blockwise liniearization using GMM built by previous step
                X1 = self.X[np.where(self.Y == 0)[0]]
                X_scaled = self.gmm_scaler.transform(X1)
                self.fit_best_GMM()
                pred_class = self.gmm.predict(X_scaled)
                ind = np.where(pred_class == pred_class[-1])[0]

                # Fitting PCA based on GMM results
                # Fit PCA sklearn's function
                X1_scaled = self.pca_scaler.fit_transform(X1[ind])
                self.pca = PCA().fit(X1_scaled)

                # Select Best PC components
                n_comp = self.select_best_PCA_comp(
                    self.pca.explained_variance_)

                # Compute Q-statistics (SPE)
                Q_thresh = self.compute_outlier_score(
                    X1_scaled, n_comp, score_type='SPE')
                threshold = self.compute_threshold(Q_thresh)

            # Append results
            self.ncomp.append(n_comp)  # n_comp
            self.resi_Q = np.hstack((self.resi_Q, Qnew))  # Residuals
            self.threshold.append(threshold)  # threshold

        else:  # abnormal sample
            self.n_stall_count += 1
            self.Y = np.hstack((self.Y, 1))

            # Append results
            self.ncomp.append(self.ncomp[-1])  # n_comp
            self.resi_Q = np.hstack(
                (self.resi_Q, Qnew))  # Residuals
            self.threshold.append(self.threshold[-1])  # threshold

    def plot_result(self, title='None', ground_truth=[]):
        ind_axs = 0
        if type(ground_truth) == list:
            plt.figure(figsize=(10, 4), dpi=200)
            plt.plot(
                np.arange(0, self.resi_Q.shape[0]), self.resi_Q, 'k.')
            plt.plot(np.arange(0, self.N_Xinit),
                     self.resi_Q[:self.N_Xinit], 'b.', label='Initial X')
            ind_abnormal = np.where(self.Y == 1)[0]
            plt.plot(ind_abnormal,
                     self.resi_Q[ind_abnormal], 'r.', label='Abnormaly')
            plt.plot(np.arange(0, len(self.threshold)),
                     self.threshold, 'k:', label='Threshold')
            plt.ylabel('Anomaly Score')
            plt.grid(ls=':', color='gray')
            plt.legend(loc='best')
            plt.title(title)
            
        else:
            fig, axs = plt.subplots(2, 1, figsize=(10, 8), dpi=200)
            axs[ind_axs].scatter(
                ground_truth.index, ground_truth['label'], c=ground_truth['label'])
            event_index = []
            for event_int in np.unique(ground_truth['label']):
                event_index.append(
                    np.where(ground_truth.values == event_int)[0][0])

            for ind in event_index:
                axs[ind_axs].axvline(
                    x=ind, color='r', linestyle='--', linewidth=1)
            axs[ind_axs].grid(ls=':', color='gray')
            axs[ind_axs].set_yticklabels(
                (['o', 'Normal'] + [f'Shift#{ind + 1}' for ind in range(8)]))
            axs[ind_axs].set_ylabel('Event')

            ind_axs += 1
            axs[ind_axs].plot(
                np.arange(0, self.resi_Q.shape[0]), self.resi_Q, 'k.')
            axs[ind_axs].plot(np.arange(0, self.N_Xinit),
                              self.resi_Q[:self.N_Xinit], 'b.', label='Initial X')
            ind_abnormal = np.where(self.Y == 1)[0]
            axs[ind_axs].plot(ind_abnormal,
                              self.resi_Q[ind_abnormal], 'r.', label='Abnormaly')
            axs[ind_axs].plot(np.arange(0, len(self.threshold)),
                              self.threshold, 'k:', label='Threshold')
            for ind in event_index:
                axs[ind_axs].axvline(
                    x=ind, color='r', linestyle='--', linewidth=1)
            axs[ind_axs].set_ylabel('Anomaly Score')
            axs[ind_axs].grid(ls=':', color='gray')
            axs[ind_axs].legend(loc='best')
            axs[ind_axs].set_title(title)


        plt.show()


    def evaluate_anomaly(self, Q, threshold):
        Y = np.zeros_like(Q)
        Y[Q > threshold] = 1
        return Y

    def compute_outlier_score(self, X, n_comp, score_type='SPE', component_use='major'):
        if score_type is 'SPE':  # Q-statistics
            if component_use is 'major':
                mat_projection = self.pca.components_[:n_comp]
            else:
                mat_projection = self.pca.components_[n_comp:]

            X_pca = (X - self.pca.mean_).dot(mat_projection.T)
            X_projected = X_pca.dot(mat_projection) + self.pca.mean_
            score = np.sqrt(np.sum((X - X_projected)
                                   ** 2, axis=-1))
        else:  # T2-statistics
            X.dot(self.pca.components_[:n_comp].T)

        return score

    def select_best_PCA_comp(self, explVar):
        '''
            cumulative percent variance (CPV) method
        '''
        if self.method_Ncomp == 'CPV':
            # Get the PC scores and the eigenspectrum
            explVar = explVar / sum(explVar)
            n_comp = explVar[(np.cumsum(explVar) <= self.cutoff)].shape[0]

            # eigengap
            # % ref.1) The rotation of eigenvectors by a perturbation (1970)
            # % ref.2) Adaptive data-derived anomaly detection in the activated... (2016)
        else:
            n_comp = np.argmax(np.abs(np.diff(explVar)))

        if n_comp == 0:
            n_comp = 1

        return n_comp

    def compute_threshold(self, score):
        if self.method_thresh == 'percentile':
            pct_threshold = np.percentile(
                score, self.alpha * 100)  # Significnace level

        else: # Theoretical Threhold
            theta1 = np.mean(score)
            theta2 = np.var(score)
            h = 2 * (theta1 ** 2) / theta2
            chi_h = chi2.ppf(self.alpha, df=h)
            pct_threshold = theta2/2/theta1 * chi_h

        return pct_threshold

    def fit_best_GMM(self):
        # Check whether it is an initial or on-line learning
        ind_normal = np.where(self.Y == 0)[0]
        X = self.X[ind_normal]
        X_scaled = self.gmm_scaler.fit_transform(X)

        # If # feature is one, "cv_type" should be "Uni"
        if X.shape[1] == 1:
            cv_types = 'Uni'
        else:
            cv_types = self.gmm_cv_types

        lowest_bic = np.infty
        fitted_gmms, bic = [], []
        n_components_range = range(self.gmm_n_comp_min, self.gmm_n_comp_max)
        cv_types = cv_types

        for cv_type in cv_types:
            for n_components in n_components_range:
                # Fit a Gaussian mixture with EM
                if X.shape[1] == 1:
                    gmm = mixture.GaussianMixture(n_components=n_components)
                else:
                    gmm = mixture.GaussianMixture(
                        n_components=n_components, covariance_type=cv_type)
                gmm.fit(X_scaled)
                fitted_gmms.append(gmm)
                bic.append(gmm.bic(X_scaled))
                if bic[-1] < lowest_bic:
                    lowest_bic = bic[-1]
                    best_gmm = gmm

        self.gmm = best_gmm
