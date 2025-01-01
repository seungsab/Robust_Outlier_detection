from sklearn.decomposition import PCA
from sklearn import mixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from scipy.stats.distributions import chi2


class pca_online():

    def __init__(self, component_use='major', gmm_run=1, n_sample_max=1e4, method_Ncomp='eigengap', method_thresh='percentile', cutoff=0.95, n_stall=5, alpha=0.99, n_comp_min=1, n_comp_max=3, cv_types=["spherical", "tied", "diag", "full"], score_type='SPE', kmeans_n_clusters_min=1, kmeans_n_clusters_max=5):
        # BASELINE MODEL
        self.pca, self.gmm = [], []  # PCA model and GMM for Block-linearization

        # SAMPLE INVENTORY
        # All samples including normal and abnormal samples
        self.X, self.Y = [], []

        # New samples of being evaluated
        self.Xnew, self.Ynew = [], []

        # Residuals and indices of anomaly for all samples
        self.resi0_Q, self.resi0_T2 = [], []
        self.N_Xinit = []  # Initial number of training samples

        # ANOMALY PARAMETER
        # History of "# Comp." and "Threshold", residuals of all samples
        self.ncomp, self.threshold, self.resi_Q, self.resi_T2 = [], [], [], []
        self.pca_scaler = StandardScaler()  # Standardization
        self.method_Ncomp, self.method_thresh = method_Ncomp, method_thresh

        # Maximal number of samples, significance level, cutoff for optimal components
        self.n_sample_max, self.alpha, self.cutoff = n_sample_max, alpha, cutoff
        # Tolerance for alarming, counts of consecutive violations
        self.n_stall, self.n_stall_count = n_stall, 0
        self.component_use = component_use
        self.score_type = score_type

        # GMM PARAMETERS
        self.gmm_scaler = StandardScaler()  # Standardization
        self.gmm_n_comp_min = n_comp_min  # Minimal # of Gaussian components
        self.gmm_n_comp_max = n_comp_max  # Maximal # of Gaussian components
        self.gmm_cv_types = cv_types  # Covariance type
        self.kmeans_run = gmm_run

        # K-means PARAMETERS 추가
        self.kmeans_n_clusters_min = kmeans_n_clusters_min
        self.kmeans_n_clusters_max = kmeans_n_clusters_max


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
        score_type = self.score_type
        component_use = self.component_use
        Q = self.compute_outlier_score(Xinit_scaled, n_comp, score_type, component_use)
        threshold = self.compute_threshold(Q)

        # Evaluate Anomalies
        self.Y = self.evaluate_anomaly(Q, threshold)
        self.fit_best_kmeans()

        # Append results
        self.ncomp = [n_comp for _ in range(self.N_Xinit)]  # n_comp
        self.resi_Q = Q  # Residuals
        self.threshold = [threshold for _ in range(self.N_Xinit)]  # threshold

    def fit_online_pca(self, Xnew):
        # Check anomality using outlier score from PCA built by previous step

        score_type = self.score_type
        component_use = self.component_use

        Xnew_scaled = self.pca_scaler.transform(Xnew)
        Qnew = self.compute_outlier_score(
            Xnew_scaled, self.ncomp[-1], score_type, component_use)

        # Evaluate abnormality
        self.X = np.concatenate([self.X, Xnew], axis=0)
        if Qnew <= self.threshold[-1]:  # normal sample
            self.n_stall_count = 0
            self.Y = np.hstack((self.Y, 0))

            # Fitting PCA based on clustering results
            # Fit PCA sklearn's function
            if not self.kmeans_run:
                X1 = self.X[np.where(self.Y == 0)[0]]
                X1_scaled = self.pca_scaler.fit_transform(X1)
                self.pca = PCA().fit(X1_scaled)

                # Select Best PC components
                n_comp = self.select_best_PCA_comp(
                    self.pca.explained_variance_)

                # Compute Q-statistics (SPE)
                Q_thresh = self.compute_outlier_score(
                    X1_scaled, n_comp, score_type, component_use)
                threshold = self.compute_threshold(Q_thresh)

            else:
                # Baseline Model Update
                # Blockwise linearization using K-means built by previous step
                X1 = self.X[np.where(self.Y == 0)[0]]
                X_scaled = self.gmm_scaler.transform(X1)
                self.fit_best_kmeans()
                pred_class = self.kmeans.predict(X_scaled)
                ind = np.where(pred_class == pred_class[-1])[0]

                # Fitting PCA based on K-means results
                # Fit PCA sklearn's function
                X1_scaled = self.pca_scaler.fit_transform(X1[ind])
                self.pca = PCA().fit(X1_scaled)

                # Select Best PC components
                n_comp = self.select_best_PCA_comp(
                    self.pca.explained_variance_)

                # Compute Q-statistics (SPE)
                Q_thresh = self.compute_outlier_score(
                    X1_scaled, n_comp, score_type, component_use)
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

    def plot_result(self, title='None', ground_truth=[], out=[]):
        plt.figure(figsize=(10, 4), dpi=200)
        plt.plot(np.arange(0, self.resi_Q.shape[0]), self.resi_Q, 'k.')
        plt.plot(np.arange(0, self.N_Xinit), self.resi_Q[:self.N_Xinit], 'b.', label='Initial X')
        ind_abnormal = np.where(self.Y == 1)[0]
        plt.plot(ind_abnormal, self.resi_Q[ind_abnormal], 'r.', label='Anomaly')
        plt.plot(np.arange(0, len(self.threshold)), self.threshold, 'k:', label='Threshold')
        plt.ylabel('Anomaly Score')
        plt.grid(ls=':', color='gray')
        plt.legend(loc='best')
        plt.title(title)

        event_index = []
        if isinstance(ground_truth, pd.DataFrame):
            for event_int in np.unique(ground_truth['label']):
                event_index.append(np.where(ground_truth.values == event_int)[0][0])
        else:
            event_index = ground_truth

        # Training 부분 점선 그리기
        for ind in event_index:
            plt.axvline(x=ind, color='b', linestyle='--', linewidth=1)

        # out을 넘파이 배열로 변환하고 1차원으로 변경
        out = np.array(out).flatten()

        # 붉은색 점선 그리기
        out_point = out[0]
        plt.axvline(x=out_point, color='r', linestyle='--', linewidth=1)
        # out 이하의 ind_abnormal 개수 출력
        num_abnormal_before_out = len(ind_abnormal[ind_abnormal < out_point])
        print("Number of abnormal points before out:", num_abnormal_before_out)
        # out 이상의 ind_abnormal 개수 출력
        num_abnormal_after_out = len(ind_abnormal[ind_abnormal >= out_point])
        print("Number of abnormal points after out:", num_abnormal_after_out)
        # out 이하와 이상의 전체 포인트 수 출력
        num_total_points_before_out = out_point - 1
        num_total_points_after_out = len(self.resi_Q) - out_point
        print("Total number of points before out:", num_total_points_before_out)
        print("Total number of points after out:", num_total_points_after_out)

        # 전체 포인트 수와 abnormal 포인트 수 출력
        num_total_points = len(self.resi_Q)
        print("Total number of points:", num_total_points)
        print("Total number of abnormal points:", len(ind_abnormal))

        # 오탐지율 계산
        false_positive_rate = (num_abnormal_before_out + (num_total_points_after_out - num_abnormal_after_out)) / num_total_points
        print("False Positive Rate (오탐지율):", false_positive_rate * 100)

        plt.show()

        # ind_abnormal을 기준으로 resi_Q를 합치기
        abnormal_resi_Q = np.concatenate([self.resi_Q[ind_abnormal], np.full(self.resi_Q.shape[0] - len(ind_abnormal), np.nan)])

        # 데이터를 데이터프레임으로 변환
        data = {
            'resi_Q': self.resi_Q,
            'threshold': self.threshold,
            'abnormal_resi_Q': abnormal_resi_Q  # ind_abnormal을 기준으로 합친 데이터
        }

        df = pd.DataFrame(data)

        # 엑셀 파일로 저장
        excel_file_path = 'self_data.xlsx'
        df.to_excel(excel_file_path, index=False)

    def evaluate_anomaly(self, Q, threshold):
        Y = np.zeros_like(Q)
        Y[Q > threshold] = 1
        return Y

    def compute_outlier_score(self, X, n_comp, score_type, component_use):

        if component_use == 'major':
            mat_projection = self.pca.components_[:n_comp]
        else:
            mat_projection = self.pca.components_[n_comp:]

        if score_type == 'SPE':  # Q-statistics
            X_pca = (X - self.pca.mean_).dot(mat_projection.T)
            X_projected = X_pca.dot(mat_projection) + self.pca.mean_
            score = np.sqrt(np.sum((X - X_projected) ** 2, axis=-1))
        else:  # T2-statistics
            X_pca = X.dot(mat_projection.T)
            X_projected = X_pca.dot(mat_projection)
            score = np.sum(X_projected ** 2, axis=-1)
#             X.dot(self.pca.components_[:n_comp].T)

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
                score, self.alpha * 100)  # Significance level

        else:  # Theoretical Threshold
            theta1 = np.mean(score)
            theta2 = np.var(score)
            h = 2 * (theta1 ** 2) / theta2
            chi_h = chi2.ppf(self.alpha, df=h)
            pct_threshold = theta2 / 2 / theta1 * chi_h

        return pct_threshold

    def fit_best_kmeans(self):
        # Check whether it is an initial or on-line learning
        ind_normal = np.where(self.Y == 0)[0]
        X = self.X[ind_normal]
        X_scaled = self.gmm_scaler.fit_transform(X)

        # Set k to 2
        kmeans_history = []
        kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X_scaled)
        kmeans_history.append(kmeans)
    
        # Save the best K-means model
        self.kmeans = kmeans

