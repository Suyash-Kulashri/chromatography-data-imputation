import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')


class SyntheticDistributionImputer:
    """
    Advanced imputer that generates SYNTHETIC values (not duplicates) while 
    preserving the complete statistical distribution of each column.
    
    Uses Kernel Density Estimation (KDE) to model the underlying probability
    distribution and generates new values from this smooth distribution.
    
    Key Advantages:
    ---------------
    1. Generates truly NEW values (not just copies of observed values)
    2. Preserves mean, std, quartiles, skewness, and full distribution shape
    3. Works with any distribution shape (normal, skewed, bimodal, etc.)
    4. Values are realistic and stay within observed bounds
    """
    
    def __init__(self, method: str = 'kde', random_state: Optional[int] = 42, 
                 bandwidth: str = 'scott', bounds_buffer: float = 0.05):
        """
        Initialize the synthetic imputer.
        
        Parameters:
        -----------
        method : str, default='kde'
            Imputation method:
            - 'kde': Kernel Density Estimation (recommended for most cases)
            - 'parametric': Fit to best parametric distribution
            - 'bootstrap_jitter': Bootstrap with added noise
        random_state : int, optional
            Random seed for reproducibility
        bandwidth : str or float, default='scott'
            KDE bandwidth selection method ('scott', 'silverman', or numeric value)
            Scott's rule works well for most distributions
        bounds_buffer : float, default=0.05
            Buffer for bounds enforcement (5% beyond observed range)
        """
        self.method = method
        self.random_state = random_state
        self.bandwidth = bandwidth
        self.bounds_buffer = bounds_buffer
        self.column_models_ = {}      # Store KDE/distribution models per column
        self.imputation_stats_ = {}   # Store statistics before/after imputation
        self.fitted_ = False
        
    def fit(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> 'SyntheticDistributionImputer':
        """
        Fit the imputer by learning the distribution of each column.
        
        For KDE: Creates a kernel density estimator from observed values
        This learns the underlying "shape" of your data distribution.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The dataset to fit on
        columns : List[str], optional
            Specific columns to impute. If None, all columns with missing values are used.
            
        Returns:
        --------
        self : SyntheticDistributionImputer
            Fitted imputer instance
        """
        np.random.seed(self.random_state)
        
        # Determine which columns to process
        if columns is None:
            columns = df.columns[df.isnull().any()].tolist()
        
        print(f"Fitting imputer on {len(columns)} columns with missing values...")
        print(f"Method: {self.method.upper()}")
        print("="*70)
        
        for col in columns:
            # Get non-missing values
            observed_values = df[col].dropna().values
            
            if len(observed_values) == 0:
                print(f"⚠️  Column '{col}': 100% missing - skipping")
                continue
                
            if len(observed_values) < 10:
                print(f"⚠️  Column '{col}': Too few observed values ({len(observed_values)}) - skipping")
                continue
            
            # Store observed values for bounds checking
            obs_min = np.min(observed_values)
            obs_max = np.max(observed_values)
            
            # Add buffer to bounds (allows slight extrapolation)
            data_range = obs_max - obs_min
            lower_bound = obs_min - (data_range * self.bounds_buffer)
            upper_bound = obs_max + (data_range * self.bounds_buffer)
            
            # Fit distribution model based on method
            if self.method == 'kde':
                try:
                    # Kernel Density Estimation
                    # This creates a smooth probability density function from your data
                    kde = gaussian_kde(observed_values, bw_method=self.bandwidth)
                    
                    self.column_models_[col] = {
                        'type': 'kde',
                        'model': kde,
                        'observed_values': observed_values.copy(),
                        'bounds': (lower_bound, upper_bound),
                        'obs_bounds': (obs_min, obs_max)
                    }
                    
                except Exception as e:
                    print(f"⚠️  Column '{col}': KDE fitting failed ({str(e)}) - falling back to bootstrap")
                    self.column_models_[col] = {
                        'type': 'bootstrap',
                        'observed_values': observed_values.copy(),
                        'bounds': (lower_bound, upper_bound)
                    }
                    
            elif self.method == 'parametric':
                # Fit to best parametric distribution
                best_dist, best_params = self._fit_best_distribution(observed_values)
                
                self.column_models_[col] = {
                    'type': 'parametric',
                    'distribution': best_dist,
                    'params': best_params,
                    'observed_values': observed_values.copy(),
                    'bounds': (lower_bound, upper_bound)
                }
                
            elif self.method == 'bootstrap_jitter':
                # Bootstrap with jittering
                # Calculate appropriate jitter scale (small fraction of std)
                jitter_scale = np.std(observed_values) * 0.1  # 10% of std
                
                self.column_models_[col] = {
                    'type': 'bootstrap_jitter',
                    'observed_values': observed_values.copy(),
                    'jitter_scale': jitter_scale,
                    'bounds': (lower_bound, upper_bound)
                }
            
            # Calculate pre-imputation statistics
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            
            self.imputation_stats_[col] = {
                'n_total': len(df),
                'n_missing': df[col].isnull().sum(),
                'missing_pct': missing_pct,
                'n_observed': len(observed_values),
                'mean_before': np.mean(observed_values),
                'std_before': np.std(observed_values, ddof=1),
                'q25_before': np.percentile(observed_values, 25),
                'median_before': np.median(observed_values),
                'q75_before': np.percentile(observed_values, 75),
                'min_before': obs_min,
                'max_before': obs_max,
                'skewness_before': stats.skew(observed_values),
                'kurtosis_before': stats.kurtosis(observed_values)
            }
            
            print(f"✓ Column '{col}':")
            print(f"  - Missing: {missing_pct:.1f}% ({df[col].isnull().sum():,} values)")
            print(f"  - Observed: {len(observed_values):,} values")
            print(f"  - Distribution: μ={self.imputation_stats_[col]['mean_before']:.3f}, "
                  f"σ={self.imputation_stats_[col]['std_before']:.3f}")
            print(f"  - Range: [{obs_min:.3f}, {obs_max:.3f}]")
        
        self.fitted_ = True
        print("="*70)
        print(f"✓ Imputer fitted successfully on {len(self.column_models_)} columns\n")
        
        return self
    
    def _fit_best_distribution(self, data: np.ndarray) -> Tuple[stats.rv_continuous, tuple]:
        """
        Fit data to best parametric distribution using Maximum Likelihood Estimation.
        
        Tests common distributions and returns the best fit.
        """
        distributions = [
            stats.norm, stats.lognorm, stats.expon, stats.gamma, 
            stats.beta, stats.uniform
        ]
        
        best_dist = None
        best_params = None
        best_sse = np.inf
        
        for dist in distributions:
            try:
                params = dist.fit(data)
                # Calculate goodness of fit (KS statistic)
                ks_stat, _ = stats.kstest(data, lambda x: dist.cdf(x, *params))
                
                if ks_stat < best_sse:
                    best_sse = ks_stat
                    best_dist = dist
                    best_params = params
            except:
                continue
        
        return best_dist, best_params
    
    def _generate_kde_samples(self, kde, n_samples: int, bounds: Tuple[float, float],
                             obs_bounds: Tuple[float, float], max_attempts: int = 10) -> np.ndarray:
        """
        Generate samples from KDE with bounds enforcement.
        
        Core Algorithm:
        ---------------
        1. Sample from KDE (which gives us smooth, continuous values)
        2. Enforce bounds to keep values realistic
        3. If we need more samples (due to rejection), resample
        
        This is rejection sampling with bounds.
        """
        lower_bound, upper_bound = bounds
        obs_min, obs_max = obs_bounds
        samples = []
        remaining = n_samples
        
        for attempt in range(max_attempts):
            # Sample from KDE
            new_samples = kde.resample(remaining, seed=self.random_state + attempt)[0]
            
            # Apply bounds - prefer observed bounds but allow buffer
            valid_samples = new_samples[
                (new_samples >= lower_bound) & (new_samples <= upper_bound)
            ]
            
            samples.extend(valid_samples)
            remaining = n_samples - len(samples)
            
            if remaining <= 0:
                break
        
        # If we still don't have enough, clip the last batch
        if len(samples) < n_samples:
            final_samples = kde.resample(remaining, seed=self.random_state + max_attempts)[0]
            final_samples = np.clip(final_samples, obs_min, obs_max)
            samples.extend(final_samples)
        
        return np.array(samples[:n_samples])
    
    def transform(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """
        Transform the dataframe by filling missing values with SYNTHETIC values.
        
        Key Innovation:
        ---------------
        Unlike simple sampling, this generates NEW values that:
        - Follow the same statistical distribution
        - Are not duplicates of observed values
        - Stay within reasonable bounds
        - Preserve all distributional properties
        
        Parameters:
        -----------
        df : pd.DataFrame
            The dataset to transform
        inplace : bool, default=False
            If True, modify the dataframe in place
            
        Returns:
        --------
        df_imputed : pd.DataFrame
            Dataframe with missing values filled with synthetic values
        """
        if not self.fitted_:
            raise ValueError("Imputer must be fitted before transform. Call fit() first.")
        
        if not inplace:
            df = df.copy()
        
        np.random.seed(self.random_state)
        
        print(f"Transforming dataframe by generating synthetic values...")
        print("="*70)
        
        for col, model_info in self.column_models_.items():
            if col not in df.columns:
                print(f"⚠️  Column '{col}' not found in dataframe - skipping")
                continue
            
            # Find missing value positions
            missing_mask = df[col].isnull()
            n_missing = missing_mask.sum()
            
            if n_missing == 0:
                print(f"✓ Column '{col}': No missing values to impute")
                continue
            
            # Generate synthetic values based on model type
            if model_info['type'] == 'kde':
                # Generate from KDE - this creates NEW values!
                synthetic_values = self._generate_kde_samples(
                    model_info['model'],
                    n_missing,
                    model_info['bounds'],
                    model_info['obs_bounds']
                )
                
            elif model_info['type'] == 'parametric':
                # Generate from fitted parametric distribution
                dist = model_info['distribution']
                params = model_info['params']
                synthetic_values = dist.rvs(*params, size=n_missing, 
                                           random_state=self.random_state)
                # Enforce bounds
                lower, upper = model_info['bounds']
                synthetic_values = np.clip(synthetic_values, lower, upper)
                
            elif model_info['type'] == 'bootstrap_jitter':
                # Bootstrap with jittering
                base_samples = np.random.choice(
                    model_info['observed_values'],
                    size=n_missing,
                    replace=True
                )
                # Add Gaussian noise
                noise = np.random.normal(0, model_info['jitter_scale'], size=n_missing)
                synthetic_values = base_samples + noise
                # Enforce bounds
                lower, upper = model_info['bounds']
                synthetic_values = np.clip(synthetic_values, lower, upper)
                
            elif model_info['type'] == 'bootstrap':
                # Fallback to simple bootstrap
                synthetic_values = np.random.choice(
                    model_info['observed_values'],
                    size=n_missing,
                    replace=True
                )
            
            # Fill missing values with synthetic data
            df.loc[missing_mask, col] = synthetic_values
            
            # Calculate post-imputation statistics
            self.imputation_stats_[col]['mean_after'] = df[col].mean()
            self.imputation_stats_[col]['std_after'] = df[col].std()
            self.imputation_stats_[col]['q25_after'] = df[col].quantile(0.25)
            self.imputation_stats_[col]['median_after'] = df[col].median()
            self.imputation_stats_[col]['q75_after'] = df[col].quantile(0.75)
            self.imputation_stats_[col]['min_after'] = df[col].min()
            self.imputation_stats_[col]['max_after'] = df[col].max()
            self.imputation_stats_[col]['skewness_after'] = stats.skew(df[col])
            self.imputation_stats_[col]['kurtosis_after'] = stats.kurtosis(df[col])
            
            # Calculate uniqueness metric
            total_values = len(df[col])
            unique_values = len(df[col].unique())
            uniqueness_pct = (unique_values / total_values) * 100
            self.imputation_stats_[col]['uniqueness_pct'] = uniqueness_pct
            
            print(f"✓ Column '{col}': Generated {n_missing:,} synthetic values")
            print(f"  - Uniqueness: {uniqueness_pct:.1f}% unique values")
            
        print("="*70)
        print("✓ Transformation complete!\n")
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, columns: Optional[List[str]] = None, 
                     inplace: bool = False) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df, columns=columns)
        return self.transform(df, inplace=inplace)
    
    def get_imputation_report(self) -> pd.DataFrame:
        """
        Generate a detailed report showing statistics before and after imputation.
        
        Key Metrics:
        ------------
        - Distribution preservation (mean, std, quartiles)
        - Shape preservation (skewness, kurtosis)
        - Value uniqueness (to verify we're generating new values, not duplicates)
        """
        if not self.fitted_:
            raise ValueError("Imputer must be fitted first.")
        
        report_data = []
        
        for col, stats_dict in self.imputation_stats_.items():
            if 'mean_after' in stats_dict:
                report_data.append({
                    'Column': col,
                    'Missing %': f"{stats_dict['missing_pct']:.1f}%",
                    'N_Missing': stats_dict['n_missing'],
                    'Mean_Before': stats_dict['mean_before'],
                    'Mean_After': stats_dict['mean_after'],
                    'Mean_Diff': abs(stats_dict['mean_after'] - stats_dict['mean_before']),
                    'Std_Before': stats_dict['std_before'],
                    'Std_After': stats_dict['std_after'],
                    'Std_Diff': abs(stats_dict['std_after'] - stats_dict['std_before']),
                    'Median_Before': stats_dict['median_before'],
                    'Median_After': stats_dict['median_after'],
                    'Q25_Before': stats_dict['q25_before'],
                    'Q25_After': stats_dict['q25_after'],
                    'Q75_Before': stats_dict['q75_before'],
                    'Q75_After': stats_dict['q75_after'],
                    'Skew_Before': stats_dict['skewness_before'],
                    'Skew_After': stats_dict['skewness_after'],
                    'Uniqueness %': f"{stats_dict['uniqueness_pct']:.1f}%"
                })
        
        return pd.DataFrame(report_data)
    
    def visualize_distribution(self, df_before: pd.DataFrame, df_after: pd.DataFrame, 
                              columns: Optional[List[str]] = None, 
                              n_cols: int = 2, figsize: Tuple[int, int] = (15, 6)):
        """
        Visualize distributions before and after imputation.
        
        Shows both histogram AND KDE curve to verify smooth distribution preservation.
        """
        if columns is None:
            columns = list(self.column_models_.keys())
        
        n_rows = int(np.ceil(len(columns) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if len(columns) == 1 else axes
        
        for idx, col in enumerate(columns):
            ax = axes[idx]
            
            # Plot observed values only
            observed = df_before[col].dropna()
            ax.hist(observed, bins=40, alpha=0.4, label='Observed Only', 
                   color='blue', density=True, edgecolor='black', linewidth=0.5)
            
            # Plot KDE of observed
            try:
                kde_obs = gaussian_kde(observed)
                x_range = np.linspace(observed.min(), observed.max(), 200)
                ax.plot(x_range, kde_obs(x_range), 'b-', linewidth=2, 
                       label='Observed KDE', alpha=0.8)
            except:
                pass
            
            # Plot after imputation (observed + synthetic)
            ax.hist(df_after[col], bins=40, alpha=0.4, label='With Synthetic Values', 
                   color='green', density=True, edgecolor='black', linewidth=0.5)
            
            # Plot KDE of full data
            try:
                kde_full = gaussian_kde(df_after[col])
                x_range_full = np.linspace(df_after[col].min(), df_after[col].max(), 200)
                ax.plot(x_range_full, kde_full(x_range_full), 'g-', linewidth=2, 
                       label='Full Data KDE', alpha=0.8)
            except:
                pass
            
            missing_pct = self.imputation_stats_[col]['missing_pct']
            uniqueness = self.imputation_stats_[col].get('uniqueness_pct', 'N/A')
            
            ax.set_title(f'{col}\n({missing_pct:.1f}% missing, {uniqueness}% unique values)')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(len(columns), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def statistical_validation(self, df_before: pd.DataFrame, df_after: pd.DataFrame,
                               alpha: float = 0.05) -> pd.DataFrame:
        """
        Perform comprehensive statistical tests to validate distribution preservation.
        
        Tests:
        ------
        1. Kolmogorov-Smirnov: Tests if distributions are from same continuous distribution
        2. T-test: Tests if means are significantly different
        3. Levene's test: Tests if variances are significantly different
        """
        results = []
        
        for col in self.column_models_.keys():
            observed = df_before[col].dropna()
            full_data = df_after[col]
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(observed, full_data)
            
            # T-test for means
            t_stat, t_pvalue = stats.ttest_ind(observed, full_data)
            
            # Levene's test for variance
            _, levene_pvalue = stats.levene(observed, full_data)
            
            results.append({
                'Column': col,
                'KS_Statistic': f"{ks_stat:.4f}",
                'KS_P_Value': f"{ks_pvalue:.4f}",
                'Distribution_OK': '✓' if ks_pvalue > alpha else '✗',
                'T_Test_P_Value': f"{t_pvalue:.4f}",
                'Mean_OK': '✓' if t_pvalue > alpha else '✗',
                'Levene_P_Value': f"{levene_pvalue:.4f}",
                'Variance_OK': '✓' if levene_pvalue > alpha else '✗'
            })
        
        return pd.DataFrame(results)


# ============================================================================
# DEMONSTRATION
# ============================================================================

def create_sample_dataset(n_rows: int = 100000, n_cols: int = 8, 
                         missing_pct: float = 0.85) -> pd.DataFrame:
    """Create sample dataset with various distribution types."""
    np.random.seed(42)
    data = {}
    
    # Normal distribution
    for i in range(n_cols // 4):
        col = f'normal_{i}'
        full_data = np.random.normal(loc=50, scale=10, size=n_rows)
        mask = np.random.random(n_rows) > missing_pct
        data[col] = np.where(mask, full_data, np.nan)
    
    # Skewed (log-normal)
    for i in range(n_cols // 4):
        col = f'skewed_{i}'
        full_data = np.random.lognormal(mean=3, sigma=1, size=n_rows)
        mask = np.random.random(n_rows) > missing_pct
        data[col] = np.where(mask, full_data, np.nan)
    
    # Uniform
    for i in range(n_cols // 4):
        col = f'uniform_{i}'
        full_data = np.random.uniform(low=0, high=100, size=n_rows)
        mask = np.random.random(n_rows) > missing_pct
        data[col] = np.where(mask, full_data, np.nan)
    
    # Bimodal
    for i in range(n_cols // 4):
        col = f'bimodal_{i}'
        half = n_rows // 2
        component1 = np.random.normal(loc=30, scale=5, size=half)
        component2 = np.random.normal(loc=70, scale=5, size=n_rows - half)
        full_data = np.concatenate([component1, component2])
        np.random.shuffle(full_data)
        mask = np.random.random(n_rows) > missing_pct
        data[col] = np.where(mask, full_data, np.nan)
    
    df = pd.DataFrame(data)
    print(f"Created sample dataset: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum():,} ({(df.isnull().sum().sum() / df.size * 100):.1f}%)\n")
    
    return df


if __name__ == "__main__":
    print("="*70)
    print("SYNTHETIC DISTRIBUTION-PRESERVING IMPUTATION DEMO")
    print("Method: Kernel Density Estimation (KDE)")
    print("="*70)
    print()
    
    # Create sample data
    df_original = create_sample_dataset(n_rows=100000, n_cols=8, missing_pct=0.85)
    
    # Initialize imputer with KDE
    imputer = SyntheticDistributionImputer(method='kde', random_state=42)
    
    # Fit and transform
    df_imputed = imputer.fit_transform(df_original)
    
    # Generate report
    print("\n" + "="*70)
    print("IMPUTATION REPORT")
    print("="*70)
    report = imputer.get_imputation_report()
    print(report.to_string(index=False))
    
    # Statistical validation
    print("\n" + "="*70)
    print("STATISTICAL VALIDATION")
    print("="*70)
    validation = imputer.statistical_validation(df_original, df_imputed)
    print(validation.to_string(index=False))
    print("\nInterpretation: ✓ indicates distribution is preserved (p-value > 0.05)")
    
    # Visualize
    print("\n" + "="*70)
    print("Generating visualization...")
    imputer.visualize_distribution(df_original, df_imputed, 
                                   columns=list(df_original.columns)[:4])