import numpy as np
from scipy import stats

def single_experiment():
    print("=====================================================")
    print(" SINGLE EXPERIMENT: A SMALL TEST")
    print("=====================================================")
    
    # 1. Experiment parameters
    n = 10           # Number of rats per group
    m = 20           # Number of cells per rat
    rho = 0.50       # Intraclass correlation coefficient (ICC)
    sigma2 = 1.0     # Total variance
    mu_control = 0.0 # Mean of the control group
    mu_hf = 0.3      # Mean of the HF group (true difference of 0.3 sigma)
    
    # Compute variances from ICC
    sigma2_u = rho * sigma2               # Between-rat variance
    sigma2_e = (1 - rho) * sigma2         # Within-rat (error) variance
    
    # 2. Data simulation
    np.random.seed(42) # Seed for reproducibility
    
    # Control group
    u_control = np.random.normal(0, np.sqrt(sigma2_u), n)
    y_control = mu_control + np.repeat(u_control, m) + np.random.normal(0, np.sqrt(sigma2_e), n*m)
    
    # HF group
    u_hf = np.random.normal(0, np.sqrt(sigma2_u), n)
    y_hf = mu_hf + np.repeat(u_hf, m) + np.random.normal(0, np.sqrt(sigma2_e), n*m)
    
    # 3. Erroneous analysis (standard t-test assuming independence)
    t_stat_std, p_val_std = stats.ttest_ind(y_control, y_hf, equal_var=True)
    
    # 4. Correct analysis (ESS correction)
    N_nominal = n * m
    deff = 1 + (m - 1) * rho
    N_eff = N_nominal / deff
    
    # Corrected degrees of freedom
    df_eff = N_eff - 2 
    
    # Compute mean difference and corrected standard error
    mean_diff = np.mean(y_hf) - np.mean(y_control)
    
    # Pooled variance using the correction
    var_pooled_std = (np.var(y_control, ddof=1) + np.var(y_hf, ddof=1)) / 2
    se_std = np.sqrt(var_pooled_std * (2/N_nominal)) # Standard error assuming independence
    se_eff = np.sqrt(var_pooled_std * (2/N_eff))     # Corrected standard error
    
    t_stat_eff = mean_diff / se_eff
    p_val_eff = 2 * stats.t.sf(np.abs(t_stat_eff), df=df_eff)
    
    # 5. Display results
    print(f"Parameters: n={n} rats/group, m={m} cells/rat, ICC={rho}")
    print(f"Observed mean difference: {mean_diff:.3f}")
    print(f"Design effect (deff): {deff:.2f}")
    print(f"Nominal size: {N_nominal}, Effective Size (ESS): {N_eff:.1f}\n")
    
    print("--- RESULTS ---")
    print(f"1. Erroneous Analysis (Ignoring Clustering):")
    print(f"   Standard Error: {se_std:.4f}")
    print(f"   t-statistic:    {t_stat_std:.2f}")
    print(f"   P-value:        {p_val_std:.4f}  -> {'REJECTS H0 (False Positive)' if p_val_std < 0.05 else 'Does not reject H0'}")
    
    print(f"\n2. Correct Analysis (Effective Sample Size):")
    print(f"   Standard Error: {se_eff:.4f}")
    print(f"   Degrees of Freedom: {df_eff:.1f}")
    print(f"   t-statistic:    {t_stat_eff:.2f}")
    print(f"   P-value:        {p_val_eff:.4f}  -> {'REJECTS H0' if p_val_eff < 0.05 else 'Does not reject H0 (Correct)'}")
    print("=====================================================")

if __name__ == "__main__":
    single_experiment()
