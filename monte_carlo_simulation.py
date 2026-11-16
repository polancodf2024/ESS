import numpy as np
import pandas as pd
from scipy import stats
import time

def monte_carlo_simulation():
    print("Starting Monte Carlo Simulation (B=10,000)...")
    print("This may take a few minutes depending on your CPU.\n")
    
    # Simulation parameters (as in the Appendix)
    B = 10000  # Number of replicates
    n_values = [5, 10, 20, 40]
    m_values = [5, 20]
    rho_values = [0.1, 0.5, 0.8]
    
    sigma2 = 1.0
    mu_control = 0.0
    mu_hf = 0.0  # Under H0, means are equal
    
    results = []
    
    start_time = time.time()
    
    for n in n_values:
        for m in m_values:
            for rho in rho_values:
                rejections_std = 0
                rejections_eff = 0
                
                # Compute parameters
                sigma2_u = rho * sigma2
                sigma2_e = (1 - rho) * sigma2
                N_nominal = n * m
                deff = 1 + (m - 1) * rho
                N_eff = N_nominal / deff
                df_eff = max(1, N_eff - 2) # Avoid negative or zero degrees of freedom
                
                for _ in range(B):
                    # Generate data for Control
                    u_c = np.random.normal(0, np.sqrt(sigma2_u), n)
                    y_c = mu_control + np.repeat(u_c, m) + np.random.normal(0, np.sqrt(sigma2_e), N_nominal)
                    
                    # Generate data for HF
                    u_h = np.random.normal(0, np.sqrt(sigma2_u), n)
                    y_h = mu_hf + np.repeat(u_h, m) + np.random.normal(0, np.sqrt(sigma2_e), N_nominal)
                    
                    # 1. Standard t-test
                    t_stat, p_val = stats.ttest_ind(y_c, y_h, equal_var=False) # Welch is more robust
                    if p_val < 0.05:
                        rejections_std += 1
                    
                    # 2. ESS-corrected t-test
                    diff = np.mean(y_h) - np.mean(y_c)
                    # Pooled variance
                    var_pooled = (np.var(y_c, ddof=1) + np.var(y_h, ddof=1)) / 2
                    se_eff = np.sqrt(var_pooled * (2 / N_eff))
                    
                    if se_eff > 0:
                        t_eff = diff / se_eff
                        p_eff = 2 * stats.t.sf(np.abs(t_eff), df=df_eff)
                        if p_eff < 0.05:
                            rejections_eff += 1
                            
                # Type I Error rate (proportion of false positives)
                rate_std = rejections_std / B
                rate_eff = rejections_eff / B
                
                results.append({
                    'n': n,
                    'm': m,
                    'rho': rho,
                    'Type_I_Error_Std': rate_std,
                    'Type_I_Error_Eff': rate_eff
                })
                
                print(f"Completed: n={n:2d}, m={m:2d}, rho={rho:.1f} | "
                      f"Type I Error Std: {rate_std:.3f} | "
                      f"Type I Error Eff: {rate_eff:.3f}")

    end_time = time.time()
    print(f"\nSimulation finished in {end_time - start_time:.2f} seconds.")
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv('ess_simulation_results.csv', index=False)
    print("Results saved to 'ess_simulation_results.csv'")
    
    print("\n--- RESULTS TABLE (Type I Error Rate) ---")
    print(df.to_string(index=False, float_format="%.3f"))

if __name__ == "__main__":
    monte_carlo_simulation()
