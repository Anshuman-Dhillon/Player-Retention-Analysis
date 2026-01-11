"""
Causal analysis module using DoWhy and EconML for treatment effect estimation.
Adapted for gaming user retention analysis.
"""
import pandas as pd
import numpy as np
from dowhy import CausalModel
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class CausalAnalyzer:
    """Performs causal inference on user retention data."""
    
    def __init__(self):
        self.model = None
        self.treatment_effect = None
    
    def prepare_data(self, df: pd.DataFrame, 
                    confounders: list = None,
                    treatment_col: str = 'treatment',
                    outcome_col: str = 'retained',
                    engagement_col: str = 'engagement_score') -> pd.DataFrame:
        """
        Prepare data for causal analysis.
        
        Args:
            df: Raw data
            confounders: List of confounder column names (auto-detected if None)
            treatment_col: Name of treatment column
            outcome_col: Name of binary outcome column
            engagement_col: Name of continuous outcome column
            
        Returns:
            Prepared DataFrame
        """
        print("\nPreparing data for causal analysis...")
        
        # Make a copy
        prepared = df.copy()
        
        # Ensure required columns exist
        if treatment_col not in prepared.columns:
            raise ValueError(f"Treatment column '{treatment_col}' not found")
        if outcome_col not in prepared.columns:
            raise ValueError(f"Outcome column '{outcome_col}' not found")
        
        # Get the variable used to create treatment (if any)
        treatment_creator = df.attrs.get('treatment_creator', None)
        
        # Auto-detect confounders if not specified
        if confounders is None:
            # Use numeric columns that aren't treatment/outcome and drop identifiers
            exclude_cols = [treatment_col, outcome_col, engagement_col]
            
            # CRITICAL: Also exclude the variable used to CREATE treatment
            if treatment_creator:
                exclude_cols.append(treatment_creator)
                print(f"⚠️ Excluding '{treatment_creator}' from confounders (used to create treatment)")
            
            candidate_confounders = [col for col in prepared.select_dtypes(include=[np.number]).columns 
                          if col not in exclude_cols]

            # Remove identifier / high-cardinality columns (e.g., PlayerID) and zero-variance cols
            confounders = []
            for col in candidate_confounders:
                nunique = prepared[col].nunique()
                if nunique <= 1:
                    print(f"  Skipping {col}: zero variance (only {nunique} unique value)")
                    continue
                # consider a column an identifier if unique values ~= number of rows
                if nunique >= 0.95 * len(prepared):
                    print(f"  Skipping {col}: likely identifier ({nunique} unique values for {len(prepared)} rows)")
                    continue
                confounders.append(col)

            print(f"Auto-detected confounders: {confounders}")
        else:
            # Still validate user-provided confounders
            validated_confounders = []
            for col in confounders:
                # Skip treatment creator
                if treatment_creator and col == treatment_creator:
                    print(f"  Skipping {col}: used to create treatment (circular logic)")
                    continue
                if col not in prepared.columns:
                    print(f"  Warning: {col} not in dataframe, skipping")
                    continue
                nunique = prepared[col].nunique()
                if nunique <= 1:
                    print(f"  Skipping {col}: zero variance")
                    continue
                if nunique >= 0.95 * len(prepared):
                    print(f"  Skipping {col}: likely identifier")
                    continue
                validated_confounders.append(col)
            confounders = validated_confounders
        
        if not confounders:
            raise ValueError("No valid confounders found! Make sure your data has numeric columns besides treatment/outcome.")
        
        # Store confounder list
        prepared.attrs['confounders'] = confounders
        prepared.attrs['treatment_col'] = treatment_col
        prepared.attrs['outcome_col'] = outcome_col
        prepared.attrs['engagement_col'] = engagement_col
        
        # Clean data
        cols_needed = [treatment_col, outcome_col] + confounders
        if engagement_col in prepared.columns:
            cols_needed.append(engagement_col)
        
        prepared = prepared.dropna(subset=cols_needed)
        
        print(f"Prepared {len(prepared)} records for analysis")
        print(f"Treatment: {prepared[treatment_col].sum()} treated, {len(prepared) - prepared[treatment_col].sum()} control")
        print(f"Outcome: {prepared[outcome_col].mean()*100:.1f}% retention rate")
        
        return prepared
    
    def estimate_treatment_effect_dowhy(self, df: pd.DataFrame) -> dict:
        """
        Use DoWhy to estimate causal effect with proper identification.
        
        Args:
            df: Prepared data from prepare_data
            
        Returns:
            Dictionary with causal estimates
        """
        treatment_col = df.attrs.get('treatment_col', 'treatment')
        outcome_col = df.attrs.get('outcome_col', 'retained')
        confounders = df.attrs.get('confounders', [])
        
        # Build causal graph
        graph_edges = []
        # Treatment affects outcome
        graph_edges.append(f"{treatment_col} -> {outcome_col}")
        # Confounders affect both treatment and outcome
        for conf in confounders:
            graph_edges.append(f"{conf} -> {treatment_col}")
            graph_edges.append(f"{conf} -> {outcome_col}")
        
        causal_graph = "digraph { " + "; ".join(graph_edges) + " }"
        
        # Create causal model
        model = CausalModel(
            data=df,
            treatment=treatment_col,
            outcome=outcome_col,
            graph=causal_graph,
            common_causes=confounders if confounders else None
        )
        
        # Identify causal effect
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        print("\n" + "="*60)
        print("DoWhy CAUSAL IDENTIFICATION")
        print("="*60)
        print(identified_estimand)
        
        # Estimate effect using backdoor criterion
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.propensity_score_weighting"
        )
        
        print("\n" + "="*60)
        print("CAUSAL ESTIMATE")
        print("="*60)
        print(estimate)
        print("="*60)
        
        # Refutation tests
        print("\nRunning refutation tests...")
        refute_random = model.refute_estimate(
            identified_estimand,
            estimate,
            method_name="random_common_cause"
        )
        
        refute_placebo = model.refute_estimate(
            identified_estimand,
            estimate,
            method_name="placebo_treatment_refuter"
        )
        
        print(f"✓ Random common cause test: {refute_random.refutation_result}")
        print(f"✓ Placebo treatment test: {refute_placebo.refutation_result}")
        
        return {
            'estimate': estimate.value,
            'estimand': identified_estimand,
            'refutation_random': refute_random,
            'refutation_placebo': refute_placebo
        }
    
    def estimate_heterogeneous_effects(self, df: pd.DataFrame) -> np.ndarray:
        """
        Use EconML to estimate heterogeneous treatment effects.
        
        Args:
            df: Prepared data
            
        Returns:
            Array of individual treatment effects
        """
        treatment_col = df.attrs.get('treatment_col', 'treatment')
        engagement_col = df.attrs.get('engagement_col', 'engagement_score')
        confounders = df.attrs.get('confounders', [])
        
        if not confounders:
            raise ValueError("No confounders available for heterogeneous effect estimation")
        
        # Double-check confounders are valid (no identifiers, no zero variance)
        filtered = []
        for c in confounders:
            if df[c].nunique() <= 1:
                print(f"  Skipping {c}: zero variance")
                continue
            if df[c].nunique() >= 0.95 * len(df):
                print(f"  Skipping {c}: likely identifier")
                continue
            filtered.append(c)
        
        if len(filtered) < len(confounders):
            removed = set(confounders) - set(filtered)
            print(f"Removed problematic confounders: {list(removed)}")
        
        confounders = filtered
        
        if not confounders:
            raise ValueError("No valid confounders remain after filtering!")
        
        # Check for multicollinearity and remove highly correlated features
        print("\nChecking for multicollinearity...")
        X_temp = df[confounders].values
        
        # Calculate correlation matrix
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_temp)
        
        # Calculate VIF (Variance Inflation Factor) approximation using correlation
        corr_matrix = np.corrcoef(X_scaled.T)
        
        # Remove variables with very high correlation (>0.95) to others
        to_remove = set()
        for i in range(len(confounders)):
            for j in range(i+1, len(confounders)):
                if abs(corr_matrix[i, j]) > 0.95:
                    print(f"  High correlation ({corr_matrix[i, j]:.3f}) between {confounders[i]} and {confounders[j]}")
                    # Keep the first one, remove the second
                    to_remove.add(confounders[j])
        
        if to_remove:
            print(f"Removing highly correlated confounders: {list(to_remove)}")
            confounders = [c for c in confounders if c not in to_remove]
        
        if not confounders:
            raise ValueError("No confounders remain after removing multicollinearity!")
        
        # Prepare features with filtered confounders
        X = df[confounders].values
        T = df[treatment_col].values
        Y = df[engagement_col].values if engagement_col in df.columns else df['retained'].values
        
        print(f"\nEstimating heterogeneous effects using {len(confounders)} features...")
        print(f"Features: {confounders}")
        
        # Basic sanity checks
        if np.unique(T).size <= 1:
            raise ValueError("Treatment column must have more than one unique value to estimate effects")
        
        # Check for perfect separation or other numerical issues
        print(f"Treatment distribution: {np.bincount(T.astype(int))}")
        print(f"Outcome range: [{Y.min():.2f}, {Y.max():.2f}]")
        
        # Standardize features to help with numerical stability
        X = scaler.fit_transform(X)
        
        # Use Causal Forest with more robust settings
        est = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=20, random_state=42),
            model_t=RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=20, random_state=42),
            discrete_treatment=True,
            random_state=42,
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=20
        )
        
        # Fit the Causal Forest
        try:
            est.fit(Y, T, X=X)
        except np.linalg.LinAlgError as e:
            print("\n❌ Causal Forest estimation failed due to numerical issues.")
            print("Falling back to simpler estimation method...")
            
            # Fallback: Use simple conditional mean differences
            treatment_effects = self._fallback_treatment_effects(df, confounders, treatment_col, engagement_col)
            self.treatment_effect = treatment_effects
            return treatment_effects
        except Exception as e:
            print(f"\n❌ Unexpected error in Causal Forest: {e}")
            print("Falling back to simpler estimation method...")
            
            treatment_effects = self._fallback_treatment_effects(df, confounders, treatment_col, engagement_col)
            self.treatment_effect = treatment_effects
            return treatment_effects
        
        # Get treatment effects
        treatment_effects = est.effect(X)
        self.treatment_effect = treatment_effects
        
        print(f"✓ Estimated effects for {len(treatment_effects)} individuals")
        
        return treatment_effects
    
    def _fallback_treatment_effects(self, df: pd.DataFrame, confounders: list, 
                                   treatment_col: str, engagement_col: str) -> np.ndarray:
        """
        Fallback method for estimating treatment effects when Causal Forest fails.
        Uses a simpler approach: matching on confounders and calculating conditional effects.
        """
        from sklearn.neighbors import NearestNeighbors
        
        print("Using k-nearest neighbors matching for treatment effect estimation...")
        
        X = df[confounders].values
        T = df[treatment_col].values
        Y = df[engagement_col].values if engagement_col in df.columns else df['retained'].values
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        treatment_effects = np.zeros(len(df))
        
        # For each unit, find similar units from opposite treatment group
        treated_idx = np.where(T == 1)[0]
        control_idx = np.where(T == 0)[0]
        
        # Fit nearest neighbors for treated group
        if len(treated_idx) > 0 and len(control_idx) > 0:
            nn_treated = NearestNeighbors(n_neighbors=min(5, len(treated_idx)))
            nn_treated.fit(X_scaled[treated_idx])
            
            nn_control = NearestNeighbors(n_neighbors=min(5, len(control_idx)))
            nn_control.fit(X_scaled[control_idx])
            
            # Estimate effects for treated units
            for i in treated_idx:
                distances, indices = nn_control.kneighbors([X_scaled[i]])
                matched_control_outcomes = Y[control_idx[indices[0]]]
                treatment_effects[i] = Y[i] - matched_control_outcomes.mean()
            
            # Estimate effects for control units
            for i in control_idx:
                distances, indices = nn_treated.kneighbors([X_scaled[i]])
                matched_treated_outcomes = Y[treated_idx[indices[0]]]
                treatment_effects[i] = matched_treated_outcomes.mean() - Y[i]
        else:
            # If we can't match, use overall average effect
            avg_effect = Y[T == 1].mean() - Y[T == 0].mean() if len(treated_idx) > 0 and len(control_idx) > 0 else 0
            treatment_effects = np.full(len(df), avg_effect)
        
        print(f"✓ Estimated effects using matching for {len(treatment_effects)} individuals")
        return treatment_effects
    
    def analyze_subgroups(self, df: pd.DataFrame, treatment_effects: np.ndarray) -> pd.DataFrame:
        """
        Analyze which user subgroups benefit most from treatment.
        
        Args:
            df: Prepared data
            treatment_effects: Individual treatment effects
            
        Returns:
            DataFrame with subgroup analysis
        """
        df_analysis = df.copy()
        df_analysis['treatment_effect'] = treatment_effects
        
        # Create segments based on available data
        subgroups = []
        
        # Try different segmentation strategies
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['treatment', 'retained', 'engagement_score', 'treatment_effect']:
                try:
                    df_analysis[f'{col}_segment'] = pd.qcut(
                        df_analysis[col], 
                        q=4, 
                        labels=['Low', 'Medium', 'High', 'Very High'],
                        duplicates='drop'
                    )
                    subgroups.append(f'{col}_segment')
                except:
                    pass
        
        # Analyze each subgroup
        results = {}
        for segment in subgroups:
            group_effects = df_analysis.groupby(segment)['treatment_effect'].agg(['mean', 'std', 'count'])
            results[segment] = group_effects
        
        # Return the most interesting one (highest variance)
        if results:
            best_segment = max(results.keys(), key=lambda x: results[x]['std'].sum())
            print(f"\nMost heterogeneous subgroup: {best_segment}")
            return results[best_segment]
        else:
            print("⚠️ Could not create meaningful subgroups")
            return pd.DataFrame()
    
    def visualize_results(self, df: pd.DataFrame, treatment_effects: np.ndarray, 
                         save_path: str = 'causal_analysis_results.png'):
        """Create visualizations of causal analysis results."""
        treatment_col = df.attrs.get('treatment_col', 'treatment')
        outcome_col = df.attrs.get('outcome_col', 'retained')
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Treatment distribution
        df['treatment_label'] = df[treatment_col].map({0: 'Control', 1: 'Early Success'})
        sns.countplot(data=df, x='treatment_label', ax=axes[0, 0], palette=['#d62728', '#2ca02c'])
        axes[0, 0].set_title('Treatment Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('')
        
        # 2. Retention by treatment
        retention_by_treatment = df.groupby('treatment_label')[outcome_col].mean()
        retention_by_treatment.plot(kind='bar', ax=axes[0, 1], color=['#d62728', '#2ca02c'])
        axes[0, 1].set_title('Retention Rate by Treatment', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Retention Rate')
        axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=0)
        
        # 3. Treatment effect distribution
        axes[1, 0].hist(treatment_effects, bins=30, edgecolor='black', alpha=0.7, color='#1f77b4')
        axes[1, 0].axvline(treatment_effects.mean(), color='red', linestyle='--', linewidth=2,
                          label=f'Mean Effect: {treatment_effects.mean():.3f}')
        axes[1, 0].set_title('Distribution of Individual Treatment Effects', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Treatment Effect')
        axes[1, 0].legend()
        
        # 4. Treatment effect scatter
        df_plot = df.copy()
        df_plot['treatment_effect'] = treatment_effects
        
        # Find a numeric column for x-axis
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                       if c not in [treatment_col, outcome_col, 'treatment_effect']]
        if numeric_cols:
            x_col = numeric_cols[0]
            sns.scatterplot(data=df_plot, x=x_col, y='treatment_effect', 
                          hue='treatment_label', alpha=0.6, ax=axes[1, 1])
            axes[1, 1].set_title(f'Treatment Effect by {x_col}', fontsize=14, fontweight='bold')
        else:
            axes[1, 1].text(0.5, 0.5, 'No numeric features\navailable for plotting', 
                          ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved as '{save_path}'")
    
    def generate_report(self, df: pd.DataFrame, dowhy_results: dict, 
                       treatment_effects: np.ndarray, subgroup_effects: pd.DataFrame) -> str:
        """Generate a summary report of findings."""
        treatment_col = df.attrs.get('treatment_col', 'treatment')
        outcome_col = df.attrs.get('outcome_col', 'retained')
        
        report = []
        report.append("=" * 70)
        report.append("CAUSAL ANALYSIS REPORT: GAMING USER RETENTION")
        report.append("=" * 70)
        report.append("")
        
        # Sample stats
        report.append("SAMPLE STATISTICS")
        report.append(f"Total players analyzed: {len(df):,}")
        report.append(f"Players with early success: {df[treatment_col].sum():,} ({df[treatment_col].mean()*100:.1f}%)")
        report.append(f"Overall retention rate: {df[outcome_col].mean()*100:.1f}%")
        report.append("")
        
        # Naive comparison
        treated_retention = df[df[treatment_col] == 1][outcome_col].mean()
        control_retention = df[df[treatment_col] == 0][outcome_col].mean()
        naive_effect = treated_retention - control_retention
        
        report.append("NAIVE COMPARISON (Correlation - May be biased)")
        report.append(f"Retention with early success: {treated_retention*100:.1f}%")
        report.append(f"Retention without early success: {control_retention*100:.1f}%")
        report.append(f"Naive difference: {naive_effect*100:.1f} percentage points")
        report.append("")
        
        # Causal estimate
        report.append("CAUSAL ESTIMATE (After controlling for confounders)")
        report.append(f"Average Treatment Effect (ATE): {dowhy_results['estimate']:.3f}")
        report.append("")
        
        # Heterogeneous effects
        report.append("HETEROGENEOUS TREATMENT EFFECTS")
        report.append(f"Mean effect: {treatment_effects.mean():.3f}")
        report.append(f"Std dev: {treatment_effects.std():.3f}")
        report.append(f"Range: [{treatment_effects.min():.3f}, {treatment_effects.max():.3f}]")
        report.append("")
        
        # Subgroup analysis
        if not subgroup_effects.empty:
            report.append("SUBGROUP ANALYSIS")
            report.append(subgroup_effects.to_string())
            report.append("")
        
        # Conclusion
        ate = dowhy_results['estimate']
        if abs(ate) > 0.05:
            direction = "positive" if ate > 0 else "negative"
            report.append("CONCLUSION")
            report.append(f"Early success has a {direction} causal effect on player retention.")
            report.append(f"Effect size: {abs(ate)*100:.1f} percentage points")
            if ate > 0:
                report.append("Recommendation: Implement features to help new players achieve early wins.")
            else:
                report.append("Recommendation: Early success alone may not retain players. Focus on other factors.")
        else:
            report.append("CONCLUSION")
            report.append("Early success shows minimal causal effect on retention.")
            report.append("Recommendation: Focus on other engagement strategies beyond early wins.")
        
        report.append("=" * 70)
        
        return "\n".join(report)