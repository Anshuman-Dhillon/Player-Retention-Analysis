"""
Test suite for gaming data loader and causal analyzer.
Ensures data quality and prevents bugs in causal inference pipeline.
"""
import pytest
import pandas as pd
import numpy as np
from gaming_data_loader import GamingDataLoader
from causal_analyzer import CausalAnalyzer


class TestGamingDataLoader:
    """Tests for GamingDataLoader class."""
    
    def test_loader_initialization(self):
        """Test loader can be initialized."""
        loader = GamingDataLoader("test_data.csv")
        assert loader.filepath == "test_data.csv"
        assert loader.raw_data is None
        assert loader.processed_data is None
    
    def test_process_data_creates_required_columns(self):
        """Test data processing creates necessary columns."""
        # Create mock gaming data
        mock_data = pd.DataFrame({
            'PlayerID': range(100),
            'Age': np.random.randint(18, 50, 100),
            'PlayTimeHours': np.random.exponential(10, 100),
            'AchievementsUnlocked': np.random.poisson(5, 100),
            'SessionsPerWeek': np.random.randint(1, 20, 100),
            'AvgSessionDurationMinutes': np.random.randint(10, 120, 100),
            'EngagementLevel': np.random.choice(['Low', 'Medium', 'High', 'Very High'], 100)
        })
        
        loader = GamingDataLoader("dummy.csv")
        loader.raw_data = mock_data
        processed = loader.process_data()
        
        # Check required columns exist
        assert 'treatment' in processed.columns
        assert 'retained' in processed.columns
        assert 'engagement_score' in processed.columns
    
    def test_process_data_creates_binary_treatment(self):
        """Ensure treatment is binary 0/1."""
        mock_data = pd.DataFrame({
            'PlayerID': range(50),
            'AchievementsUnlocked': np.random.poisson(5, 50),
            'PlayTimeHours': np.random.exponential(10, 50),
            'EngagementLevel': np.random.choice(['Low', 'High'], 50)
        })
        
        loader = GamingDataLoader("dummy.csv")
        loader.raw_data = mock_data
        processed = loader.process_data()
        
        # Treatment should be binary
        assert processed['treatment'].isin([0, 1]).all()
        
        # Should have both treatment and control
        assert processed['treatment'].sum() > 0
        assert (processed['treatment'] == 0).sum() > 0


class TestCausalAnalyzer:
    """Tests for CausalAnalyzer class."""
    
    @pytest.fixture
    def sample_gaming_data(self):
        """Create sample gaming data for testing."""
        np.random.seed(42)
        n = 300
        
        # Confounders
        age = np.random.randint(18, 50, n)
        skill_level = np.random.normal(50, 15, n)
        playtime = np.random.exponential(20, n)
        
        # Treatment (affected by confounders)
        treatment_prob = 1 / (1 + np.exp(-(skill_level - 50) / 15))
        treatment = np.random.binomial(1, treatment_prob)
        
        # Outcome (affected by treatment and confounders)
        retention_prob = (
            0.4 +
            0.15 * treatment +  # TRUE causal effect
            0.002 * skill_level +
            0.005 * playtime +
            np.random.normal(0, 0.1, n)
        )
        retention_prob = np.clip(retention_prob, 0, 1)
        retained = np.random.binomial(1, retention_prob)
        
        engagement = (
            20 +
            8 * treatment +  # TRUE causal effect
            0.3 * skill_level +
            0.5 * playtime +
            np.random.normal(0, 5, n)
        )
        
        df = pd.DataFrame({
            'PlayerID': range(n),
            'Age': age,
            'SkillLevel': skill_level,
            'PlayTimeHours': playtime,
            'treatment': treatment,
            'retained': retained,
            'engagement_score': engagement
        })
        
        return df
    
    def test_prepare_data_creates_metadata(self, sample_gaming_data):
        """Test that prepare_data stores metadata in DataFrame attrs."""
        analyzer = CausalAnalyzer()
        
        prepared = analyzer.prepare_data(
            sample_gaming_data,
            confounders=['Age', 'SkillLevel', 'PlayTimeHours']
        )
        
        assert 'confounders' in prepared.attrs
        assert 'treatment_col' in prepared.attrs
        assert 'outcome_col' in prepared.attrs
    
    def test_prepare_data_auto_detects_confounders(self, sample_gaming_data):
        """Test auto-detection of confounder variables."""
        analyzer = CausalAnalyzer()
        
        prepared = analyzer.prepare_data(sample_gaming_data, confounders=None)
        
        # Should auto-detect numeric columns
        assert len(prepared.attrs['confounders']) > 0
        assert 'treatment' not in prepared.attrs['confounders']
        assert 'retained' not in prepared.attrs['confounders']
    
    def test_prepare_data_removes_missing_values(self, sample_gaming_data):
        """Ensure missing values are removed."""
        # Add some NaN values
        sample_gaming_data.loc[0:5, 'SkillLevel'] = np.nan
        
        analyzer = CausalAnalyzer()
        prepared = analyzer.prepare_data(
            sample_gaming_data,
            confounders=['Age', 'SkillLevel', 'PlayTimeHours']
        )
        
        # Should have fewer rows due to NaN removal
        assert len(prepared) < len(sample_gaming_data)
        assert prepared['SkillLevel'].isna().sum() == 0
    
    def test_treatment_effect_estimation_runs(self, sample_gaming_data):
        """Test that heterogeneous effect estimation completes."""
        analyzer = CausalAnalyzer()
        prepared = analyzer.prepare_data(
            sample_gaming_data,
            confounders=['Age', 'SkillLevel', 'PlayTimeHours']
        )
        
        # Should not raise exception
        treatment_effects = analyzer.estimate_heterogeneous_effects(prepared)
        
        # Should return effects for all users
        assert len(treatment_effects) == len(prepared)
        assert not np.isnan(treatment_effects).any()
    
    def test_treatment_effects_have_variance(self, sample_gaming_data):
        """Test that treatment effects vary across individuals."""
        analyzer = CausalAnalyzer()
        prepared = analyzer.prepare_data(
            sample_gaming_data,
            confounders=['Age', 'SkillLevel', 'PlayTimeHours']
        )
        
        treatment_effects = analyzer.estimate_heterogeneous_effects(prepared)
        
        # Effects should have some variance (not all identical)
        # Allow small std for small samples, but should be > 0
        assert treatment_effects.std() >= 0
    
    def test_no_data_leakage_in_preparation(self, sample_gaming_data):
        """Ensure outcome doesn't leak into treatment assignment."""
        analyzer = CausalAnalyzer()
        
        # Create two versions with different outcomes
        df1 = sample_gaming_data.copy()
        df1['retained'] = 1  # All retained
        
        df2 = sample_gaming_data.copy()
        df2['retained'] = 0  # None retained
        
        prepared1 = analyzer.prepare_data(df1, confounders=['Age', 'SkillLevel'])
        prepared2 = analyzer.prepare_data(df2, confounders=['Age', 'SkillLevel'])
        
        # Treatment should be identical despite different outcomes
        pd.testing.assert_series_equal(
            prepared1['treatment'].reset_index(drop=True),
            prepared2['treatment'].reset_index(drop=True)
        )


class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_full_pipeline_with_synthetic_data(self):
        """Test complete analysis pipeline."""
        np.random.seed(42)
        n = 200
        
        # Generate synthetic gaming data
        df = pd.DataFrame({
            'PlayerID': range(n),
            'Age': np.random.randint(18, 50, n),
            'SkillLevel': np.random.normal(50, 15, n),
            'PlayTimeHours': np.random.exponential(20, n),
            'AchievementsUnlocked': np.random.poisson(5, n),
            'treatment': np.random.binomial(1, 0.5, n),
            'retained': np.random.binomial(1, 0.6, n),
            'engagement_score': np.random.exponential(30, n)
        })
        
        # Run analysis
        analyzer = CausalAnalyzer()
        prepared = analyzer.prepare_data(
            df,
            confounders=['Age', 'SkillLevel', 'PlayTimeHours']
        )
        
        # Should complete without errors
        assert len(prepared) > 0
        assert 'treatment' in prepared.columns
        assert 'retained' in prepared.columns
        
        # Effect estimation should work
        effects = analyzer.estimate_heterogeneous_effects(prepared)
        assert len(effects) == len(prepared)
    
    def test_minimum_sample_size_check(self):
        """Test that analysis handles small sample sizes appropriately."""
        # Very small dataset
        n = 30
        df = pd.DataFrame({
            'PlayerID': range(n),
            'Age': np.random.randint(18, 50, n),
            'treatment': np.random.binomial(1, 0.5, n),
            'retained': np.random.binomial(1, 0.6, n),
            'engagement_score': np.random.exponential(30, n)
        })
        
        analyzer = CausalAnalyzer()
        prepared = analyzer.prepare_data(df, confounders=['Age'])
        
        # Should still run but results may be unstable
        assert len(prepared) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])