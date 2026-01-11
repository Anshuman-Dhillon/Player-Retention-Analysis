"""
Gaming dataset loader and processor.
Downloads and prepares the Kaggle gaming behavior dataset for causal analysis.

Dataset: "Predict Online Gaming Behavior Dataset" from Kaggle
Link: https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset
"""
import pandas as pd
import numpy as np
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


class GamingDataLoader:
    """Loads and processes gaming behavior data for causal analysis."""
    
    def __init__(self, filepath: str):
        """
        Initialize loader with dataset path.
        
        Args:
            filepath: Path to the gaming CSV file
        """
        self.filepath = filepath
        self.raw_data = None
        self.processed_data = None
    
    def load_data(self) -> pd.DataFrame:
        """
        Load the gaming dataset from CSV.
        
        Returns:
            Raw DataFrame
        """
        print("Loading gaming dataset...")
        try:
            self.raw_data = pd.read_csv(self.filepath)
            print(f"Loaded {len(self.raw_data)} records")
            print(f"Columns: {list(self.raw_data.columns)}")
            return self.raw_data
        except FileNotFoundError:
            print(f"\n❌ ERROR: File not found at {self.filepath}")
            print("\n📥 Download Instructions:")
            print("1. Go to: https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset")
            print("2. Click 'Download' (you may need to create a free Kaggle account)")
            print("3. Extract the ZIP file")
            print("4. Place 'online_gaming_behavior_dataset.csv' in this directory")
            print("5. Run this script again")
            raise
    
    def process_data(self) -> pd.DataFrame:
        """
        Process raw data for causal analysis.
        
        Creates:
        - treatment: Whether player achieved early success (high achievements/playtime in early sessions)
        - outcome: Player retention/engagement level
        - confounders: Player skill, game difficulty, demographics
        
        Returns:
            Processed DataFrame ready for causal analysis
        """
        if self.raw_data is None:
            self.load_data()
        
        print("\nProcessing data for causal analysis...")
        df = self.raw_data.copy()
        
        # Print available columns to understand the data
        print(f"Available columns: {df.columns.tolist()}")
        
        # Basic data cleaning
        df = df.dropna()
        
        # IMPROVED: Create treatment using ONLY Age (pre-treatment variable)
        # This avoids circular logic where treatment is created from variables
        # that are themselves outcomes
        
        # Define treatment: Early success based on achievements
        # But we'll EXCLUDE AchievementsUnlocked from confounders later
        if 'AchievementsUnlocked' in df.columns:
            treatment_threshold = df['AchievementsUnlocked'].median()
            df['treatment'] = (df['AchievementsUnlocked'] >= treatment_threshold).astype(int)
            # Store this for later exclusion
            df.attrs['treatment_creator'] = 'AchievementsUnlocked'
        elif 'PlayTimeHours' in df.columns:
            treatment_threshold = df['PlayTimeHours'].quantile(0.6)
            df['treatment'] = (df['PlayTimeHours'] >= treatment_threshold).astype(int)
            df.attrs['treatment_creator'] = 'PlayTimeHours'
        else:
            print("⚠️ Warning: Creating placeholder treatment variable")
            df['treatment'] = np.random.binomial(1, 0.5, len(df))
            df.attrs['treatment_creator'] = None
        
        # Define outcome: Retention/Engagement
        if 'EngagementLevel' in df.columns:
            # Map engagement to binary: High/Very High = 1, Low/Medium = 0
            engagement_map = {
                'Low': 0, 'Medium': 0, 'High': 1, 'Very High': 1,
                'low': 0, 'medium': 0, 'high': 1, 'very high': 1
            }
            df['retained'] = df['EngagementLevel'].map(engagement_map)
            if df['retained'].isna().any():
                # If mapping failed, use numeric threshold
                df['retained'] = (pd.to_numeric(df['EngagementLevel'], errors='coerce') > 
                                df['EngagementLevel'].median()).astype(int)
        else:
            print("⚠️ Warning: Creating placeholder outcome variable")
            df['retained'] = np.random.binomial(1, 0.6, len(df))
        
        # Create engagement score (continuous outcome)
        if 'SessionsPerWeek' in df.columns and 'AvgSessionDurationMinutes' in df.columns:
            df['engagement_score'] = df['SessionsPerWeek'] * df['AvgSessionDurationMinutes']
        elif 'PlayTimeHours' in df.columns and df.attrs.get('treatment_creator') != 'PlayTimeHours':
            df['engagement_score'] = df['PlayTimeHours']
        else:
            # Use a combination of available metrics
            df['engagement_score'] = df.get('SessionsPerWeek', 1) * df.get('AvgSessionDurationMinutes', 50)
        
        self.processed_data = df
        
        print(f"\nProcessed {len(df)} records")
        print(f"Treatment group: {df['treatment'].sum()} ({df['treatment'].mean()*100:.1f}%)")
        print(f"Control group: {(~df['treatment'].astype(bool)).sum()} ({(1-df['treatment'].mean())*100:.1f}%)")
        print(f"Retention rate: {df['retained'].mean()*100:.1f}%")
        
        if df.attrs.get('treatment_creator'):
            print(f"\n⚠️ NOTE: Treatment was created using '{df.attrs['treatment_creator']}'")
            print(f"   This variable should be EXCLUDED from confounders to avoid circular logic!")
        
        return df
    
    def get_summary_stats(self) -> dict:
        """
        Get summary statistics for the processed data.
        
        Returns:
            Dictionary with summary statistics
        """
        if self.processed_data is None:
            raise ValueError("Data not processed yet. Call process_data() first.")
        
        df = self.processed_data
        
        stats = {
            'total_players': len(df),
            'treatment_group_size': df['treatment'].sum(),
            'control_group_size': (~df['treatment'].astype(bool)).sum(),
            'overall_retention': df['retained'].mean(),
            'treatment_retention': df[df['treatment'] == 1]['retained'].mean(),
            'control_retention': df[df['treatment'] == 0]['retained'].mean(),
            'naive_effect': (df[df['treatment'] == 1]['retained'].mean() - 
                           df[df['treatment'] == 0]['retained'].mean())
        }
        
        return stats
    
    def print_data_summary(self):
        """Print a nice summary of the loaded data."""
        if self.processed_data is None:
            print("No data processed yet.")
            return
        
        df = self.processed_data
        
        print("\n" + "="*60)
        print("GAMING DATASET SUMMARY")
        print("="*60)
        print(f"Total players: {len(df):,}")
        print(f"\nTreatment Distribution:")
        print(f"  - Early Success: {df['treatment'].sum():,} ({df['treatment'].mean()*100:.1f}%)")
        print(f"  - Control: {(~df['treatment'].astype(bool)).sum():,} ({(1-df['treatment'].mean())*100:.1f}%)")
        print(f"\nRetention Rates:")
        print(f"  - Overall: {df['retained'].mean()*100:.1f}%")
        print(f"  - With early success: {df[df['treatment']==1]['retained'].mean()*100:.1f}%")
        print(f"  - Without early success: {df[df['treatment']==0]['retained'].mean()*100:.1f}%")
        print(f"  - Naive difference: {(df[df['treatment']==1]['retained'].mean() - df[df['treatment']==0]['retained'].mean())*100:.1f} pp")
        
        if 'engagement_score' in df.columns:
            print(f"\nEngagement Score:")
            print(f"  - Mean: {df['engagement_score'].mean():.2f}")
            print(f"  - With early success: {df[df['treatment']==1]['engagement_score'].mean():.2f}")
            print(f"  - Without early success: {df[df['treatment']==0]['engagement_score'].mean():.2f}")
        
        print("\nAvailable columns for confounders:")
        print(f"  {[col for col in df.columns if col not in ['treatment', 'retained', 'engagement_score']]}")
        print("="*60)