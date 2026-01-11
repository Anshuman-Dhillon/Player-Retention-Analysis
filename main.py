"""
Main script to run causal analysis on gaming user retention data.

Usage:
    python main.py --data gaming_data.csv
    python main.py --data gaming_data.csv --confounders Age PlayTimeHours GameDifficulty
"""
import argparse
import pandas as pd
from gaming_data_loader import GamingDataLoader
from causal_analyzer import CausalAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description='Run causal analysis on gaming user retention',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --data online_gaming_behavior_dataset.csv
  python main.py --data gaming_data.csv --confounders Age PlayTimeHours

Download the dataset from:
  https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset
        """
    )
    
    parser.add_argument('--data', type=str, 
                       default='online_gaming_behavior_dataset.csv',
                       help='Path to gaming dataset CSV file')
    parser.add_argument('--confounders', type=str, nargs='*', default=None,
                       help='List of confounder column names (auto-detected if not specified)')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Use a random sample of N rows (for faster testing)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("GAMING USER RETENTION - CAUSAL ANALYSIS")
    print("=" * 70)
    print("\nResearch Question:")
    print("Does early success (achievements/wins) CAUSE players to stay engaged,")
    print("or do naturally skilled players just win more AND stay longer?")
    print("=" * 70)
    
    # Step 1: Load Data
    print("\n" + "=" * 70)
    print("STEP 1: DATA LOADING")
    print("=" * 70)
    
    loader = GamingDataLoader(args.data)
    
    try:
        raw_data = loader.load_data()
    except FileNotFoundError:
        return
    
    # Sample if requested
    if args.sample_size and args.sample_size < len(raw_data):
        print(f"\nUsing random sample of {args.sample_size} rows for faster analysis...")
        raw_data = raw_data.sample(n=args.sample_size, random_state=42)
    
    # Process data
    processed_data = loader.process_data()
    loader.print_data_summary()
    
    # Step 2: Prepare for Causal Analysis
    print("\n" + "=" * 70)
    print("STEP 2: CAUSAL ANALYSIS PREPARATION")
    print("=" * 70)
    
    analyzer = CausalAnalyzer()
    prepared_df = analyzer.prepare_data(
        processed_data, 
        confounders=args.confounders
    )
    
    # Check if we have enough data
    if len(prepared_df) < 100:
        print("\n⚠️ WARNING: Sample size is very small (< 100). Results may be unreliable.")
        print("Consider using a larger dataset or removing --sample-size filter.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Step 3: DoWhy Causal Estimation
    print("\n" + "=" * 70)
    print("STEP 3: CAUSAL EFFECT ESTIMATION (DoWhy)")
    print("=" * 70)
    
    dowhy_results = analyzer.estimate_treatment_effect_dowhy(prepared_df)
    
    # Step 4: Heterogeneous Treatment Effects
    print("\n" + "=" * 70)
    print("STEP 4: HETEROGENEOUS EFFECTS (EconML)")
    print("=" * 70)
    
    treatment_effects = analyzer.estimate_heterogeneous_effects(prepared_df)
    print(f"\nMean treatment effect: {treatment_effects.mean():.3f}")
    print(f"Std deviation: {treatment_effects.std():.3f}")
    print(f"Range: [{treatment_effects.min():.3f}, {treatment_effects.max():.3f}]")
    
    # Step 5: Subgroup Analysis
    print("\n" + "=" * 70)
    print("STEP 5: SUBGROUP ANALYSIS")
    print("=" * 70)
    
    subgroup_effects = analyzer.analyze_subgroups(prepared_df, treatment_effects)
    if not subgroup_effects.empty:
        print("\n" + subgroup_effects.to_string())
    
    # Step 6: Visualization & Reporting
    print("\n" + "=" * 70)
    print("STEP 6: RESULTS & REPORTING")
    print("=" * 70)
    
    analyzer.visualize_results(prepared_df, treatment_effects)
    
    report = analyzer.generate_report(
        prepared_df,
        dowhy_results,
        treatment_effects,
        subgroup_effects
    )
    
    print("\n" + report)
    
    # Save outputs
    with open('gaming_causal_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    processed_data.to_csv('gaming_processed_data.csv', index=False)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  ✓ gaming_causal_analysis_report.txt - Full report")
    print("  ✓ causal_analysis_results.png - Visualizations")
    print("  ✓ gaming_processed_data.csv - Processed data")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()