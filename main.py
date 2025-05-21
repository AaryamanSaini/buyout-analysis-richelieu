import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monte_carlo_simulation import MonteCarloSimulation
from financial_modeling import BuyoutModel

def main():
    """
    Run the complete private equity buyout analysis for Richelieu Hardware (TSX: RCH).
    """
    print("Starting PE Buyout Analysis for Richelieu Hardware (TSX: RCH)")
    print("-" * 70)
    
    # Create output directory if it doesn't exist
    output_path = 'output'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Step 1: Run Monte Carlo simulation
    print("\nStep 1: Running Monte Carlo simulation for revenue forecasting...")
    mc_sim = MonteCarloSimulation(
        base_revenue=1200,              # $1.2B CAD in revenue
        projection_years=5,             # 5-year projection
        num_simulations=1000,           # 1,000 possible futures
        base_case_growth_mean=0.08,     # 8% mean growth in base case
        base_case_growth_std=0.03,      # 3% standard deviation
        bull_case_prob=0.30,            # 30% probability of bull case
        bull_case_growth_boost=0.04,    # +4% growth in bull case
        bear_case_prob=0.20,            # 20% probability of bear case
        bear_case_growth_penalty=0.06   # -6% growth in bear case
    )
    
    # Run the simulation
    simulation_results = mc_sim.run_simulation()
    
    # Export simulation results to Excel
    sim_excel_path = mc_sim.export_to_excel(output_path)
    
    # Plot simulation results
    mc_sim.plot_simulation_results(output_path)
    
    print(f"Monte Carlo simulation completed with {len(simulation_results)} trajectories")
    
    # Basic statistics on the simulation results
    final_year = f"Year_{mc_sim.projection_years}"
    final_revenues = simulation_results[final_year]
    
    print(f"\nFinal Year Revenue Statistics (Year {mc_sim.projection_years}):")
    print(f"  Mean: ${final_revenues.mean():.2f}M CAD")
    print(f"  Median: ${final_revenues.median():.2f}M CAD")
    print(f"  Min: ${final_revenues.min():.2f}M CAD")
    print(f"  Max: ${final_revenues.max():.2f}M CAD")
    
    # Calculate scenario statistics
    for scenario in ['Base', 'Bull', 'Bear']:
        scenario_mask = simulation_results['Scenario'] == scenario
        scenario_count = scenario_mask.sum()
        scenario_pct = scenario_count / len(simulation_results) * 100
        scenario_final = simulation_results.loc[scenario_mask, final_year]
        
        print(f"\n{scenario} Case ({scenario_count} paths, {scenario_pct:.1f}%):")
        print(f"  Mean Final Revenue: ${scenario_final.mean():.2f}M CAD")
        print(f"  Mean CAGR: {(((scenario_final.mean() / mc_sim.base_revenue) ** (1/mc_sim.projection_years)) - 1) * 100:.2f}%")
    
    # Step 2: Run the buyout model
    print("\nStep 2: Building the PE buyout model...")
    
    buyout_model = BuyoutModel(
        simulation_results=simulation_results,
        enterprise_value=2100,           # EV of $2.1B CAD
        debt_percentage=0.6,             # 60% debt / 40% equity
        ebitda_margin_base=0.12,         # 12% initial EBITDA margin
        ebitda_margin_improvement=0.015, # +1.5% margin improvement per year
        exit_multiple_base=11.0,         # 11x EV/EBITDA at exit in base case
        exit_multiple_bull=13.0,         # 13x in bull case
        exit_multiple_bear=9.0,          # 9x in bear case
        tax_rate=0.27,                   # 27% tax rate
        capex_percent_revenue=0.03,      # CapEx at 3% of revenue
        nwc_percent_revenue=0.15,        # NWC at 15% of revenue
        nwc_change_percent_revenue_growth=0.6,  # NWC change at 60% of revenue growth
        debt_interest_rate=0.065,        # 6.5% interest rate on debt
        debt_repayment_percent_fcf=0.6,  # 60% of FCF goes to debt repayment
        minimum_cash=50                  # $50M minimum cash balance
    )
    
    # Build the model with 100 sampled paths
    model_results = buyout_model.build_model(sample_size=100)
    
    # Get summary statistics
    summary = buyout_model.summarize_results()
    
    print("\nBuyout Model IRR Summary:")
    print(summary)
    
    # Export model to Excel
    model_excel_path = buyout_model.export_to_excel(output_path)
    
    print("\nAnalysis completed successfully!")
    print("-" * 70)
    print("Check the 'output' directory for the following files:")
    print(f"  - {os.path.basename(sim_excel_path)}: Raw Monte Carlo simulation data")
    print(f"  - {os.path.basename(model_excel_path)}: Complete DCF and LBO model")
    print(f"  - revenue_projections.png: Visualization of revenue projections")

if __name__ == "__main__":
    main() 
