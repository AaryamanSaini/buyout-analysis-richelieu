import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import os

class MonteCarloSimulation:
    def __init__(self, 
                 base_revenue=1200,  # Base revenue in millions CAD
                 projection_years=5,
                 num_simulations=1000,
                 base_case_growth_mean=0.08,
                 base_case_growth_std=0.03,
                 bull_case_prob=0.30,
                 bull_case_growth_boost=0.04,
                 bear_case_prob=0.20,
                 bear_case_growth_penalty=0.06):
        """
        Initialize the Monte Carlo simulation for Richelieu Hardware revenue projections.
        
        Args:
            base_revenue: Starting revenue in CAD millions
            projection_years: Number of years to project
            num_simulations: Number of Monte Carlo simulations
            base_case_growth_mean: Mean growth rate in base case
            base_case_growth_std: Standard deviation of growth in base case
            bull_case_prob: Probability of bull case
            bull_case_growth_boost: Additional growth in bull case
            bear_case_prob: Probability of bear case
            bear_case_growth_penalty: Growth reduction in bear case
        """
        self.base_revenue = base_revenue
        self.projection_years = projection_years
        self.num_simulations = num_simulations
        self.base_case_growth_mean = base_case_growth_mean
        self.base_case_growth_std = base_case_growth_std
        self.bull_case_prob = bull_case_prob
        self.bull_case_growth_boost = bull_case_growth_boost
        self.bear_case_prob = bear_case_prob
        self.bear_case_growth_penalty = bear_case_growth_penalty
        
        # Initialize results
        self.results = None
        self.scenario_probabilities = {
            'Base': 1 - bull_case_prob - bear_case_prob,
            'Bull': bull_case_prob,
            'Bear': bear_case_prob
        }

    def run_simulation(self):
        """Run the Monte Carlo simulation for revenue projections."""
        # Initialize the results DataFrame with years as columns
        years = list(range(self.projection_years + 1))
        column_names = [f'Year_{year}' for year in years]
        scenario_column = 'Scenario'
        
        # Create empty DataFrame to store results
        self.results = pd.DataFrame(index=range(self.num_simulations), columns=[scenario_column] + column_names)
        
        # Set initial revenues for Year 0
        self.results['Year_0'] = self.base_revenue
        
        # Generate random scenarios based on probabilities
        scenarios = np.random.choice(
            ['Base', 'Bull', 'Bear'], 
            size=self.num_simulations, 
            p=[self.scenario_probabilities['Base'], 
               self.scenario_probabilities['Bull'], 
               self.scenario_probabilities['Bear']]
        )
        self.results['Scenario'] = scenarios
        
        # Run simulations for each trajectory
        for sim in range(self.num_simulations):
            scenario = scenarios[sim]
            
            # Adjust growth parameters based on scenario
            if scenario == 'Bull':
                growth_mean = self.base_case_growth_mean + self.bull_case_growth_boost
                growth_std = self.base_case_growth_std * 0.9  # Less volatility in bull case
            elif scenario == 'Bear':
                growth_mean = self.base_case_growth_mean - self.bear_case_growth_penalty
                growth_std = self.base_case_growth_std * 1.4  # More volatility in bear case
            else:  # Base case
                growth_mean = self.base_case_growth_mean
                growth_std = self.base_case_growth_std
            
            # Generate growth rates for projection years
            growth_rates = np.random.normal(growth_mean, growth_std, self.projection_years)
            
            # Calculate revenues
            current_revenue = self.base_revenue
            for year in range(1, self.projection_years + 1):
                # Ensure growth rate isn't catastrophically negative
                growth_rate = max(growth_rates[year-1], -0.20)  
                current_revenue = current_revenue * (1 + growth_rate)
                self.results.loc[sim, f'Year_{year}'] = current_revenue
        
        return self.results
    
    def export_to_excel(self, output_path='output'):
        """Export simulation results to Excel."""
        # Create output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        # File path for the output Excel file
        file_path = os.path.join(output_path, 'simulation_results.xlsx')
        
        # Export to Excel
        self.results.to_excel(file_path, index=False)
        
        print(f"Simulation results exported to {file_path}")
        return file_path
    
    def plot_simulation_results(self, output_path='output'):
        """Plot simulation results."""
        if self.results is None:
            raise ValueError("Run simulation first before plotting results")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        # Set the style for the plots
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Create a figure
        plt.figure(figsize=(14, 10))
        
        # Get years for x-axis
        years = range(self.projection_years + 1)
        
        # Plot each trajectory with transparency
        for i in range(min(300, self.num_simulations)):  # Limit to 300 trajectories for visual clarity
            scenario = self.results.loc[i, 'Scenario']
            revenue_data = self.results.iloc[i, 1:].values  # Skip the scenario column
            
            if scenario == 'Bull':
                color = 'green'
            elif scenario == 'Bear':
                color = 'red'
            else:  # Base
                color = 'blue'
                
            plt.plot(years, revenue_data, alpha=0.05, color=color)
        
        # Calculate and plot percentiles
        percentiles = [10, 25, 50, 75, 90]
        percentile_data = {}
        
        for year in years:
            year_data = self.results[f'Year_{year}']
            for p in percentiles:
                if p not in percentile_data:
                    percentile_data[p] = []
                percentile_data[p].append(np.percentile(year_data, p))
        
        # Plot percentile lines
        for p in percentiles:
            plt.plot(years, percentile_data[p], 
                     linewidth=2, 
                     color='black' if p == 50 else 'darkgray',
                     label=f"{p}th Percentile")
        
        # Plot mean for each scenario
        for scenario in ['Base', 'Bull', 'Bear']:
            scenario_mask = self.results['Scenario'] == scenario
            scenario_data = self.results[scenario_mask].iloc[:, 1:].mean()
            plt.plot(years, scenario_data, 
                     linewidth=3, 
                     label=f"{scenario} Case Mean",
                     color='darkblue' if scenario == 'Base' else ('darkgreen' if scenario == 'Bull' else 'darkred'))
        
        # Set the title and labels
        plt.title('Richelieu Hardware Revenue Projections (Monte Carlo Simulation)', fontsize=16)
        plt.xlabel('Projection Year', fontsize=14)
        plt.ylabel('Revenue (in CAD millions)', fontsize=14)
        plt.grid(True)
        plt.legend()
        
        # Save the figure
        plt.savefig(os.path.join(output_path, 'revenue_projections.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Simulation plot saved to {output_path}/revenue_projections.png")

if __name__ == "__main__":
    # Test the Monte Carlo simulation
    mc_sim = MonteCarloSimulation()
    results = mc_sim.run_simulation()
    mc_sim.export_to_excel()
    mc_sim.plot_simulation_results()
    
    print("Simulation completed successfully!") 
