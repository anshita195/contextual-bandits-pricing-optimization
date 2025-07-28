"""
UCI Online Retail Contextual Bandit Pipeline
============================================

This script implements a contextual bandit approach for markdown pricing optimization
using the UCI Online Retail dataset. The pipeline includes:

1. Data preprocessing and feature engineering
2. LinUCB bandit algorithm implementation
3. Weekly markdown optimization simulation
4. Performance evaluation and visualization

Based on the COMP (Contextual Bandits for Online Markdown Pricing) framework.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class LinUCB:
    """
    Linear Upper Confidence Bound (LinUCB) algorithm for contextual bandits.
    
    This implementation follows the LinUCB algorithm described in:
    "A Contextual-Bandit Approach to Personalized News Article Recommendation"
    by Lihong Li et al.
    """
    
    def __init__(self, n_arms, context_dim, alpha=1.0):
        """
        Initialize LinUCB.
        
        Args:
            n_arms (int): Number of arms (price points)
            context_dim (int): Dimension of context vector
            alpha (float): Exploration parameter
        """
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.alpha = alpha
        
        # Initialize A matrices (inverse covariance) for each arm
        self.A = np.array([np.eye(context_dim) for _ in range(n_arms)])
        
        # Initialize b vectors (reward observations) for each arm
        self.b = np.array([np.zeros(context_dim) for _ in range(n_arms)])
        
        # Initialize theta (coefficients) for each arm
        self.theta = np.array([np.zeros(context_dim) for _ in range(n_arms)])
        
    def select_arm(self, context):
        """
        Select arm based on LinUCB algorithm.
        
        Args:
            context (np.array): Context vector
            
        Returns:
            int: Selected arm index
        """
        # Calculate UCB values for each arm
        ucb_values = np.zeros(self.n_arms)
        
        for arm in range(self.n_arms):
            # Calculate theta for this arm
            A_inv = np.linalg.inv(self.A[arm])
            self.theta[arm] = A_inv @ self.b[arm]
            
            # Calculate UCB value
            ucb_values[arm] = (self.theta[arm] @ context + 
                              self.alpha * np.sqrt(context @ A_inv @ context))
        
        return np.argmax(ucb_values)
    
    def update(self, arm, context, reward):
        """
        Update the model with observed reward.
        
        Args:
            arm (int): Selected arm
            context (np.array): Context vector
            reward (float): Observed reward
        """
        # Update A matrix
        self.A[arm] += np.outer(context, context)
        
        # Update b vector
        self.b[arm] += reward * context

class UCIRetailPipeline:
    """
    Main pipeline for UCI Online Retail contextual bandit implementation.
    """
    
    def __init__(self, data_path, price_grid_size=15, train_ratio=0.7, alpha=1.0):
        """
        Initialize the pipeline.
        
        Args:
            data_path (str): Path to UCI Online Retail dataset
            price_grid_size (int): Number of price points for discretization
            train_ratio (float): Ratio of data to use for training
            alpha (float): LinUCB exploration parameter
        """
        self.data_path = data_path
        self.price_grid_size = price_grid_size
        self.train_ratio = train_ratio
        self.alpha = alpha
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.price_grid = None
        self.label_encoders = {}
        
        # Bandit components
        self.bandit = None
        self.context_dim = None
        
        # Results storage
        self.results = {
            'train_profits': [],
            'test_profits': [],
            'cumulative_train_profit': [],
            'cumulative_test_profit': [],
            'selected_prices': [],
            'actual_prices': []
        }
    
    def load_and_preprocess_data(self):
        """Load and preprocess the UCI Online Retail dataset."""
        print("Loading UCI Online Retail dataset...")
        
        # Load data
        self.raw_data = pd.read_excel(self.data_path)
        
        # Basic cleaning
        self.raw_data = self.raw_data.dropna(subset=['StockCode', 'UnitPrice', 'Quantity'])
        self.raw_data = self.raw_data[self.raw_data['Quantity'] > 0]
        self.raw_data = self.raw_data[self.raw_data['UnitPrice'] > 0]
        
        # Convert InvoiceDate to datetime
        self.raw_data['InvoiceDate'] = pd.to_datetime(self.raw_data['InvoiceDate'])
        
        # Extract week number
        self.raw_data['week'] = self.raw_data['InvoiceDate'].dt.isocalendar().week
        self.raw_data['year'] = self.raw_data['InvoiceDate'].dt.year
        
        # Create week identifier (year-week)
        self.raw_data['year_week'] = self.raw_data['year'].astype(str) + '_' + self.raw_data['week'].astype(str)
        
        print(f"Loaded {len(self.raw_data)} transactions")
        print(f"Date range: {self.raw_data['InvoiceDate'].min()} to {self.raw_data['InvoiceDate'].max()}")
        print(f"Week range: {self.raw_data['year_week'].min()} to {self.raw_data['year_week'].max()}")
    
    def aggregate_to_product_week(self):
        """Aggregate transactions to product-week level."""
        print("Aggregating to product-week level...")
        
        # Group by StockCode and year_week
        agg_data = self.raw_data.groupby(['StockCode', 'year_week']).agg({
            'Quantity': 'sum',
            'UnitPrice': 'last',  # Use last price of the week
            'Country': 'first'  # Use first country (most frequent would be better but simpler)
        }).reset_index()
        
        # Calculate profit
        agg_data['profit'] = agg_data['Quantity'] * agg_data['UnitPrice']
        
        # Sort by StockCode and year_week
        agg_data = agg_data.sort_values(['StockCode', 'year_week'])
        
        # Create week numbers (sequential)
        unique_weeks = sorted(agg_data['year_week'].unique())
        week_mapping = {week: i for i, week in enumerate(unique_weeks)}
        agg_data['week_num'] = agg_data['year_week'].map(week_mapping)
        
        self.processed_data = agg_data
        
        print(f"Aggregated to {len(self.processed_data)} product-week records")
        print(f"Unique products: {self.processed_data['StockCode'].nunique()}")
        print(f"Unique weeks: {self.processed_data['week_num'].nunique()}")
    
    def create_price_grid(self):
        """Create discretized price grid for bandit arms."""
        print("Creating price grid...")
        
        # Get price range (5th to 95th percentile)
        price_5th = self.processed_data['UnitPrice'].quantile(0.05)
        price_95th = self.processed_data['UnitPrice'].quantile(0.95)
        
        # Create evenly spaced price grid
        self.price_grid = np.linspace(price_5th, price_95th, self.price_grid_size)
        
        print(f"Price grid: {self.price_grid_size} points from {price_5th:.2f} to {price_95th:.2f}")
        print(f"Price points: {self.price_grid}")
    
    def engineer_features(self):
        """Engineer features for the bandit context."""
        print("Engineering features...")
        
        # Convert StockCode to string to handle mixed types
        self.processed_data['StockCode'] = self.processed_data['StockCode'].astype(str)
        
        # Encode StockCode
        stock_encoder = LabelEncoder()
        self.processed_data['stock_code_encoded'] = stock_encoder.fit_transform(self.processed_data['StockCode'])
        self.label_encoders['stock_code'] = stock_encoder
        
        # Create lag features (previous week's sales)
        self.processed_data['prev_week_sales'] = self.processed_data.groupby('StockCode')['Quantity'].shift(1).fillna(0)
        
        # Create week features
        self.processed_data['week_normalized'] = self.processed_data['week_num'] / self.processed_data['week_num'].max()
        
        # Create price features (normalized)
        price_mean = self.processed_data['UnitPrice'].mean()
        price_std = self.processed_data['UnitPrice'].std()
        self.processed_data['price_normalized'] = (self.processed_data['UnitPrice'] - price_mean) / price_std
        
        print("Features created:")
        print(f"- StockCode (encoded): {len(stock_encoder.classes_)} unique products")
        print(f"- Previous week sales")
        print(f"- Week (normalized)")
        print(f"- Price (normalized)")
    
    def split_train_test(self):
        """Split data into train and test periods."""
        print("Splitting into train/test periods...")
        
        # Get unique weeks
        unique_weeks = sorted(self.processed_data['week_num'].unique())
        split_week_idx = int(len(unique_weeks) * self.train_ratio)
        split_week = unique_weeks[split_week_idx]
        
        # Split data
        self.train_data = self.processed_data[self.processed_data['week_num'] < split_week].copy()
        self.test_data = self.processed_data[self.processed_data['week_num'] >= split_week].copy()
        
        print(f"Train period: weeks 0 to {split_week-1} ({len(self.train_data)} records)")
        print(f"Test period: weeks {split_week} onwards ({len(self.test_data)} records)")
    
    def create_context_vector(self, row):
        """Create context vector for a product-week."""
        # Context features: [stock_code_one_hot, week_normalized, price_normalized, prev_week_sales_normalized]
        
        # StockCode one-hot encoding (simplified - use encoded value directly)
        stock_code_feature = row['stock_code_encoded'] / len(self.label_encoders['stock_code'].classes_)
        
        # Normalize previous week sales
        sales_mean = self.processed_data['Quantity'].mean()
        sales_std = self.processed_data['Quantity'].std()
        prev_sales_normalized = (row['prev_week_sales'] - sales_mean) / sales_std if sales_std > 0 else 0
        
        # Create context vector
        context = np.array([
            stock_code_feature,
            row['week_normalized'],
            row['price_normalized'],
            prev_sales_normalized
        ])
        
        return context
    
    def initialize_bandit(self):
        """Initialize the LinUCB bandit."""
        print("Initializing LinUCB bandit...")
        
        # Context dimension
        self.context_dim = 4  # stock_code, week, price, prev_sales
        
        # Initialize bandit
        self.bandit = LinUCB(
            n_arms=self.price_grid_size,
            context_dim=self.context_dim,
            alpha=self.alpha
        )
        
        print(f"LinUCB initialized with {self.price_grid_size} arms and {self.context_dim} context dimensions")
    
    def simulate_weekly_optimization(self, data, is_training=True):
        """Simulate weekly markdown optimization."""
        print(f"Simulating weekly optimization ({'training' if is_training else 'testing'})...")
        
        # Group by week
        weekly_data = data.groupby('week_num')
        
        cumulative_profit = 0
        weekly_profits = []
        selected_prices = []
        actual_prices = []
        
        for week_num, week_data in weekly_data:
            week_profit = 0
            
            # Process each product in this week
            for _, row in week_data.iterrows():
                # Create context
                context = self.create_context_vector(row)
                
                # Select price using bandit
                selected_arm = self.bandit.select_arm(context)
                selected_price = self.price_grid[selected_arm]
                
                # Simulate sales at selected price
                # For simplicity, assume sales are proportional to price change
                price_ratio = selected_price / row['UnitPrice']
                simulated_quantity = row['Quantity'] * (1 + 0.5 * (1 - price_ratio))  # Simple demand model
                simulated_quantity = max(0, simulated_quantity)  # No negative sales
                
                # Calculate profit
                profit = selected_price * simulated_quantity
                week_profit += profit
                
                # Store results
                selected_prices.append(selected_price)
                actual_prices.append(row['UnitPrice'])
                
                # Update bandit (only during training)
                if is_training:
                    self.bandit.update(selected_arm, context, profit)
            
            cumulative_profit += week_profit
            weekly_profits.append(week_profit)
            
            if is_training:
                self.results['train_profits'].append(week_profit)
                self.results['cumulative_train_profit'].append(cumulative_profit)
            else:
                self.results['test_profits'].append(week_profit)
                self.results['cumulative_test_profit'].append(cumulative_profit)
        
        print(f"{'Training' if is_training else 'Testing'} completed:")
        print(f"- Total profit: {cumulative_profit:.2f}")
        print(f"- Average weekly profit: {np.mean(weekly_profits):.2f}")
        
        return cumulative_profit, weekly_profits
    
    def run_pipeline(self):
        """Run the complete pipeline."""
        print("=" * 50)
        print("UCI Online Retail Contextual Bandit Pipeline")
        print("=" * 50)
        
        # Step 1: Load and preprocess data
        self.load_and_preprocess_data()
        
        # Step 2: Aggregate to product-week level
        self.aggregate_to_product_week()
        
        # Step 3: Create price grid
        self.create_price_grid()
        
        # Step 4: Engineer features
        self.engineer_features()
        
        # Step 5: Split train/test
        self.split_train_test()
        
        # Step 6: Initialize bandit
        self.initialize_bandit()
        
        # Step 7: Train bandit
        print("\n" + "=" * 30)
        print("TRAINING PHASE")
        print("=" * 30)
        train_profit, _ = self.simulate_weekly_optimization(self.train_data, is_training=True)
        
        # Step 8: Test bandit
        print("\n" + "=" * 30)
        print("TESTING PHASE")
        print("=" * 30)
        test_profit, _ = self.simulate_weekly_optimization(self.test_data, is_training=False)
        
        # Step 9: Print summary
        print("\n" + "=" * 30)
        print("SUMMARY")
        print("=" * 30)
        print(f"Training profit: {train_profit:.2f}")
        print(f"Testing profit: {test_profit:.2f}")
        print(f"Total profit: {train_profit + test_profit:.2f}")
        
        return self.results
    
    def plot_results(self):
        """Plot the results."""
        print("Creating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Cumulative profit over time
        axes[0, 0].plot(self.results['cumulative_train_profit'], label='Training', color='blue')
        axes[0, 0].plot(range(len(self.results['cumulative_train_profit']), 
                          len(self.results['cumulative_train_profit']) + len(self.results['cumulative_test_profit'])), 
                       self.results['cumulative_test_profit'], label='Testing', color='red')
        axes[0, 0].set_title('Cumulative Profit Over Time')
        axes[0, 0].set_xlabel('Week')
        axes[0, 0].set_ylabel('Cumulative Profit')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: Weekly profit
        axes[0, 1].plot(self.results['train_profits'], label='Training', color='blue', alpha=0.7)
        axes[0, 1].plot(range(len(self.results['train_profits']), 
                             len(self.results['train_profits']) + len(self.results['test_profits'])), 
                       self.results['test_profits'], label='Testing', color='red', alpha=0.7)
        axes[0, 1].set_title('Weekly Profit')
        axes[0, 1].set_xlabel('Week')
        axes[0, 1].set_ylabel('Weekly Profit')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot 3: Price comparison (sample)
        sample_size = min(100, len(self.results['selected_prices']))
        sample_indices = np.random.choice(len(self.results['selected_prices']), sample_size, replace=False)
        
        axes[1, 0].scatter([self.results['actual_prices'][i] for i in sample_indices], 
                          [self.results['selected_prices'][i] for i in sample_indices], 
                          alpha=0.6)
        axes[1, 0].plot([0, max(self.results['actual_prices'])], [0, max(self.results['actual_prices'])], 
                       'r--', label='Perfect Match')
        axes[1, 0].set_title('Selected vs Actual Prices (Sample)')
        axes[1, 0].set_xlabel('Actual Price')
        axes[1, 0].set_ylabel('Selected Price')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot 4: Price distribution
        axes[1, 1].hist(self.results['selected_prices'], bins=20, alpha=0.7, label='Selected Prices')
        axes[1, 1].hist(self.results['actual_prices'], bins=20, alpha=0.7, label='Actual Prices')
        axes[1, 1].set_title('Price Distribution')
        axes[1, 1].set_xlabel('Price')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('uci_retail_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Results saved to 'uci_retail_results.png'")

def main():
    """Main function to run the pipeline."""
    # Initialize pipeline
    pipeline = UCIRetailPipeline(
        data_path='UCI_dataset/Online Retail.xlsx',
        price_grid_size=15,
        train_ratio=0.7,
        alpha=1.0
    )
    
    # Run pipeline
    results = pipeline.run_pipeline()
    
    # Plot results
    pipeline.plot_results()
    
    print("\nPipeline completed successfully!")
    print("Check 'uci_retail_results.png' for visualizations.")

if __name__ == "__main__":
    main() 