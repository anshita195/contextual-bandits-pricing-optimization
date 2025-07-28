"""
UCI Online Retail Contextual Bandit Demo
========================================

A simplified demo version of the contextual bandit pipeline for markdown pricing.
This version uses a smaller subset of data for faster execution and demonstration.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class LinUCB:
    """Simplified LinUCB implementation for demo."""
    
    def __init__(self, n_arms, context_dim, alpha=1.0):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.alpha = alpha
        
        # Initialize A matrices and b vectors for each arm
        self.A = np.array([np.eye(context_dim) for _ in range(n_arms)])
        self.b = np.array([np.zeros(context_dim) for _ in range(n_arms)])
        self.theta = np.array([np.zeros(context_dim) for _ in range(n_arms)])
        
    def select_arm(self, context):
        ucb_values = np.zeros(self.n_arms)
        
        for arm in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[arm])
            self.theta[arm] = A_inv @ self.b[arm]
            ucb_values[arm] = (self.theta[arm] @ context + 
                              self.alpha * np.sqrt(context @ A_inv @ context))
        
        return np.argmax(ucb_values)
    
    def update(self, arm, context, reward):
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context

def load_and_preprocess_data():
    """Load and preprocess UCI dataset with sampling for demo."""
    print("Loading UCI Online Retail dataset...")
    
    # Load data
    raw_data = pd.read_excel('UCI_dataset/Online Retail.xlsx')
    
    # Basic cleaning
    raw_data = raw_data.dropna(subset=['StockCode', 'UnitPrice', 'Quantity'])
    raw_data = raw_data[raw_data['Quantity'] > 0]
    raw_data = raw_data[raw_data['UnitPrice'] > 0]
    
    # Sample for demo (use top 100 products by sales volume)
    product_sales = raw_data.groupby('StockCode')['Quantity'].sum().sort_values(ascending=False)
    top_products = product_sales.head(100).index
    raw_data = raw_data[raw_data['StockCode'].isin(top_products)]
    
    # Convert InvoiceDate to datetime
    raw_data['InvoiceDate'] = pd.to_datetime(raw_data['InvoiceDate'])
    
    # Extract week number
    raw_data['week'] = raw_data['InvoiceDate'].dt.isocalendar().week
    raw_data['year'] = raw_data['InvoiceDate'].dt.year
    raw_data['year_week'] = raw_data['year'].astype(str) + '_' + raw_data['week'].astype(str)
    
    print(f"Loaded {len(raw_data)} transactions for top 100 products")
    return raw_data

def aggregate_data(raw_data):
    """Aggregate to product-week level."""
    print("Aggregating to product-week level...")
    
    # Group by StockCode and year_week
    agg_data = raw_data.groupby(['StockCode', 'year_week']).agg({
        'Quantity': 'sum',
        'UnitPrice': 'last',
        'Country': 'first'
    }).reset_index()
    
    # Calculate profit
    agg_data['profit'] = agg_data['Quantity'] * agg_data['UnitPrice']
    
    # Sort and create week numbers
    agg_data = agg_data.sort_values(['StockCode', 'year_week'])
    unique_weeks = sorted(agg_data['year_week'].unique())
    week_mapping = {week: i for i, week in enumerate(unique_weeks)}
    agg_data['week_num'] = agg_data['year_week'].map(week_mapping)
    
    # Convert StockCode to string
    agg_data['StockCode'] = agg_data['StockCode'].astype(str)
    
    print(f"Aggregated to {len(agg_data)} product-week records")
    print(f"Unique products: {agg_data['StockCode'].nunique()}")
    print(f"Unique weeks: {agg_data['week_num'].nunique()}")
    
    return agg_data

def create_price_grid(agg_data):
    """Create discretized price grid."""
    price_5th = agg_data['UnitPrice'].quantile(0.05)
    price_95th = agg_data['UnitPrice'].quantile(0.95)
    price_grid = np.linspace(price_5th, price_95th, 10)  # 10 price points for demo
    
    print(f"Price grid: 10 points from {price_5th:.2f} to {price_95th:.2f}")
    return price_grid

def engineer_features(agg_data):
    """Engineer features for bandit context."""
    print("Engineering features...")
    
    # Encode StockCode
    stock_encoder = LabelEncoder()
    agg_data['stock_code_encoded'] = stock_encoder.fit_transform(agg_data['StockCode'])
    
    # Create lag features
    agg_data['prev_week_sales'] = agg_data.groupby('StockCode')['Quantity'].shift(1).fillna(0)
    
    # Normalize features
    agg_data['week_normalized'] = agg_data['week_num'] / agg_data['week_num'].max()
    
    price_mean = agg_data['UnitPrice'].mean()
    price_std = agg_data['UnitPrice'].std()
    agg_data['price_normalized'] = (agg_data['UnitPrice'] - price_mean) / price_std
    
    sales_mean = agg_data['Quantity'].mean()
    sales_std = agg_data['Quantity'].std()
    agg_data['prev_sales_normalized'] = (agg_data['prev_week_sales'] - sales_mean) / sales_std
    agg_data['prev_sales_normalized'] = agg_data['prev_sales_normalized'].fillna(0)
    
    print("Features created: stock_code, week, price, prev_sales")
    return agg_data

def create_context_vector(row):
    """Create context vector for a product-week."""
    stock_code_feature = row['stock_code_encoded'] / 100  # Normalize by number of products
    
    context = np.array([
        stock_code_feature,
        row['week_normalized'],
        row['price_normalized'],
        row['prev_sales_normalized']
    ])
    
    return context

def simulate_bandit_optimization(agg_data, price_grid, train_ratio=0.7):
    """Simulate bandit optimization."""
    print("Initializing LinUCB bandit...")
    
    # Initialize bandit
    bandit = LinUCB(n_arms=len(price_grid), context_dim=4, alpha=1.0)
    
    # Split data
    unique_weeks = sorted(agg_data['week_num'].unique())
    split_week_idx = int(len(unique_weeks) * train_ratio)
    split_week = unique_weeks[split_week_idx]
    
    train_data = agg_data[agg_data['week_num'] < split_week].copy()
    test_data = agg_data[agg_data['week_num'] >= split_week].copy()
    
    print(f"Train: weeks 0 to {split_week-1} ({len(train_data)} records)")
    print(f"Test: weeks {split_week} onwards ({len(test_data)} records)")
    
    # Training phase
    print("\n=== TRAINING PHASE ===")
    train_profits = []
    cumulative_train_profit = 0
    
    for week_num in sorted(train_data['week_num'].unique()):
        week_data = train_data[train_data['week_num'] == week_num]
        week_profit = 0
        
        for _, row in week_data.iterrows():
            context = create_context_vector(row)
            selected_arm = bandit.select_arm(context)
            selected_price = price_grid[selected_arm]
            
            # Simple demand simulation
            price_ratio = selected_price / row['UnitPrice']
            simulated_quantity = row['Quantity'] * (1 + 0.3 * (1 - price_ratio))
            simulated_quantity = max(0, simulated_quantity)
            
            profit = selected_price * simulated_quantity
            week_profit += profit
            bandit.update(selected_arm, context, profit)
        
        cumulative_train_profit += week_profit
        train_profits.append(week_profit)
        
        if week_num % 5 == 0:
            print(f"Week {week_num}: Profit = {week_profit:.2f}, Cumulative = {cumulative_train_profit:.2f}")
    
    # Testing phase
    print("\n=== TESTING PHASE ===")
    test_profits = []
    cumulative_test_profit = 0
    
    for week_num in sorted(test_data['week_num'].unique()):
        week_data = test_data[test_data['week_num'] == week_num]
        week_profit = 0
        
        for _, row in week_data.iterrows():
            context = create_context_vector(row)
            selected_arm = bandit.select_arm(context)
            selected_price = price_grid[selected_arm]
            
            # Simple demand simulation
            price_ratio = selected_price / row['UnitPrice']
            simulated_quantity = row['Quantity'] * (1 + 0.3 * (1 - price_ratio))
            simulated_quantity = max(0, simulated_quantity)
            
            profit = selected_price * simulated_quantity
            week_profit += profit
        
        cumulative_test_profit += week_profit
        test_profits.append(week_profit)
        
        if week_num % 3 == 0:
            print(f"Week {week_num}: Profit = {week_profit:.2f}, Cumulative = {cumulative_test_profit:.2f}")
    
    return {
        'train_profits': train_profits,
        'test_profits': test_profits,
        'cumulative_train_profit': cumulative_train_profit,
        'cumulative_test_profit': cumulative_test_profit
    }

def plot_results(results):
    """Plot the results."""
    print("\nCreating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Cumulative profit
    train_weeks = range(len(results['train_profits']))
    test_weeks = range(len(results['train_profits']), 
                       len(results['train_profits']) + len(results['test_profits']))
    
    axes[0, 0].plot(train_weeks, np.cumsum(results['train_profits']), 
                    'b-', label='Training', linewidth=2)
    axes[0, 0].plot(test_weeks, np.cumsum(results['test_profits']), 
                    'r-', label='Testing', linewidth=2)
    axes[0, 0].set_title('Cumulative Profit Over Time', fontweight='bold')
    axes[0, 0].set_xlabel('Week')
    axes[0, 0].set_ylabel('Cumulative Profit')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Weekly profit
    axes[0, 1].plot(train_weeks, results['train_profits'], 
                    'b-', label='Training', alpha=0.7)
    axes[0, 1].plot(test_weeks, results['test_profits'], 
                    'r-', label='Testing', alpha=0.7)
    axes[0, 1].set_title('Weekly Profit', fontweight='bold')
    axes[0, 1].set_xlabel('Week')
    axes[0, 1].set_ylabel('Weekly Profit')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Profit distribution
    axes[1, 0].hist(results['train_profits'], bins=10, alpha=0.7, 
                    label='Training', color='blue')
    axes[1, 0].hist(results['test_profits'], bins=10, alpha=0.7, 
                    label='Testing', color='red')
    axes[1, 0].set_title('Profit Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Weekly Profit')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Performance summary
    summary_data = {
        'Metric': ['Total Profit', 'Avg Weekly Profit', 'Max Weekly Profit', 'Min Weekly Profit'],
        'Training': [
            results['cumulative_train_profit'],
            np.mean(results['train_profits']),
            np.max(results['train_profits']),
            np.min(results['train_profits'])
        ],
        'Testing': [
            results['cumulative_test_profit'],
            np.mean(results['test_profits']),
            np.max(results['test_profits']),
            np.min(results['test_profits'])
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    table = axes[1, 1].table(cellText=summary_df.values, 
                             colLabels=summary_df.columns,
                             cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    axes[1, 1].set_title('Performance Summary', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('uci_retail_demo_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Results saved to 'uci_retail_demo_results.png'")

def main():
    """Main demo function."""
    print("=" * 50)
    print("UCI Online Retail Contextual Bandit Demo")
    print("=" * 50)
    
    # Step 1: Load and preprocess data
    raw_data = load_and_preprocess_data()
    
    # Step 2: Aggregate data
    agg_data = aggregate_data(raw_data)
    
    # Step 3: Create price grid
    price_grid = create_price_grid(agg_data)
    
    # Step 4: Engineer features
    agg_data = engineer_features(agg_data)
    
    # Step 5: Run bandit optimization
    results = simulate_bandit_optimization(agg_data, price_grid)
    
    # Step 6: Plot results
    plot_results(results)
    
    # Step 7: Print summary
    print("\n" + "=" * 30)
    print("DEMO SUMMARY")
    print("=" * 30)
    print(f"Training profit: {results['cumulative_train_profit']:.2f}")
    print(f"Testing profit: {results['cumulative_test_profit']:.2f}")
    print(f"Total profit: {results['cumulative_train_profit'] + results['cumulative_test_profit']:.2f}")
    print(f"Average training weekly profit: {np.mean(results['train_profits']):.2f}")
    print(f"Average testing weekly profit: {np.mean(results['test_profits']):.2f}")
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main() 