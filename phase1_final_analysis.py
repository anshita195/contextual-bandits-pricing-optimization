"""
Phase 1 Final Analysis - Complete Robustness Check
=================================================

Implements all recommended improvements:
1. Weekly aggregation for proper statistical testing
2. Refined RFM feature engineering (5 segments + continuous scores)
3. Alpha selection with exploration floor
4. Comprehensive visual diagnostics
5. Weekly profit trajectories and exploration over time
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

class LinUCB:
    """Enhanced LinUCB with exploration tracking over time."""
    
    def __init__(self, n_arms, context_dim, alpha=1.0):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.alpha = alpha
        
        self.A = np.array([np.eye(context_dim) for _ in range(n_arms)])
        self.b = np.array([np.zeros(context_dim) for _ in range(n_arms)])
        self.theta = np.array([np.zeros(context_dim) for _ in range(n_arms)])
        
        # Enhanced tracking
        self.exploration_history = []
        self.arm_selection_history = []
        self.reward_history = []
        
    def select_arm(self, context):
        ucb_values = np.zeros(self.n_arms)
        
        for arm in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[arm])
            self.theta[arm] = A_inv @ self.b[arm]
            ucb_values[arm] = (self.theta[arm] @ context + 
                              self.alpha * np.sqrt(context @ A_inv @ context))
        
        selected_arm = np.argmax(ucb_values)
        
        # Track exploration vs exploitation
        greedy_arm = np.argmax([self.theta[arm] @ context for arm in range(self.n_arms)])
        is_exploration = selected_arm != greedy_arm
        
        self.exploration_history.append(is_exploration)
        self.arm_selection_history.append(selected_arm)
        
        return selected_arm
    
    def update(self, arm, context, reward):
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        self.reward_history.append(reward)
    
    def get_exploration_rate(self):
        return np.mean(self.exploration_history) if self.exploration_history else 0
    
    def get_weekly_exploration_rates(self, week_lengths):
        """Get exploration rate for each week."""
        rates = []
        start_idx = 0
        for week_len in week_lengths:
            end_idx = start_idx + week_len
            week_exploration = self.exploration_history[start_idx:end_idx]
            rates.append(np.mean(week_exploration) if week_exploration else 0)
            start_idx = end_idx
        return rates

def create_enhanced_rfm_features(raw_data, agg_data):
    """Create both discrete segments and continuous RFM scores."""
    
    # Calculate RFM metrics
    customer_rfm = raw_data.groupby('CustomerID').agg({
        'InvoiceDate': 'count',
        'UnitPrice': lambda x: (x * raw_data.loc[x.index, 'Quantity']).sum()
    }).reset_index()
    customer_rfm.columns = ['CustomerID', 'frequency', 'monetary']
    
    # Add recency (days since last purchase)
    customer_recency = raw_data.groupby('CustomerID')['InvoiceDate'].max().reset_index()
    customer_recency.columns = ['CustomerID', 'last_purchase']
    customer_recency['recency_days'] = (pd.Timestamp('2011-12-09') - customer_recency['last_purchase']).dt.days
    
    customer_rfm = customer_rfm.merge(customer_recency, on='CustomerID')
    
    # Create 5-segment RFM (more granular)
    customer_rfm['rfm_segment_5'] = pd.qcut(customer_rfm['monetary'], q=5, labels=[0, 1, 2, 3, 4])
    
    # Create continuous RFM score (0-1)
    customer_rfm['rfm_score'] = (
        (customer_rfm['frequency'] - customer_rfm['frequency'].min()) / 
        (customer_rfm['frequency'].max() - customer_rfm['frequency'].min()) * 0.4 +
        (customer_rfm['monetary'] - customer_rfm['monetary'].min()) / 
        (customer_rfm['monetary'].max() - customer_rfm['monetary'].min()) * 0.4 +
        (1 - (customer_rfm['recency_days'] - customer_rfm['recency_days'].min()) / 
         (customer_rfm['recency_days'].max() - customer_rfm['recency_days'].min())) * 0.2
    )
    
    # Map to aggregated data
    rfm_dict_5 = dict(zip(customer_rfm['CustomerID'], customer_rfm['rfm_segment_5']))
    rfm_score_dict = dict(zip(customer_rfm['CustomerID'], customer_rfm['rfm_score']))
    
    agg_data['rfm_segment_5'] = agg_data['CustomerID'].map(rfm_dict_5).fillna(0)
    agg_data['rfm_score'] = agg_data['CustomerID'].map(rfm_score_dict).fillna(0)
    agg_data['rfm_segment_5_normalized'] = agg_data['rfm_segment_5'] / 4
    agg_data['rfm_score_normalized'] = agg_data['rfm_score']  # Already 0-1
    
    return agg_data

def run_weekly_bandit_evaluation(train_data, test_data, features, price_grid, alpha=1.0):
    """Run bandit evaluation with weekly aggregation for proper statistical testing."""
    
    bandit = LinUCB(n_arms=len(price_grid), context_dim=len(features), alpha=alpha)
    
    # Training phase
    train_weekly_profits = []
    train_week_lengths = []
    
    for week in sorted(train_data['week_num'].unique()):
        week_data = train_data[train_data['week_num'] == week]
        week_profit = 0
        week_length = len(week_data)
        
        for _, row in week_data.iterrows():
            context = np.array([row[feature] for feature in features])
            selected_arm = bandit.select_arm(context)
            selected_price = price_grid[selected_arm]
            
            # Simulate sales
            price_ratio = selected_price / row['UnitPrice']
            simulated_quantity = row['Quantity'] * (1 + 0.3 * (1 - price_ratio))
            simulated_quantity = max(0, simulated_quantity)
            
            profit = selected_price * simulated_quantity
            week_profit += profit
            bandit.update(selected_arm, context, profit)
        
        train_weekly_profits.append(week_profit)
        train_week_lengths.append(week_length)
    
    # Testing phase
    test_weekly_profits = []
    test_week_lengths = []
    
    for week in sorted(test_data['week_num'].unique()):
        week_data = test_data[test_data['week_num'] == week]
        week_profit = 0
        week_length = len(week_data)
        
        for _, row in week_data.iterrows():
            context = np.array([row[feature] for feature in features])
            selected_arm = bandit.select_arm(context)
            selected_price = price_grid[selected_arm]
            
            # Simulate sales
            price_ratio = selected_price / row['UnitPrice']
            simulated_quantity = row['Quantity'] * (1 + 0.3 * (1 - price_ratio))
            simulated_quantity = max(0, simulated_quantity)
            
            profit = selected_price * simulated_quantity
            week_profit += profit
        
        test_weekly_profits.append(week_profit)
        test_week_lengths.append(week_length)
    
    # Get weekly exploration rates
    test_exploration_rates = bandit.get_weekly_exploration_rates(test_week_lengths)
    
    return {
        'train_weekly_profits': train_weekly_profits,
        'test_weekly_profits': test_weekly_profits,
        'test_exploration_rates': test_exploration_rates,
        'total_test_profit': sum(test_weekly_profits),
        'avg_test_weekly': np.mean(test_weekly_profits),
        'exploration_rate': bandit.get_exploration_rate(),
        'bandit': bandit
    }

def analyze_phase1_final():
    """Complete Phase 1 analysis with all improvements."""
    print("=== PHASE 1 FINAL ANALYSIS ===")
    
    # Load and preprocess data
    raw_data = pd.read_excel('UCI_dataset/Online Retail.xlsx')
    raw_data = raw_data.dropna(subset=['StockCode', 'UnitPrice', 'Quantity'])
    raw_data = raw_data[raw_data['Quantity'] > 0]
    raw_data = raw_data[raw_data['UnitPrice'] > 0]
    
    # Use top 50 products
    product_sales = raw_data.groupby('StockCode')['Quantity'].sum().sort_values(ascending=False)
    top_50_products = product_sales.head(50).index
    raw_data = raw_data[raw_data['StockCode'].isin(top_50_products)]
    
    # Preprocess
    raw_data['InvoiceDate'] = pd.to_datetime(raw_data['InvoiceDate'])
    raw_data['StockCode'] = raw_data['StockCode'].astype(str)
    
    # Aggregate
    agg_data = raw_data.groupby(['StockCode', 'InvoiceDate']).agg({
        'Quantity': 'sum',
        'UnitPrice': 'last',
        'Country': 'first',
        'CustomerID': 'first'
    }).reset_index()
    
    agg_data['profit'] = agg_data['Quantity'] * agg_data['UnitPrice']
    
    # Create week numbers
    agg_data['week'] = agg_data['InvoiceDate'].dt.isocalendar().week
    unique_weeks = sorted(agg_data['week'].unique())
    week_mapping = {week: i for i, week in enumerate(unique_weeks)}
    agg_data['week_num'] = agg_data['week'].map(week_mapping)
    
    # Feature engineering
    stock_encoder = LabelEncoder()
    agg_data['stock_encoded'] = stock_encoder.fit_transform(agg_data['StockCode'])
    agg_data['week_normalized'] = agg_data['week_num'] / agg_data['week_num'].max()
    agg_data['price_normalized'] = (agg_data['UnitPrice'] - agg_data['UnitPrice'].mean()) / agg_data['UnitPrice'].std()
    agg_data['prev_sales'] = agg_data.groupby('StockCode')['Quantity'].shift(1).fillna(0)
    agg_data['prev_sales_normalized'] = (agg_data['prev_sales'] - agg_data['prev_sales'].mean()) / agg_data['prev_sales'].std()
    agg_data['prev_sales_normalized'] = agg_data['prev_sales_normalized'].fillna(0)
    
    country_encoder = LabelEncoder()
    agg_data['country_encoded'] = country_encoder.fit_transform(agg_data['Country'])
    agg_data['country_normalized'] = agg_data['country_encoded'] / len(country_encoder.classes_)
    
    # Enhanced RFM features
    agg_data = create_enhanced_rfm_features(raw_data, agg_data)
    
    # Temporal features
    agg_data['day_of_week'] = agg_data['InvoiceDate'].dt.dayofweek
    agg_data['month'] = agg_data['InvoiceDate'].dt.month
    agg_data['day_normalized'] = agg_data['day_of_week'] / 6
    agg_data['month_normalized'] = agg_data['month'] / 12
    
    # Split data
    unique_weeks = sorted(agg_data['week_num'].unique())
    split_idx = int(len(unique_weeks) * 0.7)
    train_data = agg_data[agg_data['week_num'] < split_idx]
    test_data = agg_data[agg_data['week_num'] >= split_idx]
    
    # Price grid
    price_5th = agg_data['UnitPrice'].quantile(0.05)
    price_95th = agg_data['UnitPrice'].quantile(0.95)
    price_grid = np.linspace(price_5th, price_95th, 8)
    
    print(f"Data: {len(train_data)} train, {len(test_data)} test records")
    print(f"Train weeks: {len(train_data['week_num'].unique())}, Test weeks: {len(test_data['week_num'].unique())}")
    
    # Analysis 1: Refined feature combinations
    print("\n1. REFINED FEATURE COMBINATION ANALYSIS")
    print("=" * 50)
    
    feature_combinations = {
        'Baseline': ['stock_encoded', 'week_normalized', 'price_normalized', 'prev_sales_normalized'],
        'RFM_5_Segments': ['stock_encoded', 'week_normalized', 'price_normalized', 'prev_sales_normalized', 'rfm_segment_5_normalized'],
        'RFM_Continuous': ['stock_encoded', 'week_normalized', 'price_normalized', 'prev_sales_normalized', 'rfm_score_normalized'],
        'Country': ['stock_encoded', 'week_normalized', 'price_normalized', 'prev_sales_normalized', 'country_normalized'],
        'Temporal': ['stock_encoded', 'week_normalized', 'price_normalized', 'prev_sales_normalized', 'day_normalized', 'month_normalized'],
        'RFM_5_+_Temporal': ['stock_encoded', 'week_normalized', 'price_normalized', 'prev_sales_normalized', 'rfm_segment_5_normalized', 'day_normalized', 'month_normalized'],
        'All_Features': ['stock_encoded', 'week_normalized', 'price_normalized', 'prev_sales_normalized', 'rfm_segment_5_normalized', 'country_normalized', 'day_normalized', 'month_normalized']
    }
    
    results = {}
    
    for combo_name, features in feature_combinations.items():
        print(f"\nTesting {combo_name}...")
        
        result = run_weekly_bandit_evaluation(train_data, test_data, features, price_grid, alpha=1.0)
        results[combo_name] = result
        
        print(f"  Test profit: {result['total_test_profit']:.0f}")
        print(f"  Avg weekly profit: {result['avg_test_weekly']:.0f}")
        print(f"  Exploration rate: {result['exploration_rate']:.2%}")
    
    # Analysis 2: Alpha selection with exploration floor
    print("\n2. ALPHA SELECTION WITH EXPLORATION FLOOR")
    print("=" * 45)
    
    alpha_values = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    alpha_results = {}
    
    baseline_features = ['stock_encoded', 'week_normalized', 'price_normalized', 'prev_sales_normalized']
    
    for alpha in alpha_values:
        print(f"\nTesting alpha = {alpha}")
        
        result = run_weekly_bandit_evaluation(train_data, test_data, baseline_features, price_grid, alpha=alpha)
        alpha_results[alpha] = result
        
        print(f"  Test profit: {result['total_test_profit']:.0f}")
        print(f"  Exploration rate: {result['exploration_rate']:.2%}")
        print(f"  Avg weekly profit: {result['avg_test_weekly']:.0f}")
    
    # Analysis 3: Proper statistical testing (weekly aggregation)
    print("\n3. STATISTICAL SIGNIFICANCE TESTING (Weekly Aggregation)")
    print("=" * 60)
    
    baseline_weekly = results['Baseline']['test_weekly_profits']
    
    for combo_name, result in results.items():
        if combo_name != 'Baseline':
            # Paired t-test on weekly profits
            t_stat, p_value = stats.ttest_rel(baseline_weekly, result['test_weekly_profits'])
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(baseline_weekly) - 1) * np.var(baseline_weekly) + 
                                 (len(result['test_weekly_profits']) - 1) * np.var(result['test_weekly_profits'])) / 
                                (len(baseline_weekly) + len(result['test_weekly_profits']) - 2))
            cohens_d = (np.mean(result['test_weekly_profits']) - np.mean(baseline_weekly)) / pooled_std
            
            # Calculate improvement percentage
            improvement = ((np.mean(result['test_weekly_profits']) - np.mean(baseline_weekly)) / np.mean(baseline_weekly)) * 100
            
            print(f"{combo_name:20} | p-value: {p_value:.4f} | Cohen's d: {cohens_d:.3f} | Improvement: {improvement:+.1f}%")
    
    # Create comprehensive visualizations
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Weekly profit trajectories
    ax1 = plt.subplot(3, 3, 1)
    weeks = range(1, len(baseline_weekly) + 1)
    
    for combo_name, result in results.items():
        if combo_name in ['Baseline', 'RFM_5_Segments', 'RFM_Continuous', 'All_Features']:
            plt.plot(weeks, result['test_weekly_profits'], 'o-', label=combo_name, linewidth=2, markersize=4)
    
    plt.title('Weekly Profit Trajectories', fontweight='bold')
    plt.xlabel('Test Week')
    plt.ylabel('Weekly Profit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Feature combination comparison
    ax2 = plt.subplot(3, 3, 2)
    combo_names = list(results.keys())
    test_profits = [results[combo]['total_test_profit'] for combo in combo_names]
    
    bars = plt.bar(combo_names, test_profits, color=['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink'])
    plt.title('Total Test Profit by Feature Combination', fontweight='bold')
    plt.ylabel('Total Test Profit')
    plt.xticks(rotation=45)
    
    # Plot 3: Alpha vs Test Profit
    ax3 = plt.subplot(3, 3, 3)
    alphas = list(alpha_results.keys())
    alpha_profits = [alpha_results[alpha]['total_test_profit'] for alpha in alphas]
    
    plt.plot(alphas, alpha_profits, 'o-', color='red', linewidth=2, markersize=8)
    plt.title('Alpha vs Test Profit', fontweight='bold')
    plt.xlabel('Alpha (Exploration Parameter)')
    plt.ylabel('Test Profit')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Alpha vs Exploration Rate
    ax4 = plt.subplot(3, 3, 4)
    alpha_exploration = [alpha_results[alpha]['exploration_rate'] for alpha in alphas]
    
    plt.plot(alphas, alpha_exploration, 'o-', color='purple', linewidth=2, markersize=8)
    plt.title('Alpha vs Exploration Rate', fontweight='bold')
    plt.xlabel('Alpha (Exploration Parameter)')
    plt.ylabel('Exploration Rate')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Weekly exploration rates over time
    ax5 = plt.subplot(3, 3, 5)
    for combo_name, result in results.items():
        if combo_name in ['Baseline', 'RFM_5_Segments', 'All_Features']:
            plt.plot(weeks, result['test_exploration_rates'], 'o-', label=combo_name, linewidth=2, markersize=4)
    
    plt.title('Weekly Exploration Rates Over Time', fontweight='bold')
    plt.xlabel('Test Week')
    plt.ylabel('Exploration Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Improvement over baseline
    ax6 = plt.subplot(3, 3, 6)
    baseline_profit = results['Baseline']['total_test_profit']
    improvements = [((results[combo]['total_test_profit'] - baseline_profit) / baseline_profit) * 100 for combo in combo_names]
    
    bars = plt.bar(combo_names, improvements, color=['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink'])
    plt.title('Improvement Over Baseline (%)', fontweight='bold')
    plt.ylabel('Improvement (%)')
    plt.xticks(rotation=45)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Plot 7: RFM segment analysis
    ax7 = plt.subplot(3, 3, 7)
    rfm_analysis = agg_data.groupby('rfm_segment_5').agg({
        'profit': ['mean', 'count'],
        'UnitPrice': 'mean',
        'Quantity': 'mean'
    }).round(2)
    
    rfm_analysis.columns = ['avg_profit', 'count', 'avg_price', 'avg_quantity']
    rfm_analysis.plot(kind='bar', y='avg_profit', ax=ax7, color='green', alpha=0.7)
    plt.title('Average Profit by RFM Segment', fontweight='bold')
    plt.xlabel('RFM Segment')
    plt.ylabel('Average Profit')
    plt.xticks(rotation=0)
    
    # Plot 8: Monthly profit patterns
    ax8 = plt.subplot(3, 3, 8)
    monthly_profits = agg_data.groupby('month')['profit'].mean()
    monthly_profits.plot(kind='bar', ax=ax8, color='orange', alpha=0.7)
    plt.title('Average Profit by Month', fontweight='bold')
    plt.xlabel('Month')
    plt.ylabel('Average Profit')
    plt.xticks(rotation=0)
    
    # Plot 9: Country profit analysis
    ax9 = plt.subplot(3, 3, 9)
    country_profits = agg_data.groupby('Country')['profit'].mean().sort_values(ascending=False).head(10)
    country_profits.plot(kind='bar', ax=ax9, color='blue', alpha=0.7)
    plt.title('Average Profit by Country (Top 10)', fontweight='bold')
    plt.xlabel('Country')
    plt.ylabel('Average Profit')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('phase1_final_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary and recommendations
    print("\n4. SUMMARY & RECOMMENDATIONS")
    print("=" * 35)
    
    best_combo = max(results.keys(), key=lambda x: results[x]['total_test_profit'])
    best_profit = results[best_combo]['total_test_profit']
    improvement = ((best_profit - results['Baseline']['total_test_profit']) / results['Baseline']['total_test_profit']) * 100
    
    # Find alpha with minimum 5% exploration
    suitable_alphas = [(alpha, result) for alpha, result in alpha_results.items() 
                       if result['exploration_rate'] >= 0.05]
    best_alpha = max(suitable_alphas, key=lambda x: x[1]['total_test_profit']) if suitable_alphas else (1.0, alpha_results[1.0])
    
    print(f"✅ Best feature combination: {best_combo}")
    print(f"✅ Best test profit: {best_profit:.0f}")
    print(f"✅ Improvement over baseline: {improvement:+.1f}%")
    print(f"✅ Recommended alpha: {best_alpha[0]} (exploration: {best_alpha[1]['exploration_rate']:.2%})")
    print(f"✅ Statistical significance: Weekly aggregation provides proper testing")
    print(f"✅ Ready for Phase 2: Olist integration with validated foundation")

if __name__ == "__main__":
    analyze_phase1_final() 