# EXPORT SHARE BY REGION ANALYSIS (COMBINED OTHER CATEGORIES)
print("=== EXPORT SHARE BY REGION ANALYSIS ===")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from region_mapping import add_region_to_dataframe

# Load the data
df = pd.read_csv('data/exports_cleaned.csv')

# 1. APPLY REGIONAL MAPPING
# Add region column to existing dataframe
df = add_region_to_dataframe(df, 'country_of_destination', 'region')

# 2. REGIONAL EXPORT SHARE ANALYSIS
regional_analysis = df.groupby('region').agg({
    'value_fob_aud': ['sum', 'count'],
    'gross_weight_tonnes': 'sum',
    'country_of_destination': 'nunique'
}).reset_index()

# Flatten column names
regional_analysis.columns = ['region', 'total_value', 'total_shipments', 'total_weight', 'num_countries']
regional_analysis['value_billions'] = regional_analysis['total_value'] / 1e9
regional_analysis['weight_millions'] = regional_analysis['total_weight'] / 1e6
regional_analysis['market_share_pct'] = (regional_analysis['total_value'] / regional_analysis['total_value'].sum()) * 100
regional_analysis['avg_value_per_shipment'] = regional_analysis['total_value'] / regional_analysis['total_shipments']

# Sort by market share
regional_analysis = regional_analysis.sort_values('market_share_pct', ascending=False)

# 3. COMBINE SMALL REGIONS WITH "OTHER"
# Create a modified dataframe for visualization
viz_data = regional_analysis.copy()

# Find regions with <1% market share (excluding the main "Other" category)
small_regions_mask = (viz_data['market_share_pct'] < 1.0) & (viz_data['region'] != 'Other')
small_regions = viz_data[small_regions_mask]

if len(small_regions) > 0:
    # Calculate total share of small regions
    small_regions_share = small_regions['market_share_pct'].sum()
    small_regions_value = small_regions['total_value'].sum()
    small_regions_shipments = small_regions['total_shipments'].sum()
    small_regions_weight = small_regions['total_weight'].sum()
    small_regions_countries = small_regions['num_countries'].sum()
    
    # Add small regions to the main "Other" category
    other_mask = viz_data['region'] == 'Other'
    if other_mask.any():
        # Update the existing "Other" category
        viz_data.loc[other_mask, 'market_share_pct'] += small_regions_share
        viz_data.loc[other_mask, 'total_value'] += small_regions_value
        viz_data.loc[other_mask, 'total_shipments'] += small_regions_shipments
        viz_data.loc[other_mask, 'total_weight'] += small_regions_weight
        viz_data.loc[other_mask, 'num_countries'] += small_regions_countries
        viz_data.loc[other_mask, 'value_billions'] = viz_data.loc[other_mask, 'total_value'] / 1e9
        viz_data.loc[other_mask, 'weight_millions'] = viz_data.loc[other_mask, 'total_weight'] / 1e6
        viz_data.loc[other_mask, 'avg_value_per_shipment'] = viz_data.loc[other_mask, 'total_value'] / viz_data.loc[other_mask, 'total_shipments']
    
    # Remove small regions from the visualization data
    viz_data = viz_data[~small_regions_mask]
    
    print(f"Combined {len(small_regions)} small regions with 'Other' category")
    print(f"Small regions combined: {', '.join(small_regions['region'].tolist())}")
    print(f"Total small regions share: {small_regions_share:.1f}%")

# Re-sort after combining
viz_data = viz_data.sort_values('market_share_pct', ascending=False)

print(f"\n📊 COMBINED REGIONAL EXPORT SHARE:")
print("=" * 60)
for i, (_, region) in enumerate(viz_data.iterrows(), 1):
    print(f"{i}. {region['region']:<25} {region['market_share_pct']:>6.1f}% (${region['value_billions']:>6.1f}B)")

# 4. REGIONAL VISUALIZATIONS (MATPLOTLIB)
plt.style.use('default')
fig = plt.figure(figsize=(18, 10))

# 4.1 Regional Market Share Pie Chart (Combined Others)
ax1 = plt.subplot(1, 2, 1)
colors = plt.cm.Set3(np.linspace(0, 1, len(viz_data)))

wedges, texts, autotexts = ax1.pie(
    viz_data['market_share_pct'], 
    labels=viz_data['region'],
    autopct='%1.1f%%',
    colors=colors,
    startangle=90
)
ax1.set_title('Australian Export Share by Region\n(Combined Other Categories)', fontsize=16, fontweight='bold')
plt.setp(autotexts, size=11, weight='bold')

# 4.2 Regional Value Comparison Bar Chart
ax2 = plt.subplot(1, 2, 2)
bars = ax2.bar(range(len(viz_data)), viz_data['value_billions'], 
               color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax2.set_title('Export Value by Region (Billions AUD)', fontsize=16, fontweight='bold')
ax2.set_xlabel('Region', fontsize=14)
ax2.set_ylabel('Export Value (Billions AUD)', fontsize=14)
ax2.set_xticks(range(len(viz_data)))
ax2.set_xticklabels(viz_data['region'], rotation=45, ha='right', fontsize=11)
ax2.grid(True, alpha=0.3)

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, viz_data['value_billions'])):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'${value:.1f}B', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# 5. DETAILED REGIONAL BREAKDOWN
print(f"\n📋 DETAILED REGIONAL BREAKDOWN:")
print("=" * 60)
for _, region in viz_data.iterrows():
    print(f"\n🌍 {region['region']}:")
    print(f"   Market Share: {region['market_share_pct']:.1f}%")
    print(f"   Export Value: ${region['value_billions']:.1f}B")
    print(f"   Countries: {region['num_countries']}")
    print(f"   Shipments: {region['total_shipments']:,}")
    print(f"   Weight: {region['weight_millions']:.1f}M tonnes")

# 6. STRATEGIC INSIGHTS
print(f"\n🎯 KEY INSIGHTS:")
print("=" * 40)

top_region = viz_data.iloc[0]
top_3_share = viz_data.head(3)['market_share_pct'].sum()

print(f"🏆 Top Region: {top_region['region']} ({top_region['market_share_pct']:.1f}%)")
print(f"📊 Top 3 regions: {top_3_share:.1f}% of exports")
print(f"🌍 Total regions: {len(viz_data)}")

# Find expansion opportunities
emerging_regions = viz_data[
    (viz_data['market_share_pct'] < 5.0) & 
    (viz_data['num_countries'] >= 3)
].sort_values('market_share_pct', ascending=True)

if len(emerging_regions) > 0:
    print(f"\n🚀 Expansion Opportunities:")
    for _, region in emerging_regions.head(3).iterrows():
        print(f"   • {region['region']}: {region['market_share_pct']:.1f}% share, {region['num_countries']} countries")

print("\n✅ Regional Analysis Complete!")
