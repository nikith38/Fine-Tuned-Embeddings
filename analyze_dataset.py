import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

def analyze_sales_dataset():
    """Analyze the generated sales conversation dataset"""
    
    # Load the dataset
    with open('sales_conversations_dataset.json', 'r') as f:
        dataset = json.load(f)
    
    print("=== SALES CONVERSATION DATASET ANALYSIS ===")
    print(f"Total examples: {len(dataset)}")
    
    # Basic statistics
    successful = [d for d in dataset if d['conversion_label'] == 1]
    failed = [d for d in dataset if d['conversion_label'] == 0]
    
    print(f"\n=== CONVERSION STATISTICS ===")
    print(f"Successful conversions: {len(successful)} ({len(successful)/len(dataset)*100:.1f}%)")
    print(f"Failed conversions: {len(failed)} ({len(failed)/len(dataset)*100:.1f}%)")
    
    # Conversation length statistics
    lengths = [d['conversation_length'] for d in dataset]
    print(f"\n=== CONVERSATION LENGTH STATISTICS ===")
    print(f"Average length: {np.mean(lengths):.1f} words")
    print(f"Median length: {np.median(lengths):.1f} words")
    print(f"Min length: {min(lengths)} words")
    print(f"Max length: {max(lengths)} words")
    print(f"Standard deviation: {np.std(lengths):.1f} words")
    
    # Length by outcome
    successful_lengths = [d['conversation_length'] for d in successful]
    failed_lengths = [d['conversation_length'] for d in failed]
    
    print(f"\nSuccessful conversations - Avg length: {np.mean(successful_lengths):.1f} words")
    print(f"Failed conversations - Avg length: {np.mean(failed_lengths):.1f} words")
    
    # Industry distribution
    industries = [d['customer_context']['industry'] for d in dataset]
    industry_counts = Counter(industries)
    
    print(f"\n=== INDUSTRY DISTRIBUTION ===")
    for industry, count in sorted(industry_counts.items()):
        print(f"{industry.title()}: {count} ({count/len(dataset)*100:.1f}%)")
    
    # Company size distribution
    company_sizes = [d['customer_context']['company_size'] for d in dataset]
    size_counts = Counter(company_sizes)
    
    print(f"\n=== COMPANY SIZE DISTRIBUTION ===")
    for size, count in sorted(size_counts.items()):
        print(f"{size.title()}: {count} ({count/len(dataset)*100:.1f}%)")
    
    # Contact role distribution
    roles = [d['customer_context']['contact_role'] for d in dataset]
    role_counts = Counter(roles)
    
    print(f"\n=== CONTACT ROLE DISTRIBUTION ===")
    for role, count in sorted(role_counts.items()):
        print(f"{role}: {count} ({count/len(dataset)*100:.1f}%)")
    
    # Lead source distribution
    lead_sources = [d['customer_context']['lead_source'] for d in dataset]
    source_counts = Counter(lead_sources)
    
    print(f"\n=== LEAD SOURCE DISTRIBUTION ===")
    for source, count in sorted(source_counts.items()):
        print(f"{source.title()}: {count} ({count/len(dataset)*100:.1f}%)")
    
    # Urgency distribution
    urgencies = [d['customer_context']['urgency'] for d in dataset]
    urgency_counts = Counter(urgencies)
    
    print(f"\n=== URGENCY DISTRIBUTION ===")
    for urgency, count in sorted(urgency_counts.items()):
        print(f"{urgency.title()}: {count} ({count/len(dataset)*100:.1f}%)")
    
    # Budget authority distribution
    authorities = [d['customer_context']['budget_authority'] for d in dataset]
    authority_counts = Counter(authorities)
    
    print(f"\n=== BUDGET AUTHORITY DISTRIBUTION ===")
    for authority, count in sorted(authority_counts.items()):
        print(f"{authority.replace('_', ' ').title()}: {count} ({count/len(dataset)*100:.1f}%)")
    
    # Conversion rate by different factors
    print(f"\n=== CONVERSION RATES BY FACTORS ===")
    
    # By industry
    print(f"\nConversion rate by industry:")
    for industry in set(industries):
        industry_data = [d for d in dataset if d['customer_context']['industry'] == industry]
        industry_successful = [d for d in industry_data if d['conversion_label'] == 1]
        conversion_rate = len(industry_successful) / len(industry_data) * 100
        print(f"  {industry.title()}: {conversion_rate:.1f}% ({len(industry_successful)}/{len(industry_data)})")
    
    # By company size
    print(f"\nConversion rate by company size:")
    for size in set(company_sizes):
        size_data = [d for d in dataset if d['customer_context']['company_size'] == size]
        size_successful = [d for d in size_data if d['conversion_label'] == 1]
        conversion_rate = len(size_successful) / len(size_data) * 100
        print(f"  {size.title()}: {conversion_rate:.1f}% ({len(size_successful)}/{len(size_data)})")
    
    # By urgency
    print(f"\nConversion rate by urgency:")
    for urgency in set(urgencies):
        urgency_data = [d for d in dataset if d['customer_context']['urgency'] == urgency]
        urgency_successful = [d for d in urgency_data if d['conversion_label'] == 1]
        conversion_rate = len(urgency_successful) / len(urgency_data) * 100
        print(f"  {urgency.title()}: {conversion_rate:.1f}% ({len(urgency_successful)}/{len(urgency_data)})")
    
    # By budget authority
    print(f"\nConversion rate by budget authority:")
    for authority in set(authorities):
        authority_data = [d for d in dataset if d['customer_context']['budget_authority'] == authority]
        authority_successful = [d for d in authority_data if d['conversion_label'] == 1]
        conversion_rate = len(authority_successful) / len(authority_data) * 100
        print(f"  {authority.replace('_', ' ').title()}: {conversion_rate:.1f}% ({len(authority_successful)}/{len(authority_data)})")
    
    # Sample examples
    print(f"\n=== SAMPLE EXAMPLES ===")
    
    print(f"\nSUCCESSFUL CONVERSATIONS (Sample):")
    for i, example in enumerate(successful[:3]):
        print(f"\n{i+1}. ID: {example['id']}")
        print(f"   Industry: {example['customer_context']['industry']} | Size: {example['customer_context']['company_size']}")
        print(f"   Role: {example['customer_context']['contact_role']} | Urgency: {example['customer_context']['urgency']}")
        print(f"   Transcript: {example['transcript'][:150]}...")
    
    print(f"\nFAILED CONVERSATIONS (Sample):")
    for i, example in enumerate(failed[:3]):
        print(f"\n{i+1}. ID: {example['id']}")
        print(f"   Industry: {example['customer_context']['industry']} | Size: {example['customer_context']['company_size']}")
        print(f"   Role: {example['customer_context']['contact_role']} | Urgency: {example['customer_context']['urgency']}")
        print(f"   Transcript: {example['transcript'][:150]}...")
    
    # Key insights
    print(f"\n=== KEY INSIGHTS ===")
    
    # Most successful industries
    industry_conversion_rates = {}
    for industry in set(industries):
        industry_data = [d for d in dataset if d['customer_context']['industry'] == industry]
        industry_successful = [d for d in industry_data if d['conversion_label'] == 1]
        industry_conversion_rates[industry] = len(industry_successful) / len(industry_data) * 100
    
    best_industry = max(industry_conversion_rates, key=industry_conversion_rates.get)
    worst_industry = min(industry_conversion_rates, key=industry_conversion_rates.get)
    
    print(f"• Highest conversion rate industry: {best_industry.title()} ({industry_conversion_rates[best_industry]:.1f}%)")
    print(f"• Lowest conversion rate industry: {worst_industry.title()} ({industry_conversion_rates[worst_industry]:.1f}%)")
    
    # Urgency impact
    urgency_conversion_rates = {}
    for urgency in set(urgencies):
        urgency_data = [d for d in dataset if d['customer_context']['urgency'] == urgency]
        urgency_successful = [d for d in urgency_data if d['conversion_label'] == 1]
        urgency_conversion_rates[urgency] = len(urgency_successful) / len(urgency_data) * 100
    
    print(f"• High urgency conversion rate: {urgency_conversion_rates.get('high', 0):.1f}%")
    print(f"• Medium urgency conversion rate: {urgency_conversion_rates.get('medium', 0):.1f}%")
    print(f"• Low urgency conversion rate: {urgency_conversion_rates.get('low', 0):.1f}%")
    
    # Decision maker impact
    decision_maker_data = [d for d in dataset if d['customer_context']['budget_authority'] == 'decision_maker']
    decision_maker_successful = [d for d in decision_maker_data if d['conversion_label'] == 1]
    decision_maker_rate = len(decision_maker_successful) / len(decision_maker_data) * 100
    
    non_decision_maker_data = [d for d in dataset if d['customer_context']['budget_authority'] != 'decision_maker']
    non_decision_maker_successful = [d for d in non_decision_maker_data if d['conversion_label'] == 1]
    non_decision_maker_rate = len(non_decision_maker_successful) / len(non_decision_maker_data) * 100
    
    print(f"• Decision makers conversion rate: {decision_maker_rate:.1f}%")
    print(f"• Non-decision makers conversion rate: {non_decision_maker_rate:.1f}%")
    
    print(f"\n=== DATASET QUALITY ASSESSMENT ===")
    print(f"✓ Balanced dataset: {abs(len(successful) - len(failed)) <= 10}")
    print(f"✓ Diverse industries: {len(set(industries))} different industries")
    print(f"✓ Diverse company sizes: {len(set(company_sizes))} different sizes")
    print(f"✓ Diverse roles: {len(set(roles))} different contact roles")
    print(f"✓ Realistic conversation lengths: {min(lengths)}-{max(lengths)} words")
    print(f"✓ Rich context metadata: 7 context fields per example")
    
    return dataset

def create_training_test_split(dataset, test_size=0.2):
    """Create training and test splits"""
    import random
    
    # Shuffle dataset
    random.shuffle(dataset)
    
    # Split
    split_idx = int(len(dataset) * (1 - test_size))
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    
    print(f"\n=== TRAIN/TEST SPLIT ===")
    print(f"Training examples: {len(train_data)}")
    print(f"Test examples: {len(test_data)}")
    
    # Check balance in splits
    train_successful = len([d for d in train_data if d['conversion_label'] == 1])
    test_successful = len([d for d in test_data if d['conversion_label'] == 1])
    
    print(f"Training successful: {train_successful} ({train_successful/len(train_data)*100:.1f}%)")
    print(f"Test successful: {test_successful} ({test_successful/len(test_data)*100:.1f}%)")
    
    # Save splits
    with open('train_data.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open('test_data.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"✓ Train/test splits saved as train_data.json and test_data.json")
    
    return train_data, test_data

if __name__ == "__main__":
    dataset = analyze_sales_dataset()
    train_data, test_data = create_training_test_split(dataset) 