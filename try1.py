# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests",
#   "beautifulsoup4",
#   "matplotlib",
#   "pandas",
#   "numpy",
#   "scipy",
# ]
# ///

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import json
import re

def scrape_film_data():
    """Scrape film data from Wikipedia's highest-grossing films page."""
    url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    
    try:
        # Fetch the HTML content
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the main table with film data
        # Look for the table with highest-grossing films
        tables = soup.find_all('table', {'class': 'wikitable'})
        
        # The first wikitable should contain the main data
        main_table = tables[0]
        
        # Extract table rows
        rows = main_table.find_all('tr')[1:]  # Skip header row
        
        films_data = []
        
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 5:  # Ensure we have enough columns
                try:
                    # Extract data from cells
                    rank = cells[0].get_text(strip=True)
                    title = cells[1].get_text(strip=True)
                    worldwide_gross = cells[2].get_text(strip=True)
                    year = cells[3].get_text(strip=True)
                    peak = cells[4].get_text(strip=True) if len(cells) > 4 else ""
                    
                    films_data.append({
                        'Rank': rank,
                        'Title': title,
                        'Worldwide gross': worldwide_gross,
                        'Year': year,
                        'Peak': peak
                    })
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue
        
        return films_data
    
    except Exception as e:
        print(f"Error scraping data: {e}")
        return []

def clean_monetary_value(value_str):
    """Clean monetary values by removing $ and commas, converting to float."""
    if not value_str or value_str == '':
        return 0.0
    
    # Remove $ symbol, commas, and any other non-numeric characters except decimal point
    cleaned = re.sub(r'[^\d.]', '', value_str)
    
    try:
        return float(cleaned)
    except ValueError:
        return 0.0

def clean_year(year_str):
    """Extract and clean year value."""
    if not year_str:
        return 0
    
    # Extract first 4-digit number (year)
    year_match = re.search(r'\d{4}', year_str)
    if year_match:
        return int(year_match.group())
    return 0

def clean_data(films_data):
    """Clean and convert the scraped data."""
    df = pd.DataFrame(films_data)
    df.to_csv('raw_highest_grossing_films.csv', index=False)
    if df.empty:
        return df
    
    # Clean Rank column
    df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
    
    # Clean Worldwide gross column
    df['Worldwide gross'] = df['Worldwide gross'].apply(clean_monetary_value)
    
    # Clean Year column
    df['Year'] = df['Year'].apply(clean_year)
    
    # Clean Peak column
    df['Peak'] = df['Peak'].apply(clean_monetary_value)
    
    # Remove rows with invalid data
    df = df.dropna(subset=['Rank', 'Year'])
    df = df[df['Year'] > 0]
    
    return df

def answer_questions(df):
    """Answer the specified questions about the film data."""
    answers = []
    
    # Question 1: How many $2 bn movies were released before 2020?
    two_bn_before_2020 = df[(df['Worldwide gross'] >= 2000000000.0) & (df['Year'] < 2020)]
    answer1 = str(len(two_bn_before_2020))
    answers.append(answer1)
    
    # Question 2: Which is the earliest film that grossed over $1.5 bn?
    over_1_5_bn = df[df['Worldwide gross'] > 1500000000.0]
    if not over_1_5_bn.empty:
        earliest_film = over_1_5_bn.loc[over_1_5_bn['Year'].idxmin()]
        answer2 = str(earliest_film['Title'])
    else:
        answer2 = "No film found"
    answers.append(answer2)
    
    # Question 3: What's the correlation between Rank and Peak?
    valid_data = df.dropna(subset=['Rank', 'Peak'])
    if len(valid_data) > 1:
        correlation = np.corrcoef(valid_data['Rank'], valid_data['Peak'])[0, 1]
        answer3 = str(correlation)
    else:
        answer3 = "0"
    answers.append(answer3)
    
    return answers

def create_visualization(df):
    """Create scatter plot with regression line for Rank vs Peak."""
    try:
        # Filter out invalid data
        valid_data = df.dropna(subset=['Rank', 'Peak'])
        valid_data = valid_data[(valid_data['Rank'] > 0) & (valid_data['Peak'] > 0)]
        
        if len(valid_data) < 2:
            return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Scatter plot
        plt.scatter(valid_data['Rank'], valid_data['Peak'], alpha=0.7, s=50)
        
        # Calculate and plot regression line
        x = valid_data['Rank'].values
        y = valid_data['Peak'].values
        
        # Linear regression
        coeffs = np.polyfit(x, y, 1)
        regression_line = np.polyval(coeffs, x)
        
        # Sort for proper line plotting
        sort_idx = np.argsort(x)
        plt.plot(x[sort_idx], regression_line[sort_idx], 
                linestyle='dotted', color='red', linewidth=2)
        
        plt.xlabel('Rank')
        plt.ylabel('Peak (USD)')
        plt.title('Rank vs Peak Earnings')
        plt.grid(True, alpha=0.3)
        
        # Save to memory
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=80, bbox_inches='tight')
        img_buffer.seek(0)
        
        # Encode to base64
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        plt.close()
        
        return f"data:image/png;base64,{img_base64}"
    
    except Exception as e:
        print(f"Error creating visualization: {e}")
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

def main():
    """Main function to execute the complete analysis."""
    print("Scraping film data from Wikipedia...")
    raw_data = scrape_film_data()
    
    if not raw_data:
        print("Failed to scrape data")
        return ["0", "No data", "0", "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="]
    
    print(f"Scraped {len(raw_data)} films")
    with open('raw_highest_grossing_films.json', 'w') as f:
        json.dump(raw_data, f, indent=2)
    print("Cleaning data...")
    df = clean_data(raw_data)
    print(f"After cleaning: {len(df)} valid films")
    
    print("Answering questions...")
    answers = answer_questions(df)
    
    print("Creating visualization...")
    visualization = create_visualization(df)
    
    # Combine all results
    final_results = answers + [visualization]
    
    print("Results:")
    for i, result in enumerate(final_results[:3]):
        print(f"Answer {i+1}: {result}")
    
    print(f"Visualization created: {len(visualization)} characters")
    
    return final_results

if __name__ == "__main__":
    results = main()
    print("\nFinal JSON output:")
    print(json.dumps(results, indent=2))