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
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from scipy.stats import pearsonr

# 1. Scrape data from the Wikipedia page
url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find the table containing the data
table = soup.find('table', {'class': 'wikitable'})

# Extract the data from the table
headers = [th.text.strip() for th in table.find_all('th')]
rows = table.find_all('tr')[1:]  # Skip the header row
data = []
for row in rows:
    cols = row.find_all('td')
    if len(cols) >= 7:  # Ensure enough columns are present
        row_data = [col.text.strip() for col in cols]
        data.append(row_data)

df = pd.DataFrame(data, columns=headers)
df.rename(columns={'Rank': 'Rank',
                   'Title': 'Title',
                   'Worldwide gross': 'Worldwide gross',
                   'Year': 'Year',
                   'Peak': 'Peak',
                   }, inplace=True)

df.to_csv('highest_grossing_films.csv', index=False )
# Clean the 'Worldwide gross' column. Removing commas and dollar signs and converting to float.
df['Worldwide gross'] = df['Worldwide gross'].str.replace(r'[$,]', '', regex=True).astype(float)

# Convert year to numeric
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')


# 1. How many $2 bn movies were released before 2020?
movies_over_2bn_before_2020 = df[(df['Worldwide gross'] >= 2000) & (df['Year'] < 2020)]
count_2bn_before_2020 = len(movies_over_2bn_before_2020)
answer1 = [str(count_2bn_before_2020)]

# 2. Which is the earliest film that grossed over $1.5 bn?
earliest_1_5bn_film = df[df['Worldwide gross'] >= 1500].sort_values(by='Year').iloc[0]['Title']
answer2 = [earliest_1_5bn_film]

# 3. What's the correlation between the Rank and Peak?
# Corrected for missing values
df['Peak'] = pd.to_numeric(df['Peak'], errors='coerce') # Ensure peak is numeric
df_cleaned = df.dropna(subset=['Rank', 'Peak'])
correlation, _ = pearsonr(df_cleaned['Rank'].astype(float), df_cleaned['Peak'].astype(float))
answer3 = [str(correlation)]

# 4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
plt.figure(figsize=(8, 6))
plt.scatter(df_cleaned['Rank'].astype(float), df_cleaned['Peak'].astype(float))

# Regression line
z = np.polyfit(df_cleaned['Rank'].astype(float), df_cleaned['Peak'].astype(float), 1)
p = np.poly1d(z)
plt.plot(df_cleaned['Rank'].astype(float), p(df_cleaned['Rank'].astype(float)), "r--")

plt.xlabel("Rank")
plt.ylabel("Peak")
plt.title("Scatterplot of Rank and Peak with Regression Line")
plt.grid(True)

# Save the plot to a buffer
img_buffer = io.BytesIO()
plt.savefig(img_buffer, format='png')
img_buffer.seek(0)
img_data = base64.b64encode(img_buffer.read()).decode('utf-8')
image_uri = "data:image/png;base64," + img_data
plt.close()

answer4 = [image_uri]

# Construct the final output
final_answer = [answer1, answer2, answer3, answer4]
print(final_answer)