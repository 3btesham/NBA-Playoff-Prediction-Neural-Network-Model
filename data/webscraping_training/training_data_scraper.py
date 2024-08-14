#%% Importing libraries and main urls
from bs4 import BeautifulSoup
import requests

main_url = 'https://www.basketball-reference.com'
seasons_url = 'https://www.basketball-reference.com/leagues/'
awards_url = 'https://www.basketball-reference.com/awards/'
players_url = 'https://www.basketball-reference.com/players/'

#%% Getting all seasons pages urls
seasons_text = requests.get(seasons_url).text
seasons = BeautifulSoup(seasons_text, 'lxml')
season_paths = []

all_table_rows = seasons.find_all('tr')

for tr in all_table_rows:
    sl = tr.find('a')
    if sl:
        season_paths.append(sl['href'])
        
#%%Opening links to other files
sp = season_paths[0]
sl = main_url + sp
stxt = requests.get(sl).text
    
season = BeautifulSoup(stxt, 'lxml')
stable = season.find_all('div')
print(stable)

# %%
