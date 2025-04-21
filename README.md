# League of Legends Analysis: Predicting Player Positions from In-Game Data Using Machine Learning
### Vinh Tran & CC Ly

## Step 1: Introduction
### Introduction and Question Identification
#### Dataset Overview

- For our project, we’re using the **Oracle’s Elixir League of Legends Match Data** from the 2022 season. This dataset contains information from over 10,000 League of Legends professional matches. There are about 120,000 rows in total (each match contributes up to 12 rows: one per player plus two team‑summary rows).

#### Central Question

> **How accurately can we predict a player’s in‑game role (Top, Jungle, Mid, Bot, or Support) using only their post‑game performance statistics?**

#### Why It Matters

- Automatically predicting a player’s position from raw match stats has practical value for coaches, pro-players, analysts, and broadcasters. **Coaches** can see if their players' performances line up with expected performances of other players within the same position, and make statistically-backed decisions to optimize their team roster. Simarily, **pro-players** can utilize this tool to see where they are lacking in their skills, and make adjustments to improve their gameplay. **Analysts** and **broadcasters** can utilize this data as a fun and engaging statistic and classifier for audiences. 

#### Key Columns

Below are the columns relevant to our question:

| Column        | Description                                                       |
|---------------|-------------------------------------------------------------------|
| `gameid`    | Unique ID for each match (ties together all player and team rows) |
| `position`  | The role a player filled in that game (Top, Jungle, Mid, Bot, Support) |
| `kills`     | Number of enemy champions the player eliminated                   |
| `assists`   | Number of enemy champion kills the player helped secure           |
| `deaths`   | Number of times the player was eliminated by enemy champions      |
| `dpm`       | Damage per minute: average damage dealt to champions per minute   |
| `earned gpm`| Gold per minute earned by the player throughout the match         |
| `cspm`      | Creep score per minute: average minions and monsters killed per minute |
| `monsterkills` | Total number of neutral monsters killed by the player         |
## Step 2: Data Cleaning and Exploratory Data Analysis
### Data Cleaning

- To ensure that our analysis focused only on meaningful, player-level statistics relevant to role prediction, we applied several cleaning steps to the original dataset based on how the original data is structured and generated in the dateset.

#### 1. Filtered only for complete player data
```python
df = df[df['datacompleteness'] == 'complete']
```
- The dataset includes some rows marked as `incomplete`, which may result from matches where data logging failed or games were not played to completion. We filtered the DataFrame to keep only rows where `datacompleteness` was marked as `complete`, ensuring all included rows contain full, reliable statistics.

#### 2. Removed team-related summary rows
```python
df = df.groupby('gameid', group_keys=False).apply(lambda x: x.iloc[:-2])
```
- For each `gameid`, the dataset contains 12 rows: 10 for individual players and 2 for team-level summary statistics. Since our prediction task focuses on individual player performance, we removed the last two rows of each match group, which correspond to team summaries. We verified that this operation worked correctly by checking that only 10 players remained in a sample game:
```python
print(df.loc[df['gameid'] == 'ESPORTSTMNT01_2690210', 'playername'])
```

#### 3. Dropped irrelevant columns
```python
cols_to_drop = ['url', 'split', 'pick1', ..., 'monsterkillsenemyjungle']
df.drop(columns=cols_to_drop, inplace=True)
```
- We removed columns that are either:
    - Unrelated to performance metrics (e.g. url, split)
    - Draft data (e.g. pick1 to pick5)
    - Team-level objective data (e.g. firstdragon, elders, heralds, etc.)

#### 4. Dropped columns with Null values
```python
columns_with_null = df.isnull().sum()[df.isnull().sum() > 0].index.to_list()
df.drop(columns=columns_with_null, inplace=True)
```
```python
['playerid', 'teamname', 'teamid', 'ban1', 'ban2', 'ban3', 'ban4', 'ban5', 'barons', 'opp_barons', 'inhibitors', 'opp_inhibitors', 'goldat20', 'xpat20', 'csat20', 'opp_goldat20', 'opp_xpat20', 'opp_csat20', 'golddiffat20', 'xpdiffat20', 'csdiffat20', 'killsat20', 'assistsat20', 'deathsat20', 'opp_killsat20', 'opp_assistsat20', 'opp_deathsat20', 'goldat25', 'xpat25', 'csat25', 'opp_goldat25', 'opp_xpat25', 'opp_csat25', 'golddiffat25', 'xpdiffat25', 'csdiffat25', 'killsat25', 'assistsat25', 'deathsat25', 'opp_killsat25', 'opp_assistsat25', 'opp_deathsat25']
```
- We identified and removed all columns that had missing values. Upon inspection, these columns did not contain statistics that are relevant to our modeling goal (predicting roles based on in-game performance). Similar to above, these columns were either redundant information or team-level data. Keeping them would have required imputation strategies that could introduce bias. 

#### Final cleaned dataframe
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gameid</th>
      <th>datacompleteness</th>
      <th>league</th>
      <th>year</th>
      <th>playoffs</th>
      <th>date</th>
      <th>game</th>
      <th>patch</th>
      <th>participantid</th>
      <th>side</th>
      <th>position</th>
      <th>playername</th>
      <th>champion</th>
      <th>gamelength</th>
      <th>result</th>
      <th>kills</th>
      <th>deaths</th>
      <th>assists</th>
      <th>teamkills</th>
      <th>teamdeaths</th>
      <th>doublekills</th>
      <th>triplekills</th>
      <th>quadrakills</th>
      <th>pentakills</th>
      <th>firstblood</th>
      <th>firstbloodkill</th>
      <th>firstbloodassist</th>
      <th>firstbloodvictim</th>
      <th>team kpm</th>
      <th>ckpm</th>
      <th>damagetochampions</th>
      <th>dpm</th>
      <th>damageshare</th>
      <th>damagetakenperminute</th>
      <th>damagemitigatedperminute</th>
      <th>wardsplaced</th>
      <th>wpm</th>
      <th>wardskilled</th>
      <th>wcpm</th>
      <th>controlwardsbought</th>
      <th>visionscore</th>
      <th>vspm</th>
      <th>totalgold</th>
      <th>earnedgold</th>
      <th>earned gpm</th>
      <th>earnedgoldshare</th>
      <th>goldspent</th>
      <th>total cs</th>
      <th>minionkills</th>
      <th>monsterkills</th>
      <th>cspm</th>
      <th>goldat10</th>
      <th>xpat10</th>
      <th>csat10</th>
      <th>opp_goldat10</th>
      <th>opp_xpat10</th>
      <th>opp_csat10</th>
      <th>golddiffat10</th>
      <th>xpdiffat10</th>
      <th>csdiffat10</th>
      <th>killsat10</th>
      <th>assistsat10</th>
      <th>deathsat10</th>
      <th>opp_killsat10</th>
      <th>opp_assistsat10</th>
      <th>opp_deathsat10</th>
      <th>goldat15</th>
      <th>xpat15</th>
      <th>csat15</th>
      <th>opp_goldat15</th>
      <th>opp_xpat15</th>
      <th>opp_csat15</th>
      <th>golddiffat15</th>
      <th>xpdiffat15</th>
      <th>csdiffat15</th>
      <th>killsat15</th>
      <th>assistsat15</th>
      <th>deathsat15</th>
      <th>opp_killsat15</th>
      <th>opp_assistsat15</th>
      <th>opp_deathsat15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ESPORTSTMNT01_2690210</td>
      <td>complete</td>
      <td>LCKC</td>
      <td>2022</td>
      <td>0</td>
      <td>2022-01-10 07:44:08</td>
      <td>1</td>
      <td>12.01</td>
      <td>1</td>
      <td>Blue</td>
      <td>top</td>
      <td>Soboro</td>
      <td>Renekton</td>
      <td>1713</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>9</td>
      <td>19</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.32</td>
      <td>0.98</td>
      <td>15768.0</td>
      <td>552.29</td>
      <td>0.28</td>
      <td>1072.40</td>
      <td>777.79</td>
      <td>8.0</td>
      <td>0.28</td>
      <td>6.0</td>
      <td>0.21</td>
      <td>5.0</td>
      <td>26.0</td>
      <td>0.91</td>
      <td>10934</td>
      <td>7164.0</td>
      <td>250.93</td>
      <td>0.25</td>
      <td>10275.0</td>
      <td>231.0</td>
      <td>220.0</td>
      <td>11.0</td>
      <td>8.09</td>
      <td>3228.0</td>
      <td>4909.0</td>
      <td>89.0</td>
      <td>3176.0</td>
      <td>4953.0</td>
      <td>81.0</td>
      <td>52.0</td>
      <td>-44.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5025.0</td>
      <td>7560.0</td>
      <td>135.0</td>
      <td>4634.0</td>
      <td>7215.0</td>
      <td>121.0</td>
      <td>391.0</td>
      <td>345.0</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ESPORTSTMNT01_2690210</td>
      <td>complete</td>
      <td>LCKC</td>
      <td>2022</td>
      <td>0</td>
      <td>2022-01-10 07:44:08</td>
      <td>1</td>
      <td>12.01</td>
      <td>2</td>
      <td>Blue</td>
      <td>jng</td>
      <td>Raptor</td>
      <td>Xin Zhao</td>
      <td>1713</td>
      <td>0</td>
      <td>2</td>
      <td>5</td>
      <td>6</td>
      <td>9</td>
      <td>19</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.32</td>
      <td>0.98</td>
      <td>11765.0</td>
      <td>412.08</td>
      <td>0.21</td>
      <td>944.27</td>
      <td>650.16</td>
      <td>6.0</td>
      <td>0.21</td>
      <td>18.0</td>
      <td>0.63</td>
      <td>6.0</td>
      <td>48.0</td>
      <td>1.68</td>
      <td>9138</td>
      <td>5368.0</td>
      <td>188.02</td>
      <td>0.19</td>
      <td>8750.0</td>
      <td>148.0</td>
      <td>33.0</td>
      <td>115.0</td>
      <td>5.18</td>
      <td>3429.0</td>
      <td>3484.0</td>
      <td>58.0</td>
      <td>2944.0</td>
      <td>3052.0</td>
      <td>63.0</td>
      <td>485.0</td>
      <td>432.0</td>
      <td>-5.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>5366.0</td>
      <td>5320.0</td>
      <td>89.0</td>
      <td>4825.0</td>
      <td>5595.0</td>
      <td>100.0</td>
      <td>541.0</td>
      <td>-275.0</td>
      <td>-11.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ESPORTSTMNT01_2690210</td>
      <td>complete</td>
      <td>LCKC</td>
      <td>2022</td>
      <td>0</td>
      <td>2022-01-10 07:44:08</td>
      <td>1</td>
      <td>12.01</td>
      <td>3</td>
      <td>Blue</td>
      <td>mid</td>
      <td>Feisty</td>
      <td>LeBlanc</td>
      <td>1713</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>9</td>
      <td>19</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.32</td>
      <td>0.98</td>
      <td>14258.0</td>
      <td>499.40</td>
      <td>0.25</td>
      <td>581.65</td>
      <td>227.78</td>
      <td>19.0</td>
      <td>0.67</td>
      <td>7.0</td>
      <td>0.25</td>
      <td>7.0</td>
      <td>29.0</td>
      <td>1.02</td>
      <td>9715</td>
      <td>5945.0</td>
      <td>208.23</td>
      <td>0.21</td>
      <td>8725.0</td>
      <td>193.0</td>
      <td>177.0</td>
      <td>16.0</td>
      <td>6.76</td>
      <td>3283.0</td>
      <td>4556.0</td>
      <td>81.0</td>
      <td>3121.0</td>
      <td>4485.0</td>
      <td>81.0</td>
      <td>162.0</td>
      <td>71.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>5118.0</td>
      <td>6942.0</td>
      <td>120.0</td>
      <td>5593.0</td>
      <td>6789.0</td>
      <td>119.0</td>
      <td>-475.0</td>
      <td>153.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ESPORTSTMNT01_2690210</td>
      <td>complete</td>
      <td>LCKC</td>
      <td>2022</td>
      <td>0</td>
      <td>2022-01-10 07:44:08</td>
      <td>1</td>
      <td>12.01</td>
      <td>4</td>
      <td>Blue</td>
      <td>bot</td>
      <td>Gamin</td>
      <td>Samira</td>
      <td>1713</td>
      <td>0</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>9</td>
      <td>19</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.32</td>
      <td>0.98</td>
      <td>11106.0</td>
      <td>389.00</td>
      <td>0.20</td>
      <td>463.85</td>
      <td>218.88</td>
      <td>12.0</td>
      <td>0.42</td>
      <td>6.0</td>
      <td>0.21</td>
      <td>4.0</td>
      <td>25.0</td>
      <td>0.88</td>
      <td>10605</td>
      <td>6835.0</td>
      <td>239.40</td>
      <td>0.24</td>
      <td>10425.0</td>
      <td>226.0</td>
      <td>208.0</td>
      <td>18.0</td>
      <td>7.92</td>
      <td>3600.0</td>
      <td>3103.0</td>
      <td>78.0</td>
      <td>3304.0</td>
      <td>2838.0</td>
      <td>90.0</td>
      <td>296.0</td>
      <td>265.0</td>
      <td>-12.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5461.0</td>
      <td>4591.0</td>
      <td>115.0</td>
      <td>6254.0</td>
      <td>5934.0</td>
      <td>149.0</td>
      <td>-793.0</td>
      <td>-1343.0</td>
      <td>-34.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ESPORTSTMNT01_2690210</td>
      <td>complete</td>
      <td>LCKC</td>
      <td>2022</td>
      <td>0</td>
      <td>2022-01-10 07:44:08</td>
      <td>1</td>
      <td>12.01</td>
      <td>5</td>
      <td>Blue</td>
      <td>sup</td>
      <td>Loopy</td>
      <td>Leona</td>
      <td>1713</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>6</td>
      <td>9</td>
      <td>19</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.32</td>
      <td>0.98</td>
      <td>3663.0</td>
      <td>128.30</td>
      <td>0.06</td>
      <td>475.03</td>
      <td>490.12</td>
      <td>29.0</td>
      <td>1.02</td>
      <td>14.0</td>
      <td>0.49</td>
      <td>11.0</td>
      <td>69.0</td>
      <td>2.42</td>
      <td>6678</td>
      <td>2908.0</td>
      <td>101.86</td>
      <td>0.10</td>
      <td>6395.0</td>
      <td>42.0</td>
      <td>42.0</td>
      <td>0.0</td>
      <td>1.47</td>
      <td>2678.0</td>
      <td>2161.0</td>
      <td>16.0</td>
      <td>2150.0</td>
      <td>2748.0</td>
      <td>15.0</td>
      <td>528.0</td>
      <td>-587.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3836.0</td>
      <td>3588.0</td>
      <td>28.0</td>
      <td>3393.0</td>
      <td>4085.0</td>
      <td>21.0</td>
      <td>443.0</td>
      <td>-497.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>


### Univariate Analysis
<iframe
 src="assets/univ-kills-dist.html"
 width="800"
 height="600"
 frameborder="0"
 ></iframe>

 - This histogram shows the distribution of total kills per game across the dataset, and the right-skewed shape indicates that while most games have between 20 and 40 total kills, there are occasional high-kill matches. This shows that depending on the game whether that things such as game pace, aggression of players, competetiveness of players, etc,  could influence a player’s role and performance statistics, which is relevant for our model since understanding the overall distribution of kills per game helps contextualize which roles are likely to stand out based on their kill-related statistics.

### Bivariate Analysis

<iframe
 src="assets/biv-avg-kills.html"
 width="800"
 height="600"
 frameborder="0"
 ></iframe>

- This bar chart displays the average number of kills per game for each player position. We observe that Bot and Mid positions have the highest kill averages, with Sup having the lowest, supporting our idea that in-game statistics like kills can help differentiate between player roles, thus directly addressing our model’s goal of predicting position from performance metrics.


<iframe
 src="assets/biv-ka.html"
 width="800"
 height="600"
 frameborder="0"
 ></iframe>

 - This bar chart compares the average number of kills and assists per game for each player position. This shows us that **supp** players have the highest average assists and the lowest kills, and **jungle** players have more assists on average than **top**, **jungle**, **mid**. However, this also shows that additional features may be needed to accurately distinguish between the latter positions in our role prediction model.


### Interesting Aggregates
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>position</th>
      <th>bot</th>
      <th>jng</th>
      <th>mid</th>
      <th>sup</th>
      <th>top</th>
    </tr>
    <tr>
      <th>gameid</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ESPORTSTMNT01_2690210</th>
      <td>5.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>0.5</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>ESPORTSTMNT01_2690219</th>
      <td>1.5</td>
      <td>3.5</td>
      <td>3.5</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>ESPORTSTMNT01_2690227</th>
      <td>1.5</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>ESPORTSTMNT01_2690255</th>
      <td>4.5</td>
      <td>2.5</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>ESPORTSTMNT01_2690264</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>ESPORTSTMNT01_2690302</th>
      <td>5.0</td>
      <td>3.5</td>
      <td>7.0</td>
      <td>0.5</td>
      <td>5.5</td>
    </tr>
    <tr>
      <th>ESPORTSTMNT01_2690328</th>
      <td>6.0</td>
      <td>3.5</td>
      <td>7.5</td>
      <td>0.5</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>ESPORTSTMNT01_2690351</th>
      <td>1.5</td>
      <td>0.5</td>
      <td>4.0</td>
      <td>0.5</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>ESPORTSTMNT01_2690370</th>
      <td>4.5</td>
      <td>2.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>ESPORTSTMNT01_2690390</th>
      <td>2.5</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>2.0</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>

- This pivot table summarizes the number of kills per game by **player position**. Each row corresponds to a unique `gameid`, and each column represents the **total number of kills** made by players in one of the five standard League of Legends roles: **bot**, **jng** (jungle), **mid**, **sup** (support), and **top**.
- This pivot table is significant because it transforms the raw match-level data into a structured format that allows us to visualize and thus compare the kill contributions of each role across each game. By analyzing these values, we can further observe trends that tell us which roles are contributing more or less kills on average.

### Imputation
```python
columns_with_null = df.isnull().sum()[df.isnull().sum() > 0].index.to_list()
df.drop(columns=columns_with_null, inplace=True)
```
```python
['playerid', 'teamname', 'teamid', 'ban1', 'ban2', 'ban3', 'ban4', 'ban5', 'barons', 'opp_barons', 'inhibitors', 'opp_inhibitors', 'goldat20', 'xpat20', 'csat20', 'opp_goldat20', 'opp_xpat20', 'opp_csat20', 'golddiffat20', 'xpdiffat20', 'csdiffat20', 'killsat20', 'assistsat20', 'deathsat20', 'opp_killsat20', 'opp_assistsat20', 'opp_deathsat20', 'goldat25', 'xpat25', 'csat25', 'opp_goldat25', 'opp_xpat25', 'opp_csat25', 'golddiffat25', 'xpdiffat25', 'csdiffat25', 'killsat25', 'assistsat25', 'deathsat25', 'opp_killsat25', 'opp_assistsat25', 'opp_deathsat25']
```
- We did not impute any missing values. Instead, as described in Step 2, we decided to just remove any columns with missing values altogether because none of the columns with missing values were relevant to our goal of predicting roles based on in-game statistics. All of these columns were either redundant or team information.
    - We were able to make this deduction because of our prior knowledge of the game. 

## Step 3: Framing a Prediction Problem
### Problem Identification
- Our prediction problem is: **"How can we predict what role a player is playing (Top, Jungle, Mid, Bottom, or Support) based on their in-game statistics?"** This is a **multiclass classification** problem, as we are predicting multipled possible categorical roles (one out of five roles)

### Response Variable
- The **response variable** is the `position` column, which identifies the role each player fulfilled during a match: `top`, `jng`, `mid`, `bot`, or `sup`. We chose this variable because our goal is to infer a player’s role solely from their post-game performance statistics—such as kills, assists, gold earned per minute, and damage dealt per minute—rather than using manually labeled or externally sourced data.

### Evaluation Metric
- We chose **accuracy** as our primary evaluation metric. Since the five roles are fairly balanced in the dataset and carry equal importance, accuracy is an intuitive and straightforward way to measure how often our model correctly predicts a player’s role. If the class distribution were more imbalanced or if misclassifying certain roles carried different costs (e.g., support vs. mid), we might consider metrics like **F1-score** or **weighted precision/recall** instead.

### Information Available at Time of Prediction
Our model is designed to use only **post-game player statistics** (e.g., kills, deaths, assists, gold earned, damage per minute) that are known at the time the game concludes. We **exclude draft picks, team-level objectives, or opponent statistics**, as these would not be reliable or player-specific indicators for individual performance patterns. This ensures that the features used are consistent with what would be available if we were trying to infer a player's role after the game, without relying on predefined labels or external metadata.








## Step 4: Baseline Model
### Baseline Model

## Step 5: Final Model
### Final Model