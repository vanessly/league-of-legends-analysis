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
| **gameid**    | Unique ID for each match (ties together all player and team rows) |
| **position**  | The role a player filled in that game (Top, Jungle, Mid, Bot, Support) |
| **kills**     | Number of enemy champions the player eliminated                   |
| **assists**   | Number of enemy champion kills the player helped secure           |
| **deaths**    | Number of times the player was eliminated by enemy champions      |
| **dpm**       | Damage per minute: average damage dealt to champions per minute   |
| **earned gpm**| Gold per minute earned by the player throughout the match         |
| **cspm**      | Creep score per minute: average minions and monsters killed per minute |
| **monsterkills** | Total number of neutral monsters killed by the player         |
## Step 2: Data Cleaning and Exploratory Data Analysis
### Data Cleaning

- To ensure that our analysis focused only on meaningful, player-level statistics relevant to role prediction, we applied several cleaning steps to the original dataset. Each step was informed by how the data is structured and generated in professional League of Legends matches.

#### 1. Filtered only for complete player data
```python
df = df[df['datacompleteness'] == 'complete']
```
- The dataset includes some rows marked as incomplete, which may result from matches where data logging failed or games were not played to completion. We filtered the DataFrame to keep only rows where 'datacompleteness' was marked as "complete", ensuring all included rows contain full, reliable statistics.

#### 2. Removed team-related summary rows
```python
df = df.groupby('gameid', group_keys=False).apply(lambda x: x.iloc[:-2])
```
- For each gameid, the dataset contains 12 rows: 10 for individual players and 2 for team-level summary statistics. Since our prediction task focuses on individual player performance, we removed the last two rows of each match group, which correspond to team summaries. We verified that this operation worked correctly by checking that only 10 players remained in a sample game:
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
- We identified and removed all columns that had missing values. Upon inspection, these columns did not contain statistics that are relevant to our modeling goal (predicting roles based on in-game performance). Keeping them would have required imputation strategies that could introduce bias.

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
 


### Bivariate Analysis
### Interesting Aggregates
### Imputation

## Step 3: Framing a Prediction Problem
### Problem Identification

## Step 4: Baseline Model
### Baseline Model

## Step 5: Final Model
### Final Model