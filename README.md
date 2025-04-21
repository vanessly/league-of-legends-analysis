# League of Legends Analysis: Predicting Player Positions from In-Game Data Using Machine Learning
### Vinh Tran & CC Ly

## Step 1: Introduction
### Introduction and Question Identification
#### Dataset Overview

For our project, we’re using the **Oracle’s Elixir League of Legends Match Data** from the 2022 season. This dataset contains information from over 10,000 League of Legends professional matches. There are about 120,000 rows in total (each match contributes up to 12 rows: one per player plus two team‑summary rows).

#### Central Question

> **How accurately can we predict a player’s in‑game role (Top, Jungle, Mid, Bot, or Support) using only their post‑game performance statistics?**

#### Why It Matters

Automatically predicting a player’s position from raw match stats has practical value for coaches, pro-players, analysts, and broadcasters. **Coaches** can see if their players' performances line up with expected performances of other players within the same position, and make statistically-backed decisions to optimize their team roster. Simarily, **pro-players** can utilize this tool to see where they are lacking in their skills, and make adjustments to improve their gameplay. **Analysts** and **broadcasters** can utilize this data as a fun and engaging statistic and classifier for audiences. 

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

To ensure that our analysis focused only on meaningful, player-level statistics relevant to role prediction, we applied several cleaning steps to the original dataset. Each step was informed by how the data is structured and generated in professional League of Legends matches.

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
We identified and removed all columns that had missing values. Upon inspection, these columns did not contain statistics that are relevant to our modeling goal (predicting roles based on in-game performance). Keeping them would have required imputation strategies that could introduce bias.

#### Final cleaned dataframe
| gameid                | datacompleteness   | league   |   year |   playoffs | date                |   game |   patch |   participantid | side   | position   | playername   | champion   |   gamelength |   result |   kills |   deaths |   assists |   teamkills |   teamdeaths |   doublekills |   triplekills |   quadrakills |   pentakills |   firstblood |   firstbloodkill |   firstbloodassist |   firstbloodvictim |   team kpm |   ckpm |   damagetochampions |     dpm |   damageshare |   damagetakenperminute |   damagemitigatedperminute |   wardsplaced |    wpm |   wardskilled |   wcpm |   controlwardsbought |   visionscore |   vspm |   totalgold |   earnedgold |   earned gpm |   earnedgoldshare |   goldspent |   total cs |   minionkills |   monsterkills |   cspm |   goldat10 |   xpat10 |   csat10 |   opp_goldat10 |   opp_xpat10 |   opp_csat10 |   golddiffat10 |   xpdiffat10 |   csdiffat10 |   killsat10 |   assistsat10 |   deathsat10 |   opp_killsat10 |   opp_assistsat10 |   opp_deathsat10 |   goldat15 |   xpat15 |   csat15 |   opp_goldat15 |   opp_xpat15 |   opp_csat15 |   golddiffat15 |   xpdiffat15 |   csdiffat15 |   killsat15 |   assistsat15 |   deathsat15 |   opp_killsat15 |   opp_assistsat15 |   opp_deathsat15 |
|:----------------------|:-------------------|:---------|-------:|-----------:|:--------------------|-------:|--------:|----------------:|:-------|:-----------|:-------------|:-----------|-------------:|---------:|--------:|---------:|----------:|------------:|-------------:|--------------:|--------------:|--------------:|-------------:|-------------:|-----------------:|-------------------:|-------------------:|-----------:|-------:|--------------------:|--------:|--------------:|-----------------------:|---------------------------:|--------------:|-------:|--------------:|-------:|---------------------:|--------------:|-------:|------------:|-------------:|-------------:|------------------:|------------:|-----------:|--------------:|---------------:|-------:|-----------:|---------:|---------:|---------------:|-------------:|-------------:|---------------:|-------------:|-------------:|------------:|--------------:|-------------:|----------------:|------------------:|-----------------:|-----------:|---------:|---------:|---------------:|-------------:|-------------:|---------------:|-------------:|-------------:|------------:|--------------:|-------------:|----------------:|------------------:|-----------------:|
| ESPORTSTMNT01_2690210 | complete           | LCKC     |   2022 |          0 | 2022-01-10 07:44:08 |      1 |   12.01 |               1 | Blue   | top        | Soboro       | Renekton   |         1713 |        0 |       2 |        3 |         2 |           9 |           19 |             0 |             0 |             0 |            0 |            0 |                0 |                  0 |                  0 |     0.3152 | 0.9807 |               15768 | 552.294 |     0.278784  |               1072.4   |                    777.793 |             8 | 0.2802 |             6 | 0.2102 |                    5 |            26 | 0.9107 |       10934 |         7164 |      250.928 |          0.253859 |       10275 |        231 |           220 |             11 | 8.0911 |       3228 |     4909 |       89 |           3176 |         4953 |           81 |             52 |          -44 |            8 |           0 |             0 |            0 |               0 |                 0 |                0 |       5025 |     7560 |      135 |           4634 |         7215 |          121 |            391 |          345 |           14 |           0 |             1 |            0 |               0 |                 1 |                0 |
| ESPORTSTMNT01_2690210 | complete           | LCKC     |   2022 |          0 | 2022-01-10 07:44:08 |      1 |   12.01 |               2 | Blue   | jng        | Raptor       | Xin Zhao   |         1713 |        0 |       2 |        5 |         6 |           9 |           19 |             0 |             0 |             0 |            0 |            1 |                0 |                  1 |                  0 |     0.3152 | 0.9807 |               11765 | 412.084 |     0.208009  |                944.273 |                    650.158 |             6 | 0.2102 |            18 | 0.6305 |                    6 |            48 | 1.6813 |        9138 |         5368 |      188.021 |          0.19022  |        8750 |        148 |            33 |            115 | 5.1839 |       3429 |     3484 |       58 |           2944 |         3052 |           63 |            485 |          432 |           -5 |           1 |             2 |            0 |               0 |                 0 |                1 |       5366 |     5320 |       89 |           4825 |         5595 |          100 |            541 |         -275 |          -11 |           2 |             3 |            2 |               0 |                 5 |                1 |
| ESPORTSTMNT01_2690210 | complete           | LCKC     |   2022 |          0 | 2022-01-10 07:44:08 |      1 |   12.01 |               3 | Blue   | mid        | Feisty       | LeBlanc    |         1713 |        0 |       2 |        2 |         3 |           9 |           19 |             0 |             0 |             0 |            0 |            0 |                0 |                  0 |                  0 |     0.3152 | 0.9807 |               14258 | 499.405 |     0.252086  |                581.646 |                    227.776 |            19 | 0.6655 |             7 | 0.2452 |                    7 |            29 | 1.0158 |        9715 |         5945 |      208.231 |          0.210665 |        8725 |        193 |           177 |             16 | 6.7601 |       3283 |     4556 |       81 |           3121 |         4485 |           81 |            162 |           71 |            0 |           0 |             1 |            0 |               0 |                 0 |                1 |       5118 |     6942 |      120 |           5593 |         6789 |          119 |           -475 |          153 |            1 |           0 |             3 |            0 |               3 |                 3 |                2 |
| ESPORTSTMNT01_2690210 | complete           | LCKC     |   2022 |          0 | 2022-01-10 07:44:08 |      1 |   12.01 |               4 | Blue   | bot        | Gamin        | Samira     |         1713 |        0 |       2 |        4 |         2 |           9 |           19 |             0 |             0 |             0 |            0 |            1 |                0 |                  1 |                  0 |     0.3152 | 0.9807 |               11106 | 389.002 |     0.196358  |                463.853 |                    218.879 |            12 | 0.4203 |             6 | 0.2102 |                    4 |            25 | 0.8757 |       10605 |         6835 |      239.405 |          0.242201 |       10425 |        226 |           208 |             18 | 7.9159 |       3600 |     3103 |       78 |           3304 |         2838 |           90 |            296 |          265 |          -12 |           1 |             1 |            0 |               0 |                 0 |                0 |       5461 |     4591 |      115 |           6254 |         5934 |          149 |           -793 |        -1343 |          -34 |           2 |             1 |            2 |               3 |                 3 |                0 |
| ESPORTSTMNT01_2690210 | complete           | LCKC     |   2022 |          0 | 2022-01-10 07:44:08 |      1 |   12.01 |               5 | Blue   | sup        | Loopy        | Leona      |         1713 |        0 |       1 |        5 |         6 |           9 |           19 |             0 |             0 |             0 |            0 |            1 |                1 |                  0 |                  0 |     0.3152 | 0.9807 |                3663 | 128.301 |     0.0647631 |                475.026 |                    490.123 |            29 | 1.0158 |            14 | 0.4904 |                   11 |            69 | 2.4168 |        6678 |         2908 |      101.856 |          0.103054 |        6395 |         42 |            42 |              0 | 1.4711 |       2678 |     2161 |       16 |           2150 |         2748 |           15 |            528 |         -587 |            1 |           1 |             1 |            0 |               0 |                 0 |                1 |       3836 |     3588 |       28 |           3393 |         4085 |           21 |            443 |         -497 |            7 |           1 |             2 |            2 |               0 |                 6 |                2 |
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