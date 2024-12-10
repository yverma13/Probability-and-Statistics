import pandas as pd

euro12 = pd.read_csv('https://cdn.iisc.talentsprint.com/CDS/Datasets/Euro_2012_stats_TEAM.csv', sep=',')
euro12

euro12.Goals

euro12.shape[0]

euro12.info()

discipline = euro12[['Team', 'Yellow Cards', 'Red Cards']]
discipline

discipline.sort_values(['Red Cards', 'Yellow Cards'], ascending = False)

round(discipline['Yellow Cards'].mean())

euro12[euro12.Goals > 6]

euro12[euro12.Team.str.startswith('G')]

euro12.iloc[: , 0:7]

euro12.iloc[:, :-3]

euro12.loc[euro12.Team.isin(['England', 'Italy', 'Russia']), ['Team','Shooting Accuracy']]