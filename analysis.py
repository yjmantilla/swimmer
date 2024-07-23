import glob
import pandas as pd
import os
datapath = 'data/local/experiments'
pattern = os.path.join(datapath, '**','video_*.mp4')
videos= glob.glob(pattern, recursive=True)

videos = [x for x in videos if 'video.mp4' not in x]



video_dicts=[]
for video in videos:
    example=video
    video_dict={}
    for part in example.split('/')[4:]:
        for pair in part.split('_'):
            if '-' in pair:
                key,val = pair.replace('.mp4','').split('-')
                if 'v' == key or 'retrain'==key:
                    val = val.replace('d','.')
                if val.isnumeric():
                    val=float(val)

                video_dict[key]=val

    if not 'retrain' in video_dict:
        video_dict['retrain']=-1
    video_dicts.append(video_dict)


df = pd.DataFrame(video_dicts)

df

# convert to numeric

df['v'] = pd.to_numeric(df['v'])
df['retrain'] = pd.to_numeric(df['retrain'])
df['score'] = pd.to_numeric(df['score'])

# from scracth (retrain=-1)

df_scratch = df[df['retrain']==-1]

# retrained

df_retrain = df[df['retrain']!=-1]

df_scratch

# viscosity vs score plots

import matplotlib.pyplot as plt

for df_,condition in zip([df_scratch,df_retrain],['Trained from scratch on target viscosity','Retrained from viscosity=0.05 to target viscosity']):
    for gdf, group in df_.groupby('model'):
        # log scale
        group = group.sort_values('v')
        plt.plot(group['v'],group['score'],label=gdf)
        plt.xlabel('viscosity')
        plt.ylabel('score')
        plt.title('Viscosity vs Score')
        plt.xscale('log')
        plt.legend()
    plt.suptitle(condition)
    fig = plt.gcf()
    fig.set_dpi(300)  # Set the resolution to 300 dpi
    fig.savefig(f'viscosity_vs_score_{condition}.png')
    plt.close('all')

## All in same plot

for df_,condition in zip([df_scratch,df_retrain],['from-scratch','retrain-from-0.05']):
    for gdf, group in df_.groupby('model'):
        # log scale
        group = group.sort_values('v')
        plt.plot(group['v'],group['score'],label=f'{gdf} {condition}')
        plt.xlabel('viscosity')
        plt.ylabel('score')
        plt.title('Viscosity vs Score')
        plt.xscale('log')
plt.legend()
plt.suptitle('All in same plot')
fig = plt.gcf()
fig.set_dpi(300)  # Set the resolution to 300 dpi
fig.savefig(f'viscosity_vs_score_all.png')
plt.close('all')