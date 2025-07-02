from utils import *

def avg_df(df_list):
    combined = pd.concat(df_list)
    averaged_df = combined.groupby(combined.index).mean()
    averaged_df['noise'] = df_list[0]['noise']
    averaged_df = averaged_df[['noise', 'grad_abs_err', 'grad_rel_err', 'traps']]
    averaged_df = averaged_df.reset_index(drop=True)
    return averaged_df

p_values = [0,.1, .3, .7,]
t_list=[50]
shots_list=[1000]
angles=[np.pi/36 , np.pi/6, np.pi/2]
n_runs=5

for angle in angles:
    for shots in shots_list:
        for t in t_list:
            print(f"Angle shifts: {angle} Number of shots: {shots} Computational rounds: {4*shots}, Test rounds: {t} ")
            df_list=[]
            for _ in range(n_runs):
                obj=e2e(shots,t,angle)
                df_list.append(obj.run(p_values,show=False))
            print(avg_df(df_list))       