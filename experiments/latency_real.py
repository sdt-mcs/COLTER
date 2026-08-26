import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ================================
# 读取冻结后的CSV数据
# ================================

data = pd.read_csv("tail_latency_data.csv")


def get_values(workload, scheduler):

    return data[
        (data["workloads"] == workload) &
        (data["scheduler"] == scheduler)
    ]["value"].tolist()



# ================================
# 从CSV恢复原始数组
# ================================

# Kmeans

Yarn_kmeans = get_values("Kmeans", "YARN")
Toposch_n_kmeans = get_values("Kmeans", "Toposch-n")
Quasar_kmeans = get_values("Kmeans", "Quasar")
Rose_kmeans = get_values("Kmeans", "ROSE")
PID_kmeans = get_values("Kmeans", "PID-Sched")
DRL_Sched_kmeans = get_values("Kmeans", "DRL-Sched")
COLTER_kmeans = get_values("Kmeans", "COLTER")


# PageRank

Yarn_pagerank = get_values("PageRank", "YARN")
Toposch_n_pagerank = get_values("PageRank", "Toposch-n")
Quasar_pagerank = get_values("PageRank", "Quasar")
Rose_pagerank = get_values("PageRank", "ROSE")
PID_pagerank = get_values("PageRank", "PID-Sched")
DRL_Sched_pagerank = get_values("PageRank", "DRL-Sched")
COLTER_pagerank = get_values("PageRank", "COLTER")


# Terasort

Yarn_terasort = get_values("Terasort", "YARN")
Toposch_n_terasort = get_values("Terasort", "Toposch-n")
Quasar_terasort = get_values("Terasort", "Quasar")
Rose_terasort = get_values("Terasort", "ROSE")
PID_terasort = get_values("Terasort", "PID-Sched")
DRL_Sched_terasort = get_values("Terasort", "DRL-Sched")
COLTER_terasort = get_values("Terasort", "COLTER")



# ================================
# 原始绘图代码
# ================================

standardsize = 35

fig = plt.figure(figsize=(30,20))


ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)


standfont = {
    'family':'Arial',
    'size':standardsize-4,
}


n_groups = 8

index = np.arange(n_groups)

bar_width = 0.12

opacity = 0.5



def str2Number(strParam):

    if strParam == '' or strParam is None:
        res = 0

    try:
        res = float(strParam)

    except:
        res = 0

    return res



def autolabel(rects, ax):

    for rect in rects:

        height = rect.get_height()

        if str2Number(height)<=1.0:

            height='1.01'


        ax.text(
            rect.get_x()+rect.get_width()/2-0.05,
            1.01*str2Number(height),
            '%s'%float(height)+'x',
            size=22,
            rotation=90
        )



# ================================
# Kmeans
# ================================


R_yarn1=ax1.bar(
    index,
    Yarn_kmeans,
    bar_width,
    alpha=opacity,
    fill='false',
    edgecolor='black',
    color='#4c40f5',
    label='YARN'
)


R_toposch1=ax1.bar(
    index+bar_width,
    Toposch_n_kmeans,
    bar_width,
    alpha=opacity,
    fill='false',
    edgecolor='black',
    color='#2E8B57',
    label='Toposch-n'
)


R_quasar1=ax1.bar(
    index+bar_width*2,
    Quasar_kmeans,
    bar_width,
    alpha=opacity,
    fill='false',
    edgecolor='black',
    color='#87CEFA',
    label='Quasar'
)


R_rose1=ax1.bar(
    index+bar_width*3,
    Rose_kmeans,
    bar_width,
    alpha=opacity,
    fill='false',
    edgecolor='black',
    color='#FF8C00',
    label='ROSE'
)


R_pid1=ax1.bar(
    index+bar_width*4,
    PID_kmeans,
    bar_width,
    alpha=opacity,
    fill='false',
    edgecolor='black',
    color='#E74C3C',
    label='PID-Sched'
)


R_drl1=ax1.bar(
    index+bar_width*5,
    DRL_Sched_kmeans,
    bar_width,
    alpha=opacity,
    fill='false',
    edgecolor='black',
    color='#A9A9A9',
    label='DRL-Sched'
)


R_colter1=ax1.bar(
    index+bar_width*6,
    COLTER_kmeans,
    bar_width,
    alpha=opacity,
    fill='false',
    edgecolor='black',
    color='#f7fa9d',
    label='COLTER'
)



for r in [
    R_yarn1,
    R_toposch1,
    R_quasar1,
    R_rose1,
    R_pid1,
    R_drl1,
    R_colter1
]:

    autolabel(r,ax1)



ax1.set_ylim(1,2.5)

ax1.set_xticks(index+bar_width*3)

ax1.set_xticklabels(
    (
        'homepage',
        'inbox',
        'inboxlist',
        'login',
        'loginpage',
        'logout',
        'demand',
        'demandlist'
    ),
    fontname='Arial'
)


ax1.grid(
    which='major',
    axis='y',
    linestyle='--',
    linewidth=1
)


ax1.tick_params(
    labelsize=standardsize,
    color='black'
)


ax1.set_xlabel(
    '(a) Co-scheduling Kmeans with LRA',
    fontname='Arial',
    fontsize=standardsize
)



# ================================
# PageRank
# ================================


R_yarn2=ax2.bar(index,Yarn_pagerank,bar_width,alpha=opacity,fill='false',edgecolor='black',color='#4c40f5')
R_toposch2=ax2.bar(index+bar_width,Toposch_n_pagerank,bar_width,alpha=opacity,fill='false',edgecolor='black',color='#2E8B57')
R_quasar2=ax2.bar(index+bar_width*2,Quasar_pagerank,bar_width,alpha=opacity,fill='false',edgecolor='black',color='#87CEFA')
R_rose2=ax2.bar(index+bar_width*3,Rose_pagerank,bar_width,alpha=opacity,fill='false',edgecolor='black',color='#FF8C00')
R_pid2=ax2.bar(index+bar_width*4,PID_pagerank,bar_width,alpha=opacity,fill='false',edgecolor='black',color='#E74C3C')
R_drl2=ax2.bar(index+bar_width*5,DRL_Sched_pagerank,bar_width,alpha=opacity,fill='false',edgecolor='black',color='#A9A9A9')
R_colter2=ax2.bar(index+bar_width*6,COLTER_pagerank,bar_width,alpha=opacity,fill='false',edgecolor='black',color='#f7fa9d')


for r in [
    R_yarn2,
    R_toposch2,
    R_quasar2,
    R_rose2,
    R_pid2,
    R_drl2,
    R_colter2
]:

    autolabel(r,ax2)



ax2.set_ylim(1,2.5)

ax2.set_xticks(index+bar_width*3)

ax2.set_xticklabels(
    (
        'homepage',
        'inbox',
        'inboxlist',
        'login',
        'loginpage',
        'logout',
        'demand',
        'demandlist'
    ),
    fontname='Arial'
)


ax2.grid(which='major',axis='y',linestyle='--',linewidth=1)

ax2.tick_params(labelsize=standardsize,color='black')

ax2.set_xlabel(
    '(b) Co-scheduling PageRank with LRA',
    fontname='Arial',
    fontsize=standardsize
)



# ================================
# Terasort
# ================================


R_yarn3=ax3.bar(index,Yarn_terasort,bar_width,alpha=opacity,fill='false',edgecolor='black',color='#4c40f5')
R_toposch3=ax3.bar(index+bar_width,Toposch_n_terasort,bar_width,alpha=opacity,fill='false',edgecolor='black',color='#2E8B57')
R_quasar3=ax3.bar(index+bar_width*2,Quasar_terasort,bar_width,alpha=opacity,fill='false',edgecolor='black',color='#87CEFA')
R_rose3=ax3.bar(index+bar_width*3,Rose_terasort,bar_width,alpha=opacity,fill='false',edgecolor='black',color='#FF8C00')
R_pid3=ax3.bar(index+bar_width*4,PID_terasort,bar_width,alpha=opacity,fill='false',edgecolor='black',color='#E74C3C')
R_drl3=ax3.bar(index+bar_width*5,DRL_Sched_terasort,bar_width,alpha=opacity,fill='false',edgecolor='black',color='#A9A9A9')
R_colter3=ax3.bar(index+bar_width*6,COLTER_terasort,bar_width,alpha=opacity,fill='false',edgecolor='black',color='#f7fa9d')


for r in [
    R_yarn3,
    R_toposch3,
    R_quasar3,
    R_rose3,
    R_pid3,
    R_drl3,
    R_colter3
]:

    autolabel(r,ax3)



ax3.set_ylim(1,2.0)

ax3.set_xticks(index+bar_width*3)


ax3.set_xticklabels(
    (
        'homepage',
        'inbox',
        'inboxlist',
        'login',
        'loginpage',
        'logout',
        'demand',
        'demandlist'
    ),
    fontname='Arial'
)


ax3.grid(which='major',axis='y',linestyle='--',linewidth=1)

ax3.tick_params(labelsize=standardsize,color='black')


ax3.set_xlabel(
    '(c) Co-scheduling Terasort with LRA',
    fontname='Arial',
    fontsize=standardsize
)



# y轴标签

ax2.set_ylabel(
    'Tail latency increase against run-alone',
    fontname='Arial',
    fontsize=standardsize,
    labelpad=25
)



# legend

handles=[
    R_yarn1,
    R_toposch1,
    R_quasar1,
    R_rose1,
    R_pid1,
    R_drl1,
    R_colter1
]


labels=[
    'YARN',
    'Toposch-n',
    'Quasar',
    'ROSE',
    'PID-Sched',
    'DRL-Sched',
    'COLTER'
]


fig.legend(
    handles,
    labels,
    loc='upper center',
    ncol=7,
    prop=standfont,
    bbox_to_anchor=(0.5,1.01),
    frameon=True,
    edgecolor='gray'
)



plt.subplots_adjust(
    hspace=0.34,
    top=0.93
)


plt.savefig(
    "enhanced_tail_latency_comparison.pdf",
    bbox_inches='tight'
)


plt.show()