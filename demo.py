import matplotlib.pyplot as plt
import numpy as np

# Random test data
np.random.seed(123)
all_data = [np.random.normal(0, std, 100) for std in range(1, 4)]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

# rectangular box plot
bplot1 = axes[0].boxplot(all_data,
                         vert=True,   # vertical box aligmnent
                         patch_artist=True)   # fill with color

# notch shape box plot
bplot2 = axes[1].boxplot(all_data,
                         notch=True,  # notch shape
                         vert=True,   # vertical box aligmnent
                         patch_artist=True)   # fill with color

# fill with colors
colors = ['pink', 'lightblue', 'lightgreen']
for bplot in (bplot1, bplot2):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

# adding horizontal grid lines
for ax in axes:
    ax.yaxis.grid(True)
    ax.set_xticks([y+1 for y in range(len(all_data))], )
    ax.set_xlabel('xlabel')
    ax.set_ylabel('ylabel')

# add x-tick labels
plt.setp(axes, xticks=[y+1 for y in range(len(all_data))],
         xticklabels=['x1', 'x2', 'x3', 'x4'])

plt.show()

'''
""gid"  =  1060120  OR "gid"  =  1070121  OR "gid"  =  1050121  OR "gid"  =  1040120  OR "gid"  =  1090121  OR  "gid"  =  1080121 OR  "gid"  =  1050119 OR  "gid"  =  1060118 OR  "gid"  =  1080118 OR  "gid"  =  1090119 OR  "gid"  = 1080120 OR  "gid"  =1090121 OR  "gid"  = 1030119 OR  "gid"  = 1060122   OR  "gid" =  1070119
 OR "gid" =  950113  OR "gid" =  940114  OR "gid" =  960114  OR "gid" =  960112  OR "gid" =  940112  OR "gid" =  950111  OR "gid" =  970111  OR "gid" =  960110  OR "gid" =  940111  OR "gid" = 950109  OR "gid" =  980110  OR  "gid" = 940110
  OR "gid" =  900106   OR "gid" =  910107   OR "gid" = 890107   OR "gid" =  920106   OR "gid" = 910105   OR "gid" = 890105 


"gid"  =  1060120  OR "gid"  =  1070121  OR "gid"  =  1050121  OR "gid"  =  1040120  OR "gid"  =  1090121  OR  "gid"  =  1080121 OR  "gid"  =  1050119 OR  "gid"  =  1060118 OR  "gid"  =  1080118 OR  "gid"  =  1090119 OR  "gid"  = 1080120 OR  "gid"  =1090121 OR  "gid"  = 1030119 OR  "gid"  = 1060122   OR  "gid" =  1070119
 OR "gid" =  950113  OR "gid" =  940114  OR "gid" =  960114  OR "gid" =  960112  OR "gid" =  940112  OR "gid" =  950111  OR "gid" =  970111  OR "gid" =  960110  OR "gid" =  940111  OR "gid" = 950109  OR "gid" =  980110  OR  "gid" = 940110
  OR "gid" =  900106   OR "gid" =  910107   OR "gid" = 890107   OR "gid" =  920106   OR "gid" = 910105   OR "gid" = 890105 
   OR "gid" =  1140112 OR "gid" =  1130111 OR "gid" =   1120112 OR "gid" =  1110111 OR "gid" =  1100110 OR "gid" =  1080110 OR "gid" =  1070109 OR "gid" =  1050109 OR "gid" =   1030109 OR "gid" =  1020108 OR "gid" =  1010107 OR "gid" = 1020106 OR "gid" = 1000106 OR "gid" =  980106
    OR "gid" =  1170129 OR "gid" =  1160130 OR "gid" =  1150129 OR "gid" =  1160128 OR "gid" =  1180128 OR "gid" = 1180130 OR "gid" =  1170131  OR "gid" =  1150131
    '''



"gid"  =  1060120  OR "gid"  =  1070121  OR "gid"  =  1050121  OR "gid"  =  1040120  OR "gid"  =  1090121  OR  "gid"  =  1080121 OR  "gid"  =  1050119 OR  "gid"  =  1060118 OR  "gid"  =  1080118 OR  "gid"  =  1090119 OR  "gid"  = 1080120 OR  "gid"  =1090121 OR  "gid"  = 1030119 OR  "gid"  = 1060122   OR  "gid" =  1070119
 OR "gid" =  950113  OR "gid" =  940114  OR "gid" =  960114  OR "gid" =  960112  OR "gid" =  940112  OR "gid" =  950111  OR "gid" =  970111  OR "gid" =  960110  OR "gid" =  940111  OR "gid" = 950109  OR "gid" =  980110  OR  "gid" = 940110
  OR "gid" =  900106   OR "gid" =  910107   OR "gid" = 890107   OR "gid" =  920106   OR "gid" = 910105   OR "gid" = 890105 
   OR "gid" =  1140112 OR "gid" =  1130111 OR "gid" =   1120112 OR "gid" =  1110111 OR "gid" =  1100110 OR "gid" =  1080110 OR "gid" =  1070109 OR "gid" =  1050109 OR "gid" =   1030109 OR "gid" =  1020108 OR "gid" =  1010107 OR "gid" = 1020106 OR "gid" = 1000106 OR "gid" =  980106
    OR "gid" =  1170129 OR "gid" =  1160130 OR "gid" =  1150129 OR "gid" =  1160128 OR "gid" =  1180128 OR "gid" = 1180130 OR "gid" =  1170131  OR "gid" =  1150131 
    OR "gid" = 1080122 OR "gid" = 1070123 OR "gid" = 1090123 OR "gid" = 1080124
    OR "gid" = 1110119 OR "gid" = 1120118
    OR "gid" = 980114 
    OR "gid" = 980112 OR "gid" = 990111 
    OR "gid" = 950107 OR "gid" = 940108 
    OR "gid" = 920110 OR "gid" = 910109 
    OR "gid" = 1010109 
    OR "gid" = 900108 
    OR "gid" = 880104 OR "gid" = 870105 
    OR "gid" = 870107 OR "gid" = 850107 
    OR "gid" = 1030107
    OR "gid" = 1030105 OR "gid" = 1040104 
    OR "gid" = 1060110 OR "gid" = 1050111 
    OR "gid" = 1040112 
    OR "gid" = 1090109 OR "gid" = 1080108 
    OR "gid" = 1180114 OR "gid" = 1170113 OR "gid" = 1160112
    OR "gid" = 1180132 OR "gid" = 1190133 
    OR "gid" = 1170127
    OR "gid" = 1190131 
    OR "gid" = 1100120 
    OR "gid" = 1100118 
    OR "gid" = 970113  
    OR "gid" = 960108  
    OR "gid" = 930111  
    OR "gid" = 880106 OR "gid" = 860106 
    OR "gid" = 930107  
    OR "gid" = 1000110 OR "gid" = 990109
    OR "gid" = 1040110 OR "gid" = 1030111 OR "gid" = 1020110 OR "gid" = 1010111
    OR "gid" = 1150113 
    OR "gid" = 1120110
    OR "gid" = 970109
    OR "gid" = 920108 OR "gid" = 930109
    OR "gid" = 1040106

    
    OR "gid" = 1080122 OR "gid" = 1070123 OR "gid" = 1090123 OR "gid" = 1080124 #mountbatten
    OR "gid" = 1110119 OR "gid" = 1120118 # kallang mrt
    OR "gid" = 980114 # promenade
    OR "gid" = 980112 OR "gid" = 990111 # esplanade
    OR "gid" = 950107 OR "gid" = 940108 # cq mrt
    OR "gid" = 920110 OR "gid" = 910109 # Rp
    OR "gid" = 1010109 # bras basah
    OR "gid" = 900108 # Telok Ayer
    OR "gid" = 880104 OR "gid" = 870105 # outram
    OR "gid" = 870107 OR "gid" = 850107 #tanjong pagar
    OR "gid" = 1030107 #dhoby gaut
    OR "gid" = 1030105 OR "gid" = 1040104 #somerset
    OR "gid" = 1060110 OR "gid" = 1050111 #rochor 
    OR "gid" = 1040112 #bugis
    OR "gid" = 1090109 OR "gid" = 1080108 #little india
    OR "gid" = 1180114 OR "gid" = 1170113 OR "gid" = 1160112 #boon keng
    OR "gid" = 1180132 OR "gid" = 1190133 #eunos
    OR "gid" = 1170127 # payalebar


    OR "gid" = 1190131 # eunos walkway
    OR "gid" = 1100120 # underpass - Kallang mrt
    OR "gid" = 1100118 # along side river - Kallang MRT
    OR "gid" = 970113  # from marina square to Promenade MRT
    OR "gid" = 960108  # alternative bridge to Clarke Quay MRT
    OR "gid" = 930111  # alternative link to Raffles Place MRT (passes by Merlion)
    OR "gid" = 880106 OR "gid" = 860106 # link way through shophouses to Tanjong Pagar MRT
    OR "gid" = 930107  # alternative way to Clark Quay MRT from Chinatown area
    OR "gid" = 1000110 OR "gid" = 990109 # alternative way to Bras Basah from Padang
    OR "gid" = 1040110 OR "gid" = 1030111 OR "gid" = 1020110 OR "gid" = 1010111 # popular area, waterloo street, queens street, victoria street. link to Bugis & City Hall
    OR "gid" = 1150113  # missing link in path to Boon Keng MRT
    OR "gid" = 1120110  # missing link in path to Farrer Park MRT
    OR "gid" = 970109  # mising link in path across North Bridge Road to City Hall
    OR "gid" = 920108 OR "gid" = 930109  # patch up gap in CBD cluster

    OR "gid" = 1040106  #missing link in orchard road
