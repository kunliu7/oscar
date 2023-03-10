import numpy as np
import os
import matplotlib.pyplot as plt

initial_points = [[[
[ 0.39217697 ,-0.61707997],
[-0.39317966 , 0.63851928],
[ 0.38951176 ,-0.616413  ],
[ 0.38953    ,-0.61642523],
[ 0.3895388  ,-0.61632672],
[-0.39317977 , 0.63851857],
[ 0.36076272 ,-0.57803764],
[-0.38049    , 0.58644985],
[-0.38049061 , 0.58644924],
[ 0.36075038 ,-0.57805924],
[ 0.38953338 ,-0.61642002],
[-0.3836609  , 0.59086763],
[-0.38889375 , 0.60692966],
[-0.39317972 , 0.6385194 ],
[ 0.38786957 ,-0.60660995],
[-0.34160951 , 0.60202695],
], [
[ 0.39214484, -0.61710054],
[-0.393635  ,0.639378],
[ 0.38977961, -0.61653461],
[ 0.38966693, -0.61642792],
[ 0.38973261, -0.61666856],
[-0.39352946,  0.63903117],
[ 0.36038895, -0.58173922],
[-0.38146825,  0.58707778],
[-0.36399821,  0.60658416],
[ 0.36031279, -0.58222227],
[ 0.38964035, -0.61681093],
[-0.38348485,  0.59126411],
[-0.38889783,  0.60692833],
[-0.39348212,  0.63886986],
[ 0.38786686, -0.606612  ],
[-0.34376502,  0.6033871 ],
]], [[
[ 0.39222154 ,-0.61711589],
[-0.3931351  , 0.63854508],
[ 0.38958274 ,-0.6164941 ],
[ 0.38954295 ,-0.61648002],
[ 0.38961324 ,-0.61637009],
[-0.3902945  ,0.5936073],
[ 0.36081728 ,-0.5775129 ],
[-0.38046488 , 0.58644208],
[-0.38052381 , 0.58658268],
[ 0.36090837 ,-0.57796149],
[ 0.38957824 ,-0.61630497],
[ 0.38952113 ,-0.6163395 ],
[-0.38891176 , 0.60681531],
[ 0.39219862 ,-0.61702534],
[ 0.38797968 ,-0.60657954],
[-0.34163664 , 0.60208457],
], [
[ 0.39216259 ,-0.61695907],
[-0.39354769 , 0.63931384],
[ 0.3899091  ,-0.61654154],
[ 0.38957198 ,-0.61635174],
[ 0.38976301 ,-0.61651276],
[-0.3903193  , 0.59365427],
[ 0.36028103 ,-0.58179602],
[-0.38146026 , 0.5870686 ],
[-0.38079123 , 0.5872266 ],
[ 0.36025449 ,-0.58230513],
[ 0.38953047 ,-0.61681541],
[ 0.38959613 ,-0.61677436],
[-0.38894557 , 0.60699048],
[ 0.39208039 ,-0.61699561],
[ 0.38771161 ,-0.60650271],
[-0.3438026  , 0.60339252],
]]]
data, labels = [], []
# optimizer_pool = ["COBYLA"]
# optimizer_pool = ["ADAM"]

optimizer_pool = ["ADAM", "COBYLA"]
init_types = ["random", "OSCAR"]
noise_types = ["ideal", "noisy"]

seeds_dict = {
    "ADAM":   [1, 2, 3, 4, 5, 6, 7, 8,    10, 11, 12, 13, 15], # seed 9 is broken
    "COBYLA": [1, 2, 3, 4, 5,    7, 8, 9, 10, 11, 12, 13, 15], # seed 6 is broken
} # seed 14 is missing for both

for j, opt in enumerate(optimizer_pool):
    for init in init_types:
        for i, noise in enumerate(noise_types):
        # labels.append(f"{noise} {init}")
            for seed in seeds_dict[opt]:
                if noise == "noisy":
                    noise = "depolar-0.003-0.007"
                maxiter = 10000 if opt == "ADAM" else 1000
                # initial_point = initial_points[j+1][i][seed] if init=="OSCAR" else "None"
                initial_point = initial_points[j][i][seed] if init == "OSCAR" else "None"
                path = f"figs/optimization/maxcut/sv-{noise}-p=1/maxcut-sv-{noise}-n=16-p=1-{seed=}-{opt}-{maxiter=}-{initial_point}.npz"

                try:
                    opt_data = np.load(path, allow_pickle=True)
                except Exception as e:
                    print(e)
                    print(path)
                    exit(7)
                    
                # data.append(opt_data["min_eigen_solver_result"].item().optimal_value)
                data.append(len(opt_data["optimizer_path"]))
                # if init == "OSCAR":
                #     data[-1] = data[-1] + 250

data = np.array(data)
print(data.shape)

# keep the order of the optimizers, initializations and noise types
# since `len(seeds_dict["ADAM"]) == len(seeds_dict["COBYLA"])`, we can reshape the data in this way
data = data.reshape(len(optimizer_pool), len(init_types), len(noise_types), len(seeds_dict["ADAM"])) 

save_dir = "figs/reduce_QPU_queries"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_fname = f"opt={optimizer_pool}-init={init_types}-noise={noise_types}.npz"
save_path = os.path.join(save_dir, save_fname)
print(f"Saving data to {save_path}")

print(data.shape)
np.savez_compressed(
    save_path, allow_pickle=True, data=data, optimizer_pool=optimizer_pool,
    init_types=init_types, noise_types=noise_types, seeds_dict=seeds_dict,
    OSCAR_additional_queries=250,
)
exit(0)
labels = ["Random initialization", "OSCAR"]

# plt.figure(figsize=(9.6, 4.8))
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
# colors = ["pink", "lightblue", "lightgreen", "violet"]
fontsize = 16

# --- Labels for your data:
x_labels = ["Ideal", "Noisy"]
width = 0.5 / len(x_labels)
xlocations = [x * ((2 + len(data)) * width) for x in range(len(data[0]))]

symbol = "+"
ymin = min([val for dg in data for d in dg for val in d])
ymax = max([val for dg in data for d in dg for val in d])

ax = plt.gca()
ax.set_ylim(ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin))

ax.grid(True, linestyle="dotted")
ax.set_axisbelow(True)

space = len(data) / 2
offset = len(data) / 2

# --- Offset the positions per group:

group_positions = []
for num, dg in enumerate(data):
    _off = 0 - space + (0.5 + num)
    group_positions.append([x + _off * (width + 0.01) for x in xlocations])

for dg, pos, c in zip(data, group_positions, colors):
    ax.boxplot(
        dg,
        sym=symbol,
        # labels=labels_list,
        positions=pos,
        widths=width,
        boxprops=dict(facecolor=c, alpha=0.75),
        # capprops=dict(color=c),
        # whiskerprops=dict(color=c),
        flierprops=dict(color=c, markeredgecolor=c, markerfacecolor=c),
        medianprops=dict(color="yellow"),
        # notch=False,
        # vert=True,
        # whis=1.5,
        # bootstrap=None,
        # usermedians=None,
        # conf_intervals=None,
        patch_artist=True,
    )
ax.set_xticks([], [])
ax.set_xticklabels([])
ax.set_xticks(xlocations)
ax.set_xticklabels(x_labels, fontsize=fontsize)

for i, label in enumerate(labels):
    plt.plot([], c=colors[i], label=label)
plt.ylabel("Number of QPU queries", fontsize=fontsize)
# plt.ylabel("Cost function value", fontsize=fontsize)
# plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.tight_layout()
plt.savefig(f"figs/initialization_box_queries_{optimizer_pool[0]}.png")
plt.close()