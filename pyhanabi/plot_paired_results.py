import matplotlib.pyplot as plt
import os, json, argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="plot crossplay results forhanabi")
    parser.add_argument("--json", type=str, default="paired_results.json")
    args = parser.parse_args()
    return args

def sort_exps(exp_name):
    # exp_parts = exp_name.split("_")
    exp_num = 0
    if "sad" in exp_name[:3]:
        exp_num += 0
        if "predrew" in exp_name:
            exp_num += 2000
        if "-op" in exp_name:
            exp_num += 1000
    elif "obl" in exp_name or "OBL" in exp_name:
        exp_num += 10000
    elif "pb" in exp_name[:2]:
        exp_num += 20000 #+ int(exp_name[2])*100
    elif "ow" in exp_name[:2]:
        exp_num += 30000 #+ int(exp_name[2])*100
    elif "aw" in exp_name[:2]:
        exp_num += 40000 #+ int(exp_name[2])*100
    elif "bwa" in exp_name[:3]:
        exp_num += 40000 #+ int(exp_name[2])*100
    elif "op-aw" in exp_name[:5]:
        exp_num += 50000 #+ int(exp_name[2])*100
    elif "caw" in exp_name[:3]:
        exp_num += 60000 #+ int(exp_name[2])*100
    elif "saw" in exp_name[:3]:
        exp_num += 70000 #+ int(exp_name[2])*100
    elif "bgd" in exp_name[:3] or "fld" in exp_name[:3]:
        exp_num += 80000 #+ int(exp_name[2])*100

    try:
        if 'xe' in exp_name:
            exp_num += int(exp_name[exp_name.index('xe')+2:].split('_')[0])*100
        if 'sq' in exp_name:
            exp_num += int(exp_name[exp_name.index('sq')+2:].split('_')[0])*100
    except:
        print('couldnt find xe or sq')

    if "pred" in exp_name or "aux" in exp_name:
            exp_num += 100
    if "2ff" in exp_name:
        exp_num += 1
    if "sk" in exp_name:
        exp_num += 2
    
        
    # exp_num = exp_name.replace("sad","00").replace("pb","1").replace("2ffsk","3").replace("sk","2").replace("2ff","1").replace("predrew","2").replace("pred","1").replace("-xe","").replace("_","")
    return int(exp_num)
    # if len(exp_parts) > 0:
    #     first = ["sad","pb"]

def color_func(lbl):
    if "ow" in lbl or "aw" in lbl or "bwa" in lbl:
        color = "cornflowerblue"
    elif "pb" in lbl:
        color = "violet"
    elif "-op" in lbl:
        color = "lightcoral"
    elif "obl" in lbl or "OBL" in lbl:
        color = "green"
    else:
        color = "black"
    return color

bold_func = lambda lbl: True if "pred" in lbl or 'aux' in lbl else False

def plot_paired_results(exp_names, label_exp_names, save_file, fontsize = 2, grid_size = 15):
    result_array = np.zeros((len(exp_names),len(exp_names)),dtype=np.float32)
    for i,exp0 in enumerate(exp_names):
        for j, exp1 in enumerate(exp_names):
            if "$$".join([exp0,exp1]) in results_dict.keys():
                result_array[i,j] = results_dict["$$".join([exp0,exp1])][0]

    vmax = 25 #np.max(result_array)
    vmin = 0  #np.min(result_array)


    # import numpy as np
    # import matplotlib
    # import matplotlib.pyplot as plt

    # vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
    #               "potato", "wheat", "barley"]
    # farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
    #            "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

    # harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
    #                     [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
    #                     [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
    #                     [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
    #                     [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
    #                     [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
    #                     [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])
    plt.clf()
    num_color = lambda num : "gray" if num > 18 else "white"
    fig = plt.figure()
    grid = plt.GridSpec(grid_size,grid_size)
    # grid.update(wspace=0.0,hspace=0.0)
    hmap_ax = fig.add_subplot(grid[:grid_size-1,:grid_size-1])
    sum_ax = fig.add_subplot(grid[-1,:-1])
    rank_ax = fig.add_subplot(grid[:-1,-1])
    avg_ax = fig.add_subplot(grid[-1,-1])
    # fig, ax = plt.subplots()



    im = hmap_ax.imshow(result_array, vmin=vmin, vmax=vmax, aspect='auto')

    # We want to show all ticks...
    # hmap_ax.set_xticks(np.arange(len(exp_names)))
    hmap_ax.set_yticks(np.arange(len(exp_names)))
    # hmap_ax.set_yticks([])
    hmap_ax.set_xticks([])

    # create sum heatmap
    result_array_sum = (np.sum(result_array, axis=0) + np.sum(result_array, axis=1) - 2*np.diag(result_array))/(2*(result_array.shape[0]-1))

    # only taking indices of non-OP prior models
    # prior_idx = [idx for idx, name in enumerate(exp_names) if "M" in name and "-op" not in name] # SAD and SAX+AUX only
    # prior_idx = [idx for idx, name in enumerate(exp_names) if "M" in name and "-op" not in name and 'aux' not in name] #SAD only
    # prior_idx = [idx for idx, name in enumerate(exp_names)] #everything
    prior_idx = [idx for idx, name in enumerate(exp_names) if ("M" in name or "sad_" in name or "iql" in name or "vdn" in name) and "-op" not in name and 'aux' not in name] #SAD only
    # prior_idx = [idx for idx, name in enumerate(exp_names) if ("M" in name or "sad_" in name or "iql" in name or "vdn" in name or 'aux' in name[:3]) and "-op" not in name] #SAD and SAD+AUX only


    #calculate prior_idx and other exp means differently 
    #(prior_idx will have self-play score removed and be divided by one-less 
    # to make it more comparable with non-prior idx that do not have the self-play score)
    result_array_zero_selfplay = result_array.copy()
    result_array_zero_selfplay[list(range(result_array_zero_selfplay.shape[0])),list(range(result_array_zero_selfplay.shape[0]))] = 0.
    result_array_zero_selfplay
    result_array_sum = (np.sum(result_array_zero_selfplay[prior_idx,:], axis=0) + np.sum(result_array_zero_selfplay[:,prior_idx], axis=1)) / 2.
    result_array_sum[prior_idx] = result_array_sum[prior_idx] / (len(prior_idx) - 1)
    not_prior_idx = [idx for idx in range(len(exp_names)) if idx not in prior_idx]
    result_array_sum[not_prior_idx] = result_array_sum[not_prior_idx] / len(prior_idx)

    # selfplay_scores = np.diag(result_array)
    # prior_selfplay = np.zeros_like(selfplay_scores)
    # prior_selfplay[prior_idx] = selfplay_scores[prior_idx]
    # result_array_sum = (np.sum(result_array[prior_idx,:], axis=0) + np.sum(result_array[:,prior_idx], axis=1))/(2*len(prior_idx))
    
    # result_array_replace_selfplay = result_array.copy()

    # result_array_replace_selfplay[list(range(result_array_replace_selfplay.shape[0])),list(range(result_array_replace_selfplay.shape[0]))] = result_array_sum

    # # result_array_sum = (np.sum(result_array[prior_idx,:], axis=0) + np.sum(result_array[:,prior_idx], axis=1) - 2*prior_selfplay)/(2*(len(prior_idx.shape[0])-1))
    # result_array_sum = (np.sum(result_array_replace_selfplay[prior_idx,:], axis=0) + np.sum(result_array_replace_selfplay[:,prior_idx], axis=1))/(2*len(prior_idx))

    im_sum = sum_ax.imshow(result_array_sum[None], vmin=vmin, vmax=vmax, aspect='auto')

    # ... and label them with the respective list entries
    sum_ax.set_xticks(np.arange(len(label_exp_names)))
    sum_ax.set_xticklabels(label_exp_names, fontsize=fontsize)
    # sum_ax.set_xticklabels([lbl for lbl in label_exp_names if bold_func(lbl)], fontsize=fontsize, fontweight="bold")
    sum_ax.set_yticks([])
    sum_ax.set_yticklabels([])
    # Loop over data dimensions and create text annotations.
    for i in range(len(result_array_sum)):
        text = sum_ax.text(i, 0, "%3.1f"%result_array_sum[i],
                        ha="center", va="center", color=num_color(result_array_sum[i]), fontsize=fontsize)
    hmap_ax.set_yticklabels(label_exp_names, fontsize=fontsize)
    # hmap_ax.set_yticklabels([lbl for lbl in label_exp_names if bold_func(lbl)], fontsize=fontsize, fontweight="bold")
    # hmap_ax.yaxis.tick_right()
    # Rotate the tick labels and set their alignment.
    # plt.setp(hmap_ax.get_yticklabels(), ha="left")
    # for tick in hmap_ax.yaxis.get_majorticklabels():
    #     tick.set_horizontalalignment("left")
    # hmap_ax.set_yticklabels([])
    hmap_ax.set_xticklabels([])

    # Rotate the tick labels and set their alignment.
    plt.setp(sum_ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # create rank heatmap
    # sorted_labels, sorted_values = zip(*sorted(list(zip(label_exp_names,list(result_array_sum))), key=lambda x: x[1], reverse=True))
    sorted_labels, sorted_values = zip(*sorted(list(zip(exp_names,list(result_array_sum))), key=lambda x: x[1], reverse=True))
    im_rank = rank_ax.imshow(np.array(sorted_values)[:,None], vmin=vmin, vmax=vmax, aspect='auto')
    for i in range(len(sorted_values)):
        text = rank_ax.text(0, i, "%3.1f"%sorted_values[i],
                        ha="center", va="center", color=num_color(sorted_values[i]), fontsize=fontsize)
    rank_ax.set_yticks(np.arange(len(sorted_labels)))
    rank_ax.set_yticklabels(sorted_labels, fontsize=fontsize)
    # rank_ax.set_yticklabels([lbl for lbl in sorted_labels if bold_func(lbl)], fontsize=fontsize, fontweight="bold")
    rank_ax.yaxis.tick_right()
    rank_ax.set_xticks([])
    rank_ax.set_xticklabels([])


    # Loop over data dimensions and create text annotations.
    for i in range(len(exp_names)):
        for j in range(len(exp_names)):
            text = hmap_ax.text(j, i, "%3.1f"%result_array[i, j],
                        ha="center", va="center", color=num_color(result_array[i, j]), fontsize=fontsize)
    # fig.subplots_adjust(wspace=0,hspace=0)
    # fig.align_xlabels()
    # hmap_ax.set_title("Paired evaluation of trained agents")

    crossplay_mean = (np.sum(result_array) - np.sum(np.diag(result_array)))/(result_array.shape[0] * (result_array.shape[0]-1))
    crossplay_mean = np.array(crossplay_mean).reshape((1,1))
    avg_ax.imshow(crossplay_mean, vmin=vmin, vmax=vmax, aspect='auto')
    text = avg_ax.text(0, 0, "%3.1f"%crossplay_mean[0,0],
                        ha="center", va="center", color=num_color(crossplay_mean[0,0]), fontsize=fontsize)
    avg_ax.set_xticks([])
    avg_ax.set_xticklabels([])
    avg_ax.set_yticks([])
    avg_ax.set_yticklabels([])


    for ax in [hmap_ax, sum_ax, rank_ax]:
        for tklbl in ax.get_yticklabels():
            if bold_func(tklbl.get_text()):
                tklbl.set_weight("bold")
            tklbl.set_color(color_func(tklbl.get_text()))
        for tklbl in ax.get_xticklabels():
            if bold_func(tklbl.get_text()):
                tklbl.set_weight("bold")
            tklbl.set_color(color_func(tklbl.get_text()))

    fig.tight_layout()
    plt.savefig(save_file, format='png', dpi=350)#200) 

def get_crossplay_scores(result_array, alg_idx, nonZSC_idx, sem_array=None):
    assert len(alg_idx) > 0, "alg_idx is 0"
    alg_idx = np.array(alg_idx)
    not_alg_idx = [i for i in range(result_array.shape[0]) if i not in alg_idx]
    not_alg_idx = np.array(not_alg_idx)
    nonZSC_idx = np.array(nonZSC_idx)
    
    intra_alg_scores = result_array[alg_idx[:, None],alg_idx]
    intra_alg_sem = sem_array[alg_idx[:, None],alg_idx]
    
    #self-play mean
    sp_score = np.mean(np.diag(intra_alg_scores))

    #to combine sem, square it all, get mean, divide by N / then sqrt
    sp_sem = np.sqrt(np.mean(np.square(np.diag(intra_alg_sem)))/len(np.diag(intra_alg_sem)))
    
    #intra-alg XP mean
    if intra_alg_scores.shape[0] > 1:
        intra_xp_score = (np.sum(intra_alg_scores) - np.sum(np.diag(intra_alg_scores)))/(intra_alg_scores.shape[0]**2 - intra_alg_scores.shape[0])
        # [list(intra_alg_scores[:])]
        intra_xp_sem = np.sqrt(((np.sum(np.square(intra_alg_sem)) - np.sum(np.square(np.diag(intra_alg_sem))))/(intra_alg_sem.shape[0]**2 - intra_alg_sem.shape[0])) / (intra_alg_sem.shape[0]**2 - intra_alg_sem.shape[0]))
    else:
        intra_xp_score = -1
        intra_xp_sem = -1

    #inter-alg XP mean
    inter_xp_score = (np.mean(result_array[alg_idx[:, None], not_alg_idx]) + np.mean(result_array[not_alg_idx[:, None], alg_idx]))/2.
    inter_xp_sem = np.sqrt((np.mean(np.square(sem_array[alg_idx[:, None], not_alg_idx])) + np.mean(np.square(sem_array[not_alg_idx[:, None], alg_idx])))/(2*2*len(not_alg_idx)*len(alg_idx)))

    #non-ZSC XP mean
    nonZSC_xp_score = (np.mean(result_array[alg_idx[:, None], nonZSC_idx]) + np.mean(result_array[nonZSC_idx[:, None], alg_idx]))/2.
    nonZSC_xp_sem = np.sqrt((np.mean(sem_array[alg_idx[:, None], nonZSC_idx]) + np.mean(sem_array[nonZSC_idx[:, None], alg_idx]))/(2*2*len(alg_idx)*len(nonZSC_idx)))

    return (sp_score, sp_sem), (intra_xp_score, intra_xp_sem), (inter_xp_score, inter_xp_sem), (nonZSC_xp_score, nonZSC_xp_sem)



def get_all_crossplay_scores(exp_names, label_exp_names):
    result_array = np.zeros((len(exp_names),len(exp_names)),dtype=np.float32)
    sem_array = np.zeros((len(exp_names),len(exp_names)),dtype=np.float32)
    for i,exp0 in enumerate(exp_names):
        for j, exp1 in enumerate(exp_names):
            if "$$".join([exp0,exp1]) in results_dict.keys():
                result_array[i,j] = results_dict["$$".join([exp0,exp1])][0]
                sem_array[i,j] = results_dict["$$".join([exp0,exp1])][1]
    
    
    #define non-ZSC IDX
    nonZSC_idx = list([idx for idx, name in enumerate(exp_names) if "M" in name and "-op" not in name and 'aux' not in name])
    
    #get XP means
    IQL_idx = list([idx for idx, name in enumerate(label_exp_names) if "iql" in name]) 
    VDN_idx = list([idx for idx, name in enumerate(label_exp_names) if "vdn" in name]) 
    
    SAD_idx = list([idx for idx, name in enumerate(label_exp_names) if "sad" in name and "-op" not in name and 'aux' not in name]) 
    SAD_AUX_idx = list([idx for idx, name in enumerate(label_exp_names) if "sad" in name and "-op" not in name and 'aux' in name]) 
    
    SAD_OP_idx = list([idx for idx, name in enumerate(label_exp_names) if "sad" in name and "-op" in name and 'aux' not in name]) 
    SAD_AUX_OP_idx = list([idx for idx, name in enumerate(label_exp_names) if "sad" in name and "-op" in name and 'aux' in name]) 
    
    OBL_idx = list([idx for idx, name in enumerate(label_exp_names) if "OBL" in name]) 
    
    AP_idx = list([idx for idx, name in enumerate(label_exp_names) if "aw" in name and "-op" not in name and 'aux' not in name]) 
    AP_AUX_idx = list([idx for idx, name in enumerate(label_exp_names) if "aw" in name and "-op" not in name and 'aux' in name]) 

    if len(AP_idx) < 1:
        AP_idx = list([idx for idx, name in enumerate(label_exp_names) if "bwa" in name and "-op" not in name and 'aux' not in name]) 

    if len(AP_AUX_idx) < 1:
        AP_AUX_idx = list([idx for idx, name in enumerate(label_exp_names) if "bwa" in name and "-op" not in name and 'aux' in name]) 

    xp_score_dict = {}

    for alg_name, alg_idx in [('IQL',IQL_idx), ('VDN',VDN_idx), ('SAD',SAD_idx), ('SAD+AUX',SAD_AUX_idx), \
                              ('SAD+OP',SAD_OP_idx), ('SAD+AUX+OP',SAD_AUX_OP_idx), ('OBL',OBL_idx), \
                              ('SAD+AP',AP_idx), ('SAD+AUX+AP',AP_AUX_idx)]:
        if len(alg_idx) > 0:
            xp_score_dict[alg_name] = get_crossplay_scores(result_array, alg_idx, nonZSC_idx, sem_array=sem_array)
    
    return xp_score_dict

def plot_crossplay_results(exp_names_x, exp_names_y, label_exp_names_x, label_exp_names_y, save_file, fontsize=6, grid_size=8):
    result_array = np.zeros((len(exp_names_y),len(exp_names_x)),dtype=np.float32)
    for i,exp0 in enumerate(exp_names_y):
        for j, exp1 in enumerate(exp_names_x):
            if "$$".join([exp0,exp1]) in results_dict.keys():
                result_array[i,j] = results_dict["$$".join([exp0,exp1])][0]

    vmax = 25 #np.max(result_array)
    vmin = 0  #np.min(result_array)


    # import numpy as np
    # import matplotlib
    # import matplotlib.pyplot as plt

    # vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
    #               "potato", "wheat", "barley"]
    # farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
    #            "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

    # harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
    #                     [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
    #                     [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
    #                     [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
    #                     [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
    #                     [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
    #                     [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])
    plt.clf()
    fontsize = 6
    grid_size = 8
    num_color = lambda num : "gray" if num > 18 else "white"
    fig = plt.figure()
    grid = plt.GridSpec(grid_size,grid_size)
    # grid.update(wspace=0.0,hspace=0.0)
    hmap_ax = fig.add_subplot(grid[:grid_size-1,:grid_size-1])
    sum_ax = fig.add_subplot(grid[-1,:-1])
    rank_ax = fig.add_subplot(grid[:-1,-1])
    avg_ax = fig.add_subplot(grid[-1,-1])
    # fig, ax = plt.subplots()



    im = hmap_ax.imshow(result_array, vmin=vmin, vmax=vmax, aspect='auto')

    # We want to show all ticks...
    # hmap_ax.set_xticks(np.arange(len(exp_names)))
    hmap_ax.set_yticks(np.arange(len(exp_names_y)))
    # hmap_ax.set_yticks([])
    hmap_ax.set_xticks([])

    # create sum heatmap
    # result_array_sum = (np.sum(result_array, axis=0) + np.sum(result_array, axis=1) - 2*np.diag(result_array))/(2*(result_array.shape[0]-1))

    # only taking indices of non-OP prior models
    # prior_idx = [idx for idx, name in enumerate(exp_names) if "M" in name and "-op" not in name]
    prior_idx = list([idx for idx, name in enumerate(exp_names_y) if "M" in name and "-op" not in name and 'aux' not in name]) # + \
                # list([idx for idx, name in enumerate(exp_names_x) if "M" in name and "-op" not in name and 'aux' not in name])
    # selfplay_scores = np.diag(result_array)
    # prior_selfplay = np.zeros_like(selfplay_scores)
    # prior_selfplay[prior_idx] = selfplay_scores[prior_idx]
    # result_array_sum = (np.sum(result_array[prior_idx,:], axis=0) + np.sum(result_array[:,prior_idx], axis=1))/(2*len(prior_idx))
    
    # result_array_replace_selfplay = result_array.copy()

    # result_array_replace_selfplay[list(range(result_array_replace_selfplay.shape[0])),list(range(result_array_replace_selfplay.shape[0]))] = result_array_sum

    # result_array_sum = (np.sum(result_array[prior_idx,:], axis=0) + np.sum(result_array[:,prior_idx], axis=1) - 2*prior_selfplay)/(2*(len(prior_idx.shape[0])-1))
    result_array_sum_x = np.sum(result_array[prior_idx,:], axis=0)/len(prior_idx)

    im_sum = sum_ax.imshow(result_array_sum_x[None], vmin=vmin, vmax=vmax, aspect='auto')

    # ... and label them with the respective list entries
    sum_ax.set_xticks(np.arange(len(label_exp_names_x)))
    sum_ax.set_xticklabels(label_exp_names_x, fontsize=fontsize)
    # sum_ax.set_xticklabels([lbl for lbl in label_exp_names if bold_func(lbl)], fontsize=fontsize, fontweight="bold")
    sum_ax.set_yticks([])
    sum_ax.set_yticklabels([])
    # Loop over data dimensions and create text annotations.
    for i in range(len(result_array_sum_x)):
        text = sum_ax.text(i, 0, "%3.1f"%result_array_sum_x[i],
                        ha="center", va="center", color=num_color(result_array_sum_x[i]), fontsize=fontsize)
    hmap_ax.set_yticklabels(label_exp_names_y, fontsize=fontsize)
    # hmap_ax.set_yticklabels([lbl for lbl in label_exp_names if bold_func(lbl)], fontsize=fontsize, fontweight="bold")
    # hmap_ax.yaxis.tick_right()
    # Rotate the tick labels and set their alignment.
    # plt.setp(hmap_ax.get_yticklabels(), ha="left")
    # for tick in hmap_ax.yaxis.get_majorticklabels():
    #     tick.set_horizontalalignment("left")
    # hmap_ax.set_yticklabels([])
    hmap_ax.set_xticklabels([])

    # Rotate the tick labels and set their alignment.
    plt.setp(sum_ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # create rank heatmap
    sorted_labels, sorted_values = zip(*sorted(list(zip(label_exp_names_x,list(result_array_sum_x))), key=lambda x: x[1], reverse=True))
    im_rank = rank_ax.imshow(np.array(sorted_values)[:,None], vmin=vmin, vmax=vmax, aspect='auto')
    for i in range(len(sorted_values)):
        text = rank_ax.text(0, i, "%3.1f"%sorted_values[i],
                        ha="center", va="center", color=num_color(sorted_values[i]), fontsize=fontsize)
    rank_ax.set_yticks(np.arange(len(sorted_labels)))
    rank_ax.set_yticklabels(sorted_labels, fontsize=fontsize)
    # rank_ax.set_yticklabels([lbl for lbl in sorted_labels if bold_func(lbl)], fontsize=fontsize, fontweight="bold")
    rank_ax.yaxis.tick_right()
    rank_ax.set_xticks([])
    rank_ax.set_xticklabels([])


    # Loop over data dimensions and create text annotations.
    for i in range(len(exp_names_y)):
        for j in range(len(exp_names_x)):
            text = hmap_ax.text(j, i, "%3.1f"%result_array[i, j],
                        ha="center", va="center", color=num_color(result_array[i, j]), fontsize=fontsize)
    # fig.subplots_adjust(wspace=0,hspace=0)
    # fig.align_xlabels()
    # hmap_ax.set_title("Paired evaluation of trained agents")

    crossplay_mean = np.mean(result_array)
    crossplay_mean = np.array(crossplay_mean).reshape((1,1))
    avg_ax.imshow(crossplay_mean, vmin=vmin, vmax=vmax, aspect='auto')
    text = avg_ax.text(0, 0, "%3.1f"%crossplay_mean[0,0],
                        ha="center", va="center", color=num_color(crossplay_mean[0,0]), fontsize=fontsize)
    avg_ax.set_xticks([])
    avg_ax.set_xticklabels([])
    avg_ax.set_yticks([])
    avg_ax.set_yticklabels([])


    for ax in [hmap_ax, sum_ax, rank_ax]:
        for tklbl in ax.get_yticklabels():
            if bold_func(tklbl.get_text()):
                tklbl.set_weight("bold")
            tklbl.set_color(color_func(tklbl.get_text()))
        for tklbl in ax.get_xticklabels():
            if bold_func(tklbl.get_text()):
                tklbl.set_weight("bold")
            tklbl.set_color(color_func(tklbl.get_text()))

    fig.tight_layout()
    plt.savefig(save_file, format='png', dpi=350)#200)

def label_prior_models(exp_name):
    if "M" in exp_name.split("_")[-1]:
        Mnum = int(exp_name.split("_")[-1].replace(".pthw","")[1:])
        suffix = ""
        if Mnum >= 0 and Mnum < 3:
            suffix = ""
            # num_ff_layer = 1
            # skip_connect = False
        elif Mnum >= 3 and Mnum < 6:
            suffix = "_sk"
            # num_ff_layer = 1
            # skip_connect = True
        elif Mnum >= 6 and Mnum < 9:
            suffix = "_2ff"
            # num_ff_layer = 2
            # skip_connect = False
        else:
            suffix = "_2ffsk"
            # num_ff_layer = 2
            # skip_connect = True
        return "_".join(exp_name.split("_")[:-1]) + suffix
    elif "_" not in exp_name:
        return exp_name.replace(".pthw","")
    else:
        return "_".join(exp_name.split("_")[:-1])

args = parse_args()

results_dict = None
if os.path.exists(args.json):
    with open(args.json,'r') as rf:
        results_dict = json.loads(rf.read())

# # simple code to save prior results to another json file
# prior_dict = {}
# for key in results_dict.keys():
#     one, two = key.split("$$")
#     if "M" in one and "M" in two:
#         prior_dict[key] = results_dict[key]


# with open("prior_crossplay_results.json",'w') as wf:
#     json.dump(prior_dict, wf)

# quit()

plot_dir = "paired_results_plots"
# plot_dir = ""

all_exp_names = sorted(list(set([key.split("$$")[0] for key in results_dict.keys()])))
exp_names = [exp_name for exp_name in all_exp_names if "$$".join([exp_name,exp_name]) in results_dict.keys() and results_dict["$$".join([exp_name,exp_name])][0] >= 15.]
exp_names = [exp_name for exp_name in all_exp_names if 'obl' not in exp_name]
# exp_names = [exp_name for exp_name in exp_names if "sad_" not in exp_name or "M" in exp_name] #for older results with self-trained SAD agents
label_exp_names = [label_prior_models(exp_name).replace("prebrief","pb").replace("_2p","").replace("xent","xe").replace("pred","aux") for exp_name in exp_names]
max_length = np.max([len(nm) for nm in label_exp_names])
# label_exp_names = [nm + " "*(max_length - len(nm)) for nm in label_exp_names]
exp_names, label_exp_names = zip(*sorted(list(zip(exp_names,label_exp_names)), key=lambda x: sort_exps(x[1])))
print("removed exps for failure to self-play:")
print(set(all_exp_names) - set(exp_names))

print("looking at cross-play scores between %d agents totaling %d pairings calculated from %d individual games"%(len(exp_names), len(exp_names)**2, (len(exp_names)**2)*2500))

all_exp_names, all_label_exp_names = exp_names, label_exp_names

plot_paired_results(exp_names, label_exp_names, os.path.join(plot_dir, os.path.basename(args.json.replace(".json",".png"))))

baseline_names, baseline_exp_names = zip(*list([pair for pair in list(zip(all_exp_names, all_label_exp_names)) if 'M' in pair[0] and not 'aux' in pair[0] and '-op' not in pair[0]]))

# # plot AUX crossplay results
# exp_names, label_exp_names = zip(*list([pair for pair in list(zip(all_exp_names, all_label_exp_names)) if 'M' in pair[0] and '-op' not in pair[1] and 'aux' in pair[1]]))
# if len(exp_names) > 0:
#   plot_crossplay_results(exp_names, baseline_names, label_exp_names, baseline_exp_names, os.path.join(plot_dir, "AUX_"+os.path.basename(args.json.replace(".json",".png"))))

# # plot OP crossplay results
# exp_names, label_exp_names = zip(*list([pair for pair in list(zip(all_exp_names, all_label_exp_names)) if 'M' in pair[0] and '-op' in pair[1] and 'aux' not in pair[1]]))
# if len(exp_names) > 0:
#     plot_crossplay_results(exp_names, baseline_names, label_exp_names, baseline_exp_names, os.path.join(plot_dir, "OP_"+os.path.basename(args.json.replace(".json",".png"))))

# # plot OP + AUX crossplay results
# exp_names, label_exp_names = zip(*list([pair for pair in list(zip(all_exp_names, all_label_exp_names)) if 'M' in pair[0] and '-op' in pair[1] and 'aux' in pair[1]]))
# if len(exp_names) > 0:
#     plot_crossplay_results(exp_names, baseline_names, label_exp_names, baseline_exp_names, os.path.join(plot_dir, "OP-AUX_"+os.path.basename(args.json.replace(".json",".png"))))

# plot AW crossplay results
exp_names_label_exp_names = list([pair for pair in list(zip(all_exp_names, all_label_exp_names)) if 'aw' in pair[0] and 'aux' not in pair[1]])
if len(exp_names_label_exp_names) > 0:
    plot_crossplay_results(exp_names, baseline_names, label_exp_names, baseline_exp_names, os.path.join(plot_dir, "AW_"+os.path.basename(args.json.replace(".json",".png"))))
    plot_paired_results(exp_names, label_exp_names, os.path.join(plot_dir, "self_AW_"+os.path.basename(args.json.replace(".json",".png"))), fontsize = 6, grid_size = 8)

# plot AW + AUX crossplay results
exp_names_label_exp_names = list([pair for pair in list(zip(all_exp_names, all_label_exp_names)) if 'aw' in pair[0] and 'aux' in pair[1]])
if len(exp_names_label_exp_names) > 0:
    exp_names, label_exp_names = zip(*exp_names_label_exp_names)
    plot_crossplay_results(exp_names, baseline_names, label_exp_names, baseline_exp_names, os.path.join(plot_dir, "AW-AUX_"+os.path.basename(args.json.replace(".json",".png"))))
    plot_paired_results(exp_names, label_exp_names, os.path.join(plot_dir, "self_AW-AUX_"+os.path.basename(args.json.replace(".json",".png"))), fontsize = 6, grid_size = 8)

# plot BWA crossplay results
exp_names_label_exp_names = list([pair for pair in list(zip(all_exp_names, all_label_exp_names)) if 'bwa' in pair[0] and 'aux' not in pair[1]])
if len(exp_names_label_exp_names) > 0:
    exp_names, label_exp_names = zip(*exp_names_label_exp_names)
    plot_crossplay_results(exp_names, baseline_names, label_exp_names, baseline_exp_names, os.path.join(plot_dir, "AW_"+os.path.basename(args.json.replace(".json",".png"))))
    plot_paired_results(exp_names, label_exp_names, os.path.join(plot_dir, "self_AW_"+os.path.basename(args.json.replace(".json",".png"))), fontsize = 6, grid_size = 8)

# plot BWA + AUX crossplay results
exp_names_label_exp_names = list([pair for pair in list(zip(all_exp_names, all_label_exp_names)) if 'bwa' in pair[0] and 'aux' in pair[1]])
if len(exp_names_label_exp_names) > 0:
    exp_names, label_exp_names = zip(*exp_names_label_exp_names)
    plot_crossplay_results(exp_names, baseline_names, label_exp_names, baseline_exp_names, os.path.join(plot_dir, "AW-AUX_"+os.path.basename(args.json.replace(".json",".png"))))
    plot_paired_results(exp_names, label_exp_names, os.path.join(plot_dir, "self_AW-AUX_"+os.path.basename(args.json.replace(".json",".png"))), fontsize = 6, grid_size = 8)

# exp_names, label_exp_names = zip(*list([pair for pair in list(zip(all_exp_names, all_label_exp_names)) if 'M' in pair[0] and '-op' not in pair[1] and 'aux' in pair[1]]))
# if len(exp_names) > 0:
#     plot_paired_results(exp_names, label_exp_names, os.path.join(plot_dir, "AUX.png"), fontsize = 6, grid_size = 8)

# exp_names, label_exp_names = zip(*list([pair for pair in list(zip(all_exp_names, all_label_exp_names)) if 'M' in pair[0] and '-op' not in pair[1] and 'aux' not in pair[1]]))
# if len(exp_names) > 0:
#     plot_paired_results(exp_names, label_exp_names, os.path.join(plot_dir, "SAD.png"), fontsize = 6, grid_size = 8)

# exp_names, label_exp_names = zip(*list([pair for pair in list(zip(all_exp_names, all_label_exp_names)) if 'M' in pair[0] and '-op' in pair[1] and 'aux' in pair[1]]))
# if len(exp_names) > 0:
#     plot_paired_results(exp_names, label_exp_names, os.path.join(plot_dir, "OP-AUX.png"), fontsize = 6, grid_size = 8)

# exp_names, label_exp_names = zip(*list([pair for pair in list(zip(all_exp_names, all_label_exp_names)) if 'M' in pair[0] and '-op' in pair[1] and 'aux' not in pair[1]]))
# if len(exp_names) > 0:
#     plot_paired_results(exp_names, label_exp_names, os.path.join(plot_dir, "OP.png"), fontsize = 6, grid_size = 8)

# exp_names, label_exp_names = zip(*list([pair for pair in list(zip(all_exp_names, all_label_exp_names)) if 'sad' in pair[1] and not 'aux' in pair[1]]))
# if len(exp_names) > 0:
#     plot_paired_results(exp_names, label_exp_names, "paired_results_plots/sad.png")
# exp_names, label_exp_names = zip(*list([pair for pair in list(zip(all_exp_names, all_label_exp_names)) if 'sad_aux' in pair[1] and not 'auxrew' in pair[1]]))
# if len(exp_names) > 0:
#     plot_paired_results(exp_names, label_exp_names, "paired_results_plots/sad_pred.png")
# exp_names, label_exp_names = zip(*list([pair for pair in list(zip(all_exp_names, all_label_exp_names)) if 'sad_auxrew' in pair[1]]))
# if len(exp_names) > 0:
#     plot_paired_results(exp_names, label_exp_names, "paired_results_plots/sad_predrew.png")
# exp_names, label_exp_names = zip(*list([pair for pair in list(zip(all_exp_names, all_label_exp_names)) if 'pb' in pair[1] and not 'aux' in pair[1]]))
# if len(exp_names) > 0:
#     plot_paired_results(exp_names, label_exp_names, "paired_results_plots/pb.png")
# exp_names, label_exp_names = zip(*list([pair for pair in list(zip(all_exp_names, all_label_exp_names)) if 'pb' in pair[1] and 'aux' in pair[1]]))
# if len(exp_names) > 0:
#     plot_paired_results(exp_names, label_exp_names, "paired_results_plots/pb_pred.png")


xp_score_dict = get_all_crossplay_scores(all_exp_names, all_label_exp_names)

alg_names = ['IQL', 'VDN', 'SAD', 'SAD\\newline+AUX', 'SAD\\newline\\newline+OP', 'SAD\\newline+AUX\\newline+OP', 'OBL','SAD\\newline\\newline+AP', 'SAD\\newline+AUX\\newline+AP']
alg_names = [alg_name for alg_name in alg_names if alg_name.replace("\\newline","") in xp_score_dict.keys()]
score_names = ['Self-Play', 'Intra-Alg XP', 'Inter-Alg XP', 'non-ZSC XP']

# print("\\begin{tabular}{c"+"|l"*len(alg_names)+"} \\toprule")

# level = 0
# titleline = ""
# done = False
# while not done:
#     # titleline += "& "
#     done = True
#     for alg_name in alg_names:
#         alg_name_split = alg_name.split("\\newline")
#         if len(alg_name_split) > level:
#             done = not len(alg_name_split) > level+1
#             titleline += "&\t%s "%(alg_name_split[level])
#         else:
#             titleline += "&\t "
#     titleline += "\\\\ \n"
#     level += 1
# titleline += " \\midrule"
# print(titleline)

# for i,score_name in enumerate(['Self-Play', 'Intra-Alg XP', 'Inter-Alg XP', 'non-ZSC XP']):
#     scores = []
#     for alg_name in alg_names:
#         alg_name = alg_name.replace("\\newline","")
#         scores.append(xp_score_dict[alg_name][i])
#     max_score = np.max(list(zip(*scores))[0])
#     score_str = ""
#     for score in scores:
#         if score[0] == max_score:
#             score_str += ("& \\textbf{%2.1f}$\\pm "%score[0])+("%.2f$"%score[1])[1:]
#         else:
#             score_str += ("& $%2.1f\\pm "%score[0])+("%.2f$"%score[1])[1:]

#     print(score_name +"\t" + score_str + "\\\\")

# print("\\end{tabular}")


# Do it where algorithms are on the side
print("\\begin{tabular}{l"+"|r"*len(score_names)+"} \\toprule")

level = 0
titleline = ""
done = False
while not done:
    # titleline += "& "
    done = True
    for alg_name in score_names:
        alg_name_split = alg_name.split("\\newline")
        if len(alg_name_split) > level:
            done = not len(alg_name_split) > level+1
            titleline += "&\t%s "%(alg_name_split[level])
        else:
            titleline += "&\t "
    titleline += "\\\\ \n"
    level += 1
titleline += " \\midrule"
print(titleline)

score_maxes = [(score_name, np.max([xp_score_dict[alg_name.replace("\\newline","")][i] for alg_name in alg_names])) for i, score_name in enumerate(score_names)]
score_maxes = dict(score_maxes)

for alg_name in alg_names:
    scores = []
    score_str = ""
    alg_name = alg_name.replace("\\newline","")
    for i, score_name in enumerate(score_names):
        # scores.append(xp_score_dict[alg_name][i])
    # max_score = np.max(list(zip(*scores))[0])
    # for score in score_names, scores:
        score = xp_score_dict[alg_name][i]    
        if score[0] == score_maxes[score_name]:
            score_str += ("& \\textbf{%2.1f}$\\pm$ "%score[0])+"\\textbf{"+("%.2f}"%score[1])[1:]
        else:
            score_str += ("& $%2.1f\\pm$ "%score[0])+("%.2f"%score[1])[1:]

    print(alg_name +"\t" + score_str + "\\\\")

print("\\end{tabular}")
















# '''["dimgray","lightgray","lightsteelblue","cornflowerblue","royalblue","violet","mediumorchid","darkviolet","firebrick"]'''
# colors = {
#     "sad":"lightgray",
#     "predtask":"lightgray",
#     "pid":"lightcoral",
#     # "2ffsk":"royalblue",
#     "prebrief":"cornflowerblue",
#     "pid_prebrief":"violet",
#     # "prebrief_sqrt":"darkviolet",
# }

# darker_colors = {
#     "lightgray":"dimgray",
#     "lightcoral":"firebrick",
#     "cornflowerblue":"royalblue",
#     "violet":"darkviolet",
# }

# seen_labels = set()
# for level in [5,10,15,20,21,22,23,24]:
#     plt.axhline(level,linestyle="dotted",color="lightgray")
# for file_key in sorted(results_dict.keys())[::-1]:
#     # print(results_dict[file_key])
#     x, scores, perfects = zip(*results_dict[file_key])
#     color = "lightgray"
#     for typekey in colors.keys():
#         if typekey in file_key:
#             color = colors[typekey]
#             if "2ffsk" in file_key:
#                 color = darker_colors[color]
#     # plt.xscale("symlog")
#     label = "_".join(file_key.split("_")[:-1])
#     plt.plot(x, scores, label=None if label in seen_labels else label, color=color)
#     seen_labels.update([label])

# plt.xlabel("Epochs")
# plt.ylabel("Hanabi Score")
# plt.legend()
# plt.savefig("results.png",format='png')
# plt.xlim((0,500))
# plt.ylim((10,25))
# plt.savefig("results_500.png",format='png')