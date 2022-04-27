import matplotlib.pyplot as plt
import numpy as np

def smooth(data, weight = 0.9):
    last = data[0]
    res = []
    for point in data:
        smoothed_val = last * weight + (1-weight) * point
        res.append(smoothed_val)
        last = smoothed_val
    return res

if __name__ == '__main__':
    font1 = {'family': 'Times New Roman',
             'color': 'black',
             'weight': 'normal',
             'size': 16,
             }
    #rew_gcn = np.load('reward_cen_gcn.npy')
    reward_ori = np.load('reward_new.npy')
    #reward_one = np.load('reward_one_hot.npy')

    #rew_gcn = smooth(rew_gcn)
    reward_ori = smooth(reward_ori)
    #reward_one = smooth(reward_one)

    x = np.arange(len(reward_ori))
    #plt.xticks(x)

    #plt.plot(x, rew_gcn, color='g', label='GCN')
    plt.plot(x, reward_ori, color='r', label='ORI')
    #plt.plot(x, reward_one, color='b', label='ONE')
    plt.xlabel('Training Epoch', fontdict=font1)
    plt.ylabel('Reward', fontdict=font1)
    plt.grid()

    plt.grid(True)
    plt.grid(color='grey', linestyle='--')  # 修改网r't格颜色，类型为虚线

    plt.legend()
    plt.show()






