#! /usr/bin/python

import sys
import numpy as np
import cPickle as pickle
import matplotlib
from matplotlib import pyplot as plt

from visualize_distortion import GPModel



def main():
    model_folder = sys.argv[1]

    matplotlib.rcParams.update({'font.size': 14})
    plt.figure(figsize=(12,3))

    for num, model_folder in enumerate(sys.argv[1:4]):
        with open(model_folder+'/pose0.uv') as f:
            obj = pickle.load(f)
            det_i = obj[:,:2]
            undistortion = obj[:,2:]

        with open(model_folder+'/pose0.gp') as f:
            gpmodel = pickle.load(f)


        residual = gpmodel.predict(det_i) - undistortion
        plt.subplot(1, 3, num+1)
        plt.title('%d mm' % int(model_folder.split('/')[-1]))
        plt.grid(b=True, which='major', color='#ededed', linestyle='-')
        plt.plot(residual[:,0], residual[:,1], 'o',
            markersize=3, markerfacecolor='#CC2529', markeredgecolor='#922428')
        S = .75
        plt.yticks([0., S])
        plt.xticks([-S, 0., S])
        plt.xlim([-S, S])
        plt.ylim([-S, S])
        if num != 0:
            plt.setp( plt.gca().get_yticklabels(), visible=False)
        plt.gca().set_axisbelow(True)

    plt.show()


if __name__ == '__main__':
    main()