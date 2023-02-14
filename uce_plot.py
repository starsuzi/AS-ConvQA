import matplotlib
from matplotlib import pyplot as plt
import numpy as np


def plot_conf(acc, conf):
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.25))
    ax.plot([0,1], [0,1], 'k--')
    ax.plot(conf.data.cpu().numpy(), acc.data.cpu().numpy(), marker='.')
    ax.set_xlabel(r'confidence')
    ax.set_ylabel(r'accuracy')
    ax.set_xticks((np.arange(0, 1.1, step=0.2)))
    ax.set_yticks((np.arange(0, 1.1, step=0.2)))

    return fig, ax

def plot_uncert(err, entr):
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.25))
    
    ax.plot([0,1], [0,1], 'k--') 
    ax.plot(entr.data.cpu().numpy(), err.data.cpu().numpy(), marker='.')
    ax.set_xticks((np.arange(0, 1.1, step=0.2)))
    ax.set_ylabel(r'error')
    ax.set_xlabel(r'uncertainty')
    ax.set_xticks((np.arange(0, 1.1, step=0.2)))
    ax.set_yticks((np.arange(0, 1.1, step=0.2)))

    return fig, ax

def plot_save_conf(args, ece, acc, conf, title, save_pth):
    fig, ax = plot_conf(acc, conf)
    textstr = r'ECE = {:.2f}'.format(ece.item()*100)
    props = dict(boxstyle='round', facecolor='white', alpha=0.75)
    ax.text(0.075, 0.925, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=props
            )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_pth+'.pdf')
    fig.savefig(save_pth+'.png')

def plot_save_entr(args, uce, err, entr, title, save_pth):
    fig, ax = plot_uncert(err, entr)
    textstr = r'UCE = {:.2f}'.format(uce.item()*100)
    props = dict(boxstyle='round', facecolor='white', alpha=0.75)
    ax.text(0.075, 0.925, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=props
            )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_pth+'.pdf')
    fig.savefig(save_pth+'.png')