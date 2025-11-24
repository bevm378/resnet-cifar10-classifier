import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

####################################################
# Basic neural net utilities
####################################################

def display_num_param(net):
	nb_param = 0
	for param in net.parameters():
	    nb_param += param.numel()
	print('There are {} ({:.2f} million) parameters in this neural network'.format(
		nb_param, nb_param/1e6)
	     )

def get_error( scores , labels ):
    bs=scores.size(0)
    predicted_labels = scores.argmax(dim=1)
    indicator = (predicted_labels == labels)
    num_matches=indicator.sum()
    return 1-num_matches.float()/bs    

####################################################
# display images and templates
####################################################

def show(X):
    if X.dim() == 3 and X.size(0) == 3:
        plt.imshow( np.transpose(  X.numpy() , (1, 2, 0))  )
        plt.show()
    elif X.dim() == 2:
        plt.imshow(   X.numpy() , cmap='gray'  )
        plt.show()
    else:
        print('WRONG TENSOR SIZE')


def show_template(template):
    vm=0.3  
    height=4
    width=6  
    f, ax1 = plt.subplots()
    temp=template.detach().numpy()
    cmap_templates = sns.diverging_palette(262, 10, sep=1, n=16, s=99, l=50, 
                             center="dark", as_cmap=True)
    sns.heatmap(temp, cmap=cmap_templates, ax=ax1, cbar=True,square=True,
        vmin=-vm, vmax=vm )  
    plt.show()


####################################################
# show_prop functions for cifar, mnist, fashion-mnist
####################################################

def show_prob_cifar(p):

    p=p.data.squeeze().numpy()

    ft=15
    label = ('airplane', 'automobile', 'Bird', 'Cat', 'Deer', 
             'Dog', 'Frog', 'Horse', 'Ship','Truck' )
    y_pos = np.arange(len(p))*1.2
    target=2
    width=0.9
    col= 'blue'
  
    plt.rcdefaults()
    fig, ax = plt.subplots()

    # the plot
    ax.barh(y_pos, p, width , align='center', color=col)
    ax.set_xlim([0, 1.3])

    # y label
    ax.set_yticks(y_pos)
    ax.set_yticklabels(label, fontsize=ft)
    ax.invert_yaxis()  

    # x label
    ax.set_xticklabels([])
    ax.set_xticks([])
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(4)

    for i in range(len(p)):
        str_nb="{0:.2f}".format(p[i])
        ax.text( p[i] + 0.05 , y_pos[i] ,str_nb ,
                 horizontalalignment='left', verticalalignment='center',
                 transform=ax.transData, color= col,fontsize=ft)

    plt.show()



def show_prob_mnist(p):

    p=p.data.squeeze().numpy()

    ft=15
    label = ('zero', 'one', 'two', 'three', 'four', 'five',
             'six', 'seven', 'eight','nine')
    y_pos = np.arange(len(p))*1.2
    target=2
    width=0.9
    col= 'blue'
   
    plt.rcdefaults()
    fig, ax = plt.subplots()

    # the plot
    ax.barh(y_pos, p, width , align='center', color=col)
    ax.set_xlim([0, 1.3])

    # y label
    ax.set_yticks(y_pos)
    ax.set_yticklabels(label, fontsize=ft)
    ax.invert_yaxis()  

    # x label
    ax.set_xticklabels([])
    ax.set_xticks([])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(4)

    for i in range(len(p)):
        str_nb="{0:.2f}".format(p[i])
        ax.text( p[i] + 0.05 , y_pos[i] ,str_nb ,
                 horizontalalignment='left', verticalalignment='center',
                 transform=ax.transData, color= col,fontsize=ft)

    plt.show()
    


def show_prob_fashion_mnist(p):

    p=p.data.squeeze().numpy()

    ft=15
    label = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 
             'Sandal', 'Shirt', 'Sneaker', 'Bag','Boot')
    y_pos = np.arange(len(p))*1.2
    target=2
    width=0.9
    col= 'blue'

    plt.rcdefaults()
    fig, ax = plt.subplots()

    # the plot
    ax.barh(y_pos, p, width , align='center', color=col)
    ax.set_xlim([0, 1.3])

    # y label
    ax.set_yticks(y_pos)
    ax.set_yticklabels(label, fontsize=ft)
    ax.invert_yaxis()  
    
    # x label
    ax.set_xticklabels([])
    ax.set_xticks([])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(4)

    for i in range(len(p)):
        str_nb="{0:.2f}".format(p[i])
        ax.text( p[i] + 0.05 , y_pos[i] ,str_nb ,
                 horizontalalignment='left', verticalalignment='center',
                 transform=ax.transData, color= col,fontsize=ft)

    plt.show()
    
