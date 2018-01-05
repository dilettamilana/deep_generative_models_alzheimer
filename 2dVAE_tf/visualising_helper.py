
import cv2
import sys
import math
from math import floor
from sklearn import manifold
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import random
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


class Bin():
    def __init__(self, new_data_point):
        self.xlist=[]
        self.ylist=[]
        self.add_data_point(new_data_point)

    def add_data_point(self, new_data_point):
        self.xlist.append(new_data_point[0])
        self.ylist.append(new_data_point[1])

class Bin3D():
    def __init__(self, new_data_point):
        self.xlist=[]
        self.ylist=[]
        self.zlist=[]
        self.add_data_point(new_data_point)

    def add_data_point(self, new_data_point):
        self.xlist.append(new_data_point[0])
        self.ylist.append(new_data_point[1])
        self.zlist.append(new_data_point[2])

def shuffle(images, labels):
    n_samples = images.shape[0]
    perm = np.arange(n_samples)
    np.random.shuffle(perm)
    return images[perm], labels[perm]

class Visualiser():
    def __init__(self, vae, _3d, n_z, size_batch, save_dir, img_x, img_y, img_z, x_train, x_test, y_train, y_test):

        #(self, self.n_z, self.size_batch, self.save_dir, self.img_x, self.img_y, self.img_z, self.x_train, self.x_test, self.y_train, self.y_test)

        self.vae = vae
        self._3d=_3d
        self.n_z= n_z
        self.size_batch= size_batch
        self.save_dir=save_dir
        self.img_x=img_x
        self.img_y=img_y
        self.img_z=img_z
        self.x_train=x_train
        self.x_test=x_test
        self.y_train=y_train
        self.y_test=y_test

    def mmse_scatter(self, x, y, labels, imageData, zoom, variable_for_scatter):
        title="mmse"

        print("mmse | x: "+ str(x.shape)+ " y: "+str(y.shape))

        min=np.amin(labels[:, variable_for_scatter])
        max = np.amax(labels[:, variable_for_scatter])
        min_bin=int(5*floor(min/5.)) #for lower ages we have less patients: bin of size 10
        max_bin=int(1*floor(max/1.)) #for higher ages we have more patients: bin of size 5

        bin_boundaries = []
        for i in range(min_bin, max_bin+1):
            if (i<=25 and i%5==0):
                bin_boundaries.append(i)
            elif (i>=25 and i%1==0):
                bin_boundaries.append(i)
        print(bin_boundaries)
        total_boundaries= len(bin_boundaries)+1
        bins=[]
        bins=[None]*total_boundaries

        for i in range(len(x)):
            x0, y0 = x[i], y[i]

            label = labels[i, variable_for_scatter]
            corresponding_bin= np.digitize(label, bin_boundaries)
            #print("label "+ str(label)+ " is just greater than "+ str(bin_boundaries[corresponding_bin-1]))
            #print("corresponding_bin ",corresponding_bin)
            if (bins[corresponding_bin]== None):
                bins[corresponding_bin]=Bin([x0,y0])
            else:
                bins[corresponding_bin].add_data_point([x0,y0])
            #print("added: ", bins[corresponding_bin])

        colors = cm.rainbow(np.linspace(0, 1, total_boundaries))
        fig, ax = plt.subplots()
        for i in range(total_boundaries):
            color=colors[i]
            bin=bins[i]
            if (bin!= None): #there might be no one corresponding to that bin in the current dataset
                ax.scatter(bin.xlist, bin.ylist, color=color)


        plt.title(title+" scatter plot")
        sm = plt.cm.ScalarMappable(cmap="rainbow", norm=plt.Normalize(vmin=min_bin, vmax=max_bin))
        sm._A = [] # fake up the array of the scalar mappable
        plt.colorbar(sm)

        plt.savefig(self.directory + "scatter_" + str(title) + ".png")
        #plt.show()
        plt.close()

    def age_scatter(self, x, y, labels, imageData, zoom, variable_for_scatter):
        title="age"
        print("age | x: "+ str(x.shape)+ " y: "+str(y.shape))

        min=np.amin(labels[:, variable_for_scatter])
        max = np.amax(labels[:, variable_for_scatter])
        min_bin=int(10*floor(min/10.)) #for lower ages we have less patients: bin of size 10
        max_bin=int(5*floor(max/5.)) #for higher ages we have more patients: bin of size 5

        bin_boundaries = []
        for i in range(min_bin, max_bin+1):
            if (i<=50 and i%10==0):
                bin_boundaries.append(i)
            elif (i>=50 and i%5==0):
                bin_boundaries.append(i)
        #print(bin_boundaries)
        total_boundaries= len(bin_boundaries)+1
        bins=[]
        bins=[None]*total_boundaries
        print(len(x))
        for i in range(len(x)):
            x0, y0 = x[i], y[i]
            # Convert to image
            img = imageData[i]

            #img = np.reshape(imageData[i], newshape=[self.img_x, self.img_y, self.img_z])
            #image = OffsetImage(img, zoom=zoom)
            label = labels[i, variable_for_scatter]
            corresponding_bin= np.digitize(label, bin_boundaries)
            #print("label "+ str(label)+ " is just greater than "+ str(bin_boundaries[corresponding_bin-1]))
            #print("corresponding_bin ",corresponding_bin)
            if (bins[corresponding_bin]== None):
                bins[corresponding_bin]=Bin([x0,y0])
            else:
                bins[corresponding_bin].add_data_point([x0,y0])
            #print("added: ", bins[corresponding_bin])

        colors = cm.rainbow(np.linspace(0, 1, total_boundaries))
        fig, ax = plt.subplots()
        for i in range(total_boundaries):
            color=colors[i]
            bin=bins[i]
            if (bin!= None): #there might be no one corresponding to that bin in the current dataset
                ax.scatter(bin.xlist, bin.ylist, color=color)

        plt.title(title + " scatter plot")
        sm = plt.cm.ScalarMappable(cmap="rainbow", norm=plt.Normalize(vmin=min_bin, vmax=max_bin))
        sm._A = [] # fake up the array of the scalar mappable
        plt.colorbar(sm)

        plt.savefig(self.directory + "scatter_" + str(title) + ".png")
        #plt.show()
        plt.close()

    def sex_scatter(self, x, y, labels, imageData, zoom, variable_for_scatter):
        title="sex"
        print("age | x: "+ str(x.shape)+ " y: "+str(y.shape))
        count_g0 = 0
        count_g1 = 0

        g0_samples_x=[]
        g0_samples_y=[]
        g1_samples_x=[]
        g1_samples_y=[]

        for i in range(len(x)):
            x0, y0 = x[i], y[i]
            label = labels[i, variable_for_scatter]

            if (label == 0):
                count_g0 += 1
                g0_samples_x.append(x0)
                g0_samples_y.append(y0)
            else: #  (label == 1):
                count_g1 += 1
                g1_samples_x.append(x0)
                g1_samples_y.append(y0)

        print("g0_samples: "+str(count_g0) + " and in fact " + str(len(g0_samples_x)))
        #for i in range(len(g0_samples_x)):
            #print(str(g0_samples_x[i]) + "," + str(g0_samples_y[i]))

        print("g1_samples: "+str(count_g1) + " and in fact " + str(len(g1_samples_x)))
        #for i in range(len(g1_samples_x)):
            #print(str(g1_samples_x[i]) + "," + str(g1_samples_y[i]))

        g0_plot, = plt.plot(g0_samples_x, g0_samples_y, 'ro', label="G0")
        g1_plot, = plt.plot(g1_samples_x, g1_samples_y, 'bo', label="G1")
        plt.legend(handles=[g0_plot, g1_plot])
        plt.title(title + " scatter plot")
        plt.savefig(self.directory + "scatter_" + str(title) + ".png")
        plt.close()

    def aibl_oasis_harp_scatter(self, x, y, labels, imageData, zoom, variable_for_scatter):
        title="aibl_oasis_harp"

        print("aibl oasis harp | x: "+ str(x.shape)+ " y: "+str(y.shape))
        count_aibl = 0
        count_oasis = 0
        count_harp = 0

        aibl_samples_x=[]
        aibl_samples_y=[]
        oasis_samples_x=[]
        oasis_samples_y=[]
        harp_samples_x=[]
        harp_samples_y=[]

        for i in range(len(x)):
            x0, y0 = x[i], y[i]
            label = labels[i, variable_for_scatter]

            if (label == 0):  # AIBL
                count_aibl += 1
                aibl_samples_x.append(x0)
                aibl_samples_y.append(y0)
            elif (label == 1):  # oasis
                count_oasis += 1
                oasis_samples_x.append(x0)
                oasis_samples_y.append(y0)
            else:  # harp
                count_harp += 1
                harp_samples_x.append(x0)
                harp_samples_y.append(y0)

        print("aibl samples")
        print(str(count_aibl) + " and in fact " + str(len(aibl_samples_x)))
        #for i in range(len(aibl_samples_x)):
            #print(str(aibl_samples_x[i]) + "," + str(aibl_samples_y[i]))

        print("oasis samples")
        print(str(count_oasis) + " and in fact " + str(len(oasis_samples_x)))
        #for i in range(len(oasis_samples_x)):
            #print(str(oasis_samples_x[i]) + "," + str(oasis_samples_y[i]))

        print("harp samples")
        print(str(count_harp) + " and in fact " + str(len(harp_samples_x)))
        #for i in range(len(harp_samples_x)):
            #print(str(harp_samples_x[i]) + "," + str(harp_samples_y[i]))

        aibl_plot, = plt.plot(aibl_samples_x, aibl_samples_y, 'ro', label="AIBL")
        oasis_plot, = plt.plot(oasis_samples_x, oasis_samples_y, 'bo', label="OASIS")
        harp_plot, = plt.plot(harp_samples_x, harp_samples_y, 'go', label="HARP")
        plt.legend(handles=[aibl_plot, oasis_plot, harp_plot])
        plt.title(title + " scatter plot")
        plt.savefig(self.directory + "scatter_" + str(title) + ".png")
        plt.close()

    def ad_nc_mci_scatter(self, x, y, labels, imageData, zoom, variable_for_scatter):
        #title=sys._getframe().f_code.co_name #never tested
        title="ad_nc_mci"
        label_dict = {0: "NC", 1: "MCI", 2: "AD"}
        count_ad = 0
        count_mci = 0
        count_nc = 0
        ad_samples_x = []
        ad_samples_y = []
        mci_samples_x = []
        mci_samples_y = []
        nc_samples_x = []
        nc_samples_y = []

        for i in range(len(x)):
            x0, y0 = x[i], y[i]
            label=label_dict.get(np.asscalar(labels[i, variable_for_scatter]))

            if (label=="NC"):
                count_nc+=1
                nc_samples_x.append(x0)
                nc_samples_y.append(y0)
            elif (label=="MCI"):
                count_mci+=1
                mci_samples_x.append(x0)
                mci_samples_y.append(y0)
            else:
                count_ad+=1
                ad_samples_x.append(x0)
                ad_samples_y.append(y0)

        print("nc_samples: "+str(count_nc) + " and in fact " + str(len(nc_samples_x)))

        print("mci_samples: "+str(count_mci) + " and in fact " + str(len(mci_samples_x)))

        print("ad_samples: "+ str(count_ad) + " and in fact " + str(len(ad_samples_x)))

        print("LALA ", len(nc_samples_y))
        print("LALA ", len(mci_samples_y))
        print("LALA ", len(ad_samples_y))
        print(nc_samples_x)
        print(mci_samples_x)
        print(ad_samples_x)

        print(nc_samples_y)
        print(mci_samples_y)
        print(ad_samples_y)

        nc_plot, = plt.plot(nc_samples_x, nc_samples_y, 'ro', label="NC")
        mci_plot, = plt.plot(mci_samples_x, mci_samples_y, 'bo', label="MCI")
        ad_plot, = plt.plot(ad_samples_x, ad_samples_y, 'go', label="AD")
        plt.legend(handles=[nc_plot, mci_plot, ad_plot])
        plt.title(title + " scatter plot")
        plt.savefig(self.directory + "scatter_" + str(title) + ".png")
        plt.close()

        ad_plus_mci_x = ad_samples_x + mci_samples_x
        ad_plus_mci_y = ad_samples_y + mci_samples_y
        nc_plot, = plt.plot(nc_samples_x, nc_samples_y, 'ro', label="NC")
        ad_mci_plot, = plt.plot(ad_plus_mci_x, ad_plus_mci_y, 'bo', label="MCI+AD")
        plt.legend(handles=[nc_plot, ad_mci_plot])
        plt.title("AD+MCI vs NC scatter plot")
        plt.savefig(self.directory + "scatter_admci_nc.png")
        plt.close()

        nc_plus_mci_x=nc_samples_x + mci_samples_x
        nc_plus_mci_y=nc_samples_y + mci_samples_y
        ad_plot,=plt.plot(ad_samples_x, ad_samples_y, 'go', label="AD")
        nc_plus_mci_plot,=plt.plot(nc_plus_mci_x, nc_plus_mci_y, 'ro', label="NC+MCI")
        plt.legend (handles=[ad_plot, nc_plus_mci_plot])
        plt.title("AD vs NC+MCI scatter plot")
        plt.savefig(self.directory+"scatter_ad_ncmci.png")
        plt.close()

    def mmse_scatter_3d(self, x, y, z, labels, imageData, zoom, variable_for_scatter):
        title="mmse"

        min=np.amin(labels[:, variable_for_scatter])
        max = np.amax(labels[:, variable_for_scatter])
        min_bin=int(5*floor(min/5.)) #for lower ages we have less patients: bin of size 10
        max_bin=int(1*floor(max/1.)) #for higher ages we have more patients: bin of size 5

        bin_boundaries = []
        for i in range(min_bin, max_bin+1):
            if (i<=25 and i%5==0):
                bin_boundaries.append(i)
            elif (i>=25 and i%1==0):
                bin_boundaries.append(i)
        print(bin_boundaries)
        total_boundaries= len(bin_boundaries)+1
        bins=[]
        bins=[None]*total_boundaries

        for i in range(len(x)):
            x0, y0 = x[i], y[i]

            label = labels[i, variable_for_scatter]
            corresponding_bin= np.digitize(label, bin_boundaries)
            #print("label "+ str(label)+ " is just greater than "+ str(bin_boundaries[corresponding_bin-1]))
            #print("corresponding_bin ",corresponding_bin)
            if (bins[corresponding_bin]== None):
                bins[corresponding_bin]=Bin([x0,y0])
            else:
                bins[corresponding_bin].add_data_point([x0,y0])
            #print("added: ", bins[corresponding_bin])

        colors = cm.rainbow(np.linspace(0, 1, total_boundaries))
        fig, ax = plt.subplots()
        for i in range(total_boundaries):
            color=colors[i]
            bin=bins[i]
            if (bin!= None): #there might be no one corresponding to that bin in the current dataset
                ax.scatter(bin.xlist, bin.ylist, color=color)

        plt.title(title+" scatter plot")
        sm = plt.cm.ScalarMappable(cmap="rainbow", norm=plt.Normalize(vmin=min_bin, vmax=max_bin))
        sm._A = [] # fake up the array of the scalar mappable
        plt.colorbar(sm)

        plt.savefig(self.directory + "scatter_" + str(title) + ".png")
        #plt.show()
        plt.close()

    def age_scatter_3d(self, x, y, z, labels, imageData, zoom, variable_for_scatter):

        title="age"

        min=np.amin(labels[:, variable_for_scatter])
        max = np.amax(labels[:, variable_for_scatter])
        min_bin=int(10*floor(min/10.)) #for lower ages we have less patients: bin of size 10
        max_bin=int(5*floor(max/5.)) #for higher ages we have more patients: bin of size 5

        bin_boundaries = []
        for i in range(min_bin, max_bin+1):
            if (i<=50 and i%10==0):
                bin_boundaries.append(i)
            elif (i>=50 and i%5==0):
                bin_boundaries.append(i)
        #print(bin_boundaries)
        total_boundaries= len(bin_boundaries)+1
        bins=[]
        bins=[None]*total_boundaries

        for i in range(len(x)):
            x0, y0, z0 = x[i], y[i], z[i]

            label = labels[i, variable_for_scatter]
            corresponding_bin= np.digitize(label, bin_boundaries)
            #print("label "+ str(label)+ " is just greater than "+ str(bin_boundaries[corresponding_bin-1]))
            #print("corresponding_bin ",corresponding_bin)
            if (bins[corresponding_bin]== None):
                bins[corresponding_bin]=Bin3D([x0,y0, z0])
            else:
                bins[corresponding_bin].add_data_point([x0,y0, z0])
            #print("added: ", bins[corresponding_bin])

        colors = cm.rainbow(np.linspace(0, 1, total_boundaries))

        fig = plt.figure(figsize=[40, 20])
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        for i in range(total_boundaries):
            color=colors[i]
            bin=bins[i]
            if (bin!= None): #there might be no one corresponding to that bin in the current dataset
                ax.scatter(bin.xlist, bin.ylist, bin.zlist,s=1000, color=color)

        ax.set_title(title + " scatter plot")
        sm = plt.cm.ScalarMappable(cmap="rainbow", norm=plt.Normalize(vmin=min_bin, vmax=max_bin))
        sm._A = [] # fake up the array of the scalar mappable
        plt.colorbar(sm)

        plt.savefig(self.directory + "scatter_" + str(title) + "_3d.png")
        #plt.show()
        plt.close()

    def sex_scatter_3d(self, x, y, z, labels, imageData, zoom, variable_for_scatter):
        title="sex"
        count_g0 = 0
        count_g1 = 0

        g0_samples_x=[]
        g0_samples_y=[]
        g1_samples_x=[]
        g1_samples_y=[]

        for i in range(len(x)):
            x0, y0 = x[i], y[i]
            label = labels[i, variable_for_scatter]

            if (label == 0):
                count_g0 += 1
                g0_samples_x.append(x0)
                g0_samples_y.append(y0)
            else: #  (label == 1):
                count_g1 += 1
                g1_samples_x.append(x0)
                g1_samples_y.append(y0)

        print("g0_samples: "+str(count_g0) + " and in fact " + str(len(g0_samples_x)))
        #for i in range(len(g0_samples_x)):
            #print(str(g0_samples_x[i]) + "," + str(g0_samples_y[i]))

        print("g1_samples: "+str(count_g1) + " and in fact " + str(len(g1_samples_x)))
        #for i in range(len(g1_samples_x)):
            #print(str(g1_samples_x[i]) + "," + str(g1_samples_y[i]))

        g0_plot, = plt.plot(g0_samples_x, g0_samples_y, 'ro', label="G0")
        g1_plot, = plt.plot(g1_samples_x, g1_samples_y, 'bo', label="G1")
        plt.legend(handles=[g0_plot, g1_plot])
        plt.title(title + " scatter plot")
        plt.savefig(self.directory + "scatter_" + str(title) + ".png")
        plt.close()

    def aibl_oasis_harp_scatter_3d(self, x, y, z, labels, imageData, zoom, variable_for_scatter):
        title="aibl_oasis_harp"
        count_aibl = 0
        count_oasis = 0
        count_harp = 0

        aibl_samples_x=[]
        aibl_samples_y=[]
        oasis_samples_x=[]
        oasis_samples_y=[]
        harp_samples_x=[]
        harp_samples_y=[]

        for i in range(len(x)):
            x0, y0 = x[i], y[i]
            label = labels[i, variable_for_scatter]

            if (label == 0):  # AIBL
                count_aibl += 1
                aibl_samples_x.append(x0)
                aibl_samples_y.append(y0)
            elif (label == 1):  # oasis
                count_oasis += 1
                oasis_samples_x.append(x0)
                oasis_samples_y.append(y0)
            else:  # harp
                count_harp += 1
                harp_samples_x.append(x0)
                harp_samples_y.append(y0)

        print("aibl samples")
        print(str(count_aibl) + " and in fact " + str(len(aibl_samples_x)))
        #for i in range(len(aibl_samples_x)):
            #print(str(aibl_samples_x[i]) + "," + str(aibl_samples_y[i]))

        print("oasis samples")
        print(str(count_oasis) + " and in fact " + str(len(oasis_samples_x)))
        #for i in range(len(oasis_samples_x)):
            #print(str(oasis_samples_x[i]) + "," + str(oasis_samples_y[i]))

        print("harp samples")
        print(str(count_harp) + " and in fact " + str(len(harp_samples_x)))
        #for i in range(len(harp_samples_x)):
            #print(str(harp_samples_x[i]) + "," + str(harp_samples_y[i]))

        aibl_plot, = plt.plot(aibl_samples_x, aibl_samples_y, 'ro', label="AIBL")
        oasis_plot, = plt.plot(oasis_samples_x, oasis_samples_y, 'bo', label="OASIS")
        harp_plot, = plt.plot(harp_samples_x, harp_samples_y, 'go', label="HARP")
        plt.legend(handles=[aibl_plot, oasis_plot, harp_plot])
        plt.title(title + " scatter plot")
        plt.savefig(self.directory + "scatter_" + str(title) + ".png")
        plt.close()

    def ad_nc_mci_scatter_3d(self, x, y, z, labels, imageData, zoom, variable_for_scatter):
        #title=sys._getframe().f_code.co_name #never tested
        title="ad_nc_mci"
        label_dict = {0: "NC", 1: "MCI", 2: "AD"}
        count_ad = 0
        count_mci = 0
        count_nc = 0
        ad_samples_x = []
        ad_samples_y = []
        ad_samples_z = []
        mci_samples_x = []
        mci_samples_y = []
        mci_samples_z = []
        nc_samples_x = []
        nc_samples_y = []
        nc_samples_z = []

        for i in range(len(x)):
            x0, y0, z0 = x[i], y[i], z[i]
            label=label_dict.get(np.asscalar(labels[i, variable_for_scatter]))

            if (label=="NC"):
                count_nc+=1
                nc_samples_x.append(x0)
                nc_samples_y.append(y0)
                nc_samples_z.append(z0)
            elif (label=="MCI"):
                count_mci+=1
                mci_samples_x.append(x0)
                mci_samples_y.append(y0)
                mci_samples_z.append(z0)
            else:
                count_ad+=1
                ad_samples_x.append(x0)
                ad_samples_y.append(y0)
                ad_samples_z.append(z0)

        print("nc_samples: "+str(count_nc) + " and in fact " + str(len(nc_samples_x)))

        print("mci_samples: "+str(count_mci) + " and in fact " + str(len(mci_samples_x)))

        print("ad_samples: "+ str(count_ad) + " and in fact " + str(len(ad_samples_x)))

        #################### NC vs MCI vs AS  - 3 views ###################
        s=500
        fig = plt.figure(figsize=[40, 20])
        ax = fig.add_subplot(1, 3, 1, projection='3d')
        ax.scatter(nc_samples_x, nc_samples_y, nc_samples_z, 'ro', s=s, label="NC")
        ax.scatter(mci_samples_x, mci_samples_y, mci_samples_z, 'bo',s=s, label="MCI")
        ax.scatter(ad_samples_x, ad_samples_y, ad_samples_z, 'go',s=s, label="AD")
        ax.legend()
        ax.set_title(title + " scatter plot")

        ax = fig.add_subplot(1, 3, 2, projection='3d')
        ax.scatter(nc_samples_x, nc_samples_y, nc_samples_z, 'ro', s=s, label="NC")
        ax.scatter(mci_samples_x, mci_samples_y, mci_samples_z, 'bo', s=s, label="MCI")
        ax.scatter(ad_samples_x, ad_samples_y, ad_samples_z, 'go', s=s, label="AD")
        ax.view_init(30, 90)

        ax = fig.add_subplot(1, 3, 3, projection='3d')
        ax.scatter(nc_samples_x, nc_samples_y, nc_samples_z, 'ro', s=s, label="NC")
        ax.scatter(mci_samples_x, mci_samples_y, mci_samples_z, 'bo', s=s, label="MCI")
        ax.scatter(ad_samples_x, ad_samples_y, ad_samples_z, 'go', s=s, label="AD")
        ax.view_init(30, 180)

        fig.savefig(self.directory + "scatter_" + str(title) + "_3d.png")
        plt.close()

        #################### NC vs [MCI + AD]  - 3 views ###################

        fig = plt.figure(figsize=[40, 20])

        ad_plus_mci_x = ad_samples_x + mci_samples_x
        ad_plus_mci_y = ad_samples_y + mci_samples_y
        ad_plus_mci_z = ad_samples_z + mci_samples_z

        ax = fig.add_subplot(1, 3, 1, projection='3d')
        ax.scatter(nc_samples_x, nc_samples_y, nc_samples_z, 'ro', s=s, label="NC")
        ax.scatter(ad_plus_mci_x, ad_plus_mci_y, ad_plus_mci_z, 'bo', s=s, label="MCI+AD")
        ax.legend()
        ax.set_title("AD+MCI vs NC scatter plot")

        ax = fig.add_subplot(1, 3, 2, projection='3d')
        ax.scatter(nc_samples_x, nc_samples_y, nc_samples_z, 'ro', s=s, label="NC")
        ax.scatter(ad_plus_mci_x, ad_plus_mci_y, ad_plus_mci_z, 'bo', s=s, label="MCI+AD")
        ax.view_init(30, 90)

        ax = fig.add_subplot(1, 3, 3, projection='3d')
        ax.scatter(nc_samples_x, nc_samples_y, nc_samples_z, 'ro', s=s, label="NC")
        ax.scatter(ad_plus_mci_x, ad_plus_mci_y, ad_plus_mci_z, 'bo', s=s, label="MCI+AD")
        ax.view_init(30, 180)

        fig.savefig(self.directory + "scatter_admci_nc_3d.png")
        plt.close()

        #################### [NC + MCI] + AD  - 3 views ###################
        fig = plt.figure(figsize=[40, 20])

        nc_plus_mci_x=nc_samples_x + mci_samples_x
        nc_plus_mci_y=nc_samples_y + mci_samples_y
        nc_plus_mci_z = nc_samples_z + mci_samples_z

        ax = fig.add_subplot(1, 3, 1, projection='3d')
        ax.scatter(ad_samples_x, ad_samples_y, ad_samples_z, 'go', s=s, label="AD")
        ax.scatter(nc_plus_mci_x, nc_plus_mci_y, nc_plus_mci_z, 'ro', s=s, label="NC+MCI")
        ax.legend()
        ax.set_title("AD vs NC+MCI scatter plot")

        ax = fig.add_subplot(1, 3, 2, projection='3d')
        ax.scatter(ad_samples_x, ad_samples_y, ad_samples_z, 'go', s=s, label="AD")
        ax.scatter(nc_plus_mci_x, nc_plus_mci_y, nc_plus_mci_z, 'ro', s=s, label="NC+MCI")
        ax.view_init(30, 90)

        ax = fig.add_subplot(1, 3, 3, projection='3d')
        ax.scatter(ad_samples_x, ad_samples_y, ad_samples_z, 'go', s=s, label="AD")
        ax.scatter(nc_plus_mci_x, nc_plus_mci_y, nc_plus_mci_z, 'ro', s=s, label="NC+MCI")
        ax.view_init(30, 180)

        fig.savefig(self.directory+"scatter_ad_ncmci_3d.png")
        plt.close()

    # Scatter with images instead of points
    def imscatter2d(self, x, y, labels, ax, imageData, fig, zoom):

        # for every patient:
        # 0     label
        # 1     dataset_code
        # 2     sex
        # 3     age
        # 4     mmse
        # 5     cdr

        #dataset code
        # 0 is AIBL
        # 1 is oasis
        # 2 is harp
        print("pronti per scatter: ")
        print("x shape: ", x.shape)
        print("y shape: ", y.shape)

        self.ad_nc_mci_scatter(x, y, labels, imageData, zoom, variable_for_scatter=0)

        self.aibl_oasis_harp_scatter(x, y, labels, imageData, zoom, variable_for_scatter=1)

        self.sex_scatter(x, y, labels, imageData, zoom, variable_for_scatter=2)

        self.age_scatter(x, y, labels, imageData, zoom, variable_for_scatter=3)

        self.mmse_scatter(x, y, labels, imageData, zoom, variable_for_scatter=4)

    def imscatter3d(self, x, y, z, labels, imageData, zoom):
        # for every patient:
        # 0     label
        # 1     dataset_code
        # 2     sex
        # 3     age
        # 4     mmse
        # 5     cdr

        # dataset code
        # 0 is AIBL
        # 1 is oasis
        # 2 is harp

        self.ad_nc_mci_scatter_3d(x, y, z, labels, imageData, zoom, variable_for_scatter=0)

        #not tested yet
        #self.aibl_oasis_3d_harp_scatter(x, y, z, labels, imageData, zoom, variable_for_scatter=1)

        #not tested yet
        #self.sex_3d_scatter(x, y, z, labels, imageData, zoom, variable_for_scatter=2)

        self.age_scatter_3d(x, y, z, labels, imageData, zoom, variable_for_scatter=3)

        #not tested yet
        #self.mmse_3d_scatter(x, y, z, labels, imageData, zoom, variable_for_scatter=4)

    # Show dataset images with T-sne projection of latent space encoding
    #dataset comes as a parameter, so he can choose either train or test datasets
    def ScatterOfLatentSpace(self, dataset_images, dataset_encoded_images, dataset_labels, display=True, n_components=2):

        X=dataset_images
        labels=dataset_labels

        labels=dataset_labels
        print("labels shape ", labels.shape)

        # Compute latent space representation
        print("Computing latent space projection...")
        X_encoded = dataset_encoded_images
        print("X_encoded shape: ", X_encoded.shape)

        # Compute t-SNE embedding of latent space
        print("Computing t-SNE embedding...")
        tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
        X_tsne = tsne.fit_transform(X_encoded)

        print("\n\n\n shape of X_tsne: ", X_tsne.shape)

        # Plot images according to t-sne embedding
        if display:
            print("Plotting t-SNE visualization...")

            print("before imscatter X and labels shape: "+ str(X.shape) + str(labels.shape))
            if (n_components==2):
                fig, ax = plt.subplots()
                self.imscatter2d(X_tsne[:, 0], X_tsne[:, 1], labels, imageData=X, ax=ax, fig=fig, zoom=0.3)
            else:
                #fig = plt.figure(figsize=[20, 20])
                #ax = fig.add_subplot(111, projection='3d')
                self.imscatter3d(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], labels, imageData=X, zoom=0.3)
        else:
            return X_tsne

        # Scatter with images instead of points

    def realimagescatter(self, x, y, labels, imageData,ax, fig, zoom):
        print("real image scatter | x: "+str(x.shape)+" y: "+str(y.shape))
        images = []
        for i in range(len(x)):
            x0, y0 = x[i], y[i]
            img = np.reshape(imageData[i], newshape=[self.img_x, self.img_y])
            image = OffsetImage(img, zoom=zoom)
            ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)

            images.append(ax.add_artist(ab))

        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale()

    def realimagescatter_3d(self, x, y, z, labels, imageData,ax, fig, zoom):

        images = []
        for i in range(len(x)):
            x0, y0, z0 = x[i], y[i], z[i]
            img = np.reshape(imageData[i], newshape=[self.img_x, self.img_y, self.img_z])
            image = OffsetImage(img, zoom=zoom)
            ab = AnnotationBbox(image, (x0, y0, z0), xycoords='data', frameon=False)
            images.append(ax.add_artist(ab))

        ax.update_datalim(np.column_stack([x, y, z]))
        ax.autoscale()

    def computeTSNEProjectionOfLatentSpace(self, dataset_images, dataset_encoded_images, dataset_labels, n_components, _3d, max_images=500, display=True):
        assert (_3d and n_components==3) or ((not _3d) and n_components==2), "check your settings: _3d has to be False if n_components=2 or True if n_components=3."

        #dataset_images, dataset_labels = shuffle(dataset_images, dataset_labels)
        X = dataset_images[0:max_images]
        labels = dataset_labels[0:max_images]

        # Compute latent space representation
        print("Computing latent space projection...")
        X_encoded = dataset_encoded_images[0:max_images]

        # Compute t-SNE embedding of latent space
        print("Computing t-SNE embedding...")
        tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
        X_tsne = tsne.fit_transform(X_encoded)

        # Plot images according to t-sne embedding
        if display:
            print("Plotting t-SNE visualization...")
            fig, ax = plt.subplots(figsize=[20, 20])

            if (_3d):
                ax = fig.gca(projection='3d')
                self.realimagescatter_3d(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:,2], labels, imageData=X, ax=ax, fig=fig, zoom=0.3)
                plt.savefig(self.directory + "TSNE_latentspace_3d.png", cmap='gray')

            else:
                print("x_tsne: ", X_tsne.shape)
                self.realimagescatter(X_tsne[:, 0], X_tsne[:, 1], labels, imageData=X, ax=ax, fig=fig, zoom=0.3)
                plt.savefig(self.directory + "TSNE_latentspace.png", cmap='gray')
            plt.close()

        else:
            return X_tsne

    # Show dataset images with T-sne projection of pixel space
    def computeTSNEProjectionOfPixelSpace(self, dataset_images, dataset_labels, n_components, _3d=False, display=True):
        assert (_3d and n_components==3) or ((not _3d) and n_components==2), "check your settings: _3d has to be False if n_components=2 or True if n_components=3."

        #X=self.x_test[:1000]
        #labels = self.y_test[:1000]
        max_images=50
        dataset_images, dataset_labels =shuffle(dataset_images, dataset_labels)
        X=dataset_images[0:max_images]
        labels=dataset_labels[0:max_images]
        # Compute t-SNE embedding of latent space
        print("Computing t-SNE embedding...")
        tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
        X_tsne = tsne.fit_transform(X.reshape([-1, self.img_x * self.img_y * 1]))

        # Plot images according to t-sne embedding
        if display:
            print("Plotting t-SNE visualization...")
            fig, ax = plt.subplots(figsize=[20,20])
            if (_3d):
                ax = fig.gca(projection='3d')
                self.realimagescatter_3d(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:,2], labels, imageData=X, ax=ax, fig=fig, zoom=0.3)
                plt.savefig(self.directory + "TSNE_pixelspace_3d.png", cmap='gray')
                #plt.show()
            else:
                self.realimagescatter(X_tsne[:, 0], X_tsne[:, 1], labels, imageData=X, ax=ax, fig=fig, zoom=0.3)
                plt.savefig(self.directory + "TSNE_pixelspace.png", cmap='gray')
            plt.close()

        else:
            return X_tsne

    # Reconstructions for samples in dataset
    def getReconstructedImages(self, X, name_addition):

        print("\n\n\n X.shape: ", X.shape[0])
        #nbSamples = X.shape[0]
        nbSamples=X.shape[0]
        nbSquares = int(math.sqrt(nbSamples))
        nbSquaresHeight = 2 * nbSquares
        nbSquaresWidth = nbSquaresHeight
        resultImage = np.zeros((nbSquaresHeight * self.img_x, int(nbSquaresWidth * self.img_x / 2), X.shape[-1]))
        print("result image shape: ", resultImage.shape)
        reconstructedX = self.vae.predict(X, batch_size=self.batch_size)
        iterations= None

        if nbSamples >= 10:
            iterations=10
        else:
            iterations=nbSamples

        print("iterations: "+ str(iterations)+"\nnbSamples: "+str(nbSamples)+"\nnbSquare: "+str(nbSquares)+ "\nnbSquareHeight: "+ str(nbSquaresHeight)+"\nnbSquaresWidth: "+str(nbSquaresWidth))

        count=0
        print("reconstructed image shape: ", reconstructedX.shape)
        for i in range(iterations):
            original = X[i]
            reconstruction = reconstructedX[i]
            print("reconstruction shape", reconstruction.shape)
            rowIndex = i % nbSquaresWidth
            columnIndex = (i - rowIndex) / nbSquaresHeight
            print("row index "+ str(rowIndex)+"column index " + str(columnIndex))
            print("in for, shape: ["+str(int(rowIndex * self.img_x))+":"+str(int((rowIndex + 1) * self.img_x))+","+ str(int(columnIndex * 2 * self.img_x))+":"+str(int((columnIndex + 1) * 2 * self.img_x))+",:]" )
            resultImage[int(rowIndex * self.img_x):((rowIndex + 1) * self.img_x), int(columnIndex * 2 * self.img_x):int((columnIndex + 1) * 2 * self.img_x), :] = np.hstack([original, reconstruction])
            print("********+"+str(count)+"***********")
            count+=1
            figure = np.reshape(reconstructedX[i], newshape=[self.img_x, self.img_y])
            plt.imshow(figure)
            plt.savefig(self.directory + "reconstruct_"+str(name_addition)+"_" + str(i) + ".png", cmap='gray')
            #plt.show()
        return resultImage

    # Reconstructions for samples in dataset
    def visualizeReconstructedImages(self, save=True, label=False):
        X_train = self.x_train[:self.x_test.shape[0]]
        X_test = self.x_test

        trainReconstruction = self.getReconstructedImages(X_train)
        testReconstruction = self.getReconstructedImages(X_test)

        if not save:
            print("Generating 10 image reconstructions...")

        result = np.hstack([trainReconstruction, np.zeros([trainReconstruction.shape[0], 5, trainReconstruction.shape[-1]]),
                            testReconstruction])
        result = (result * 255.).astype(np.uint8)

        if save:
            cv2.imwrite(self.directory + "reconstructions_{}.png".format(label), result)
        else:
            cv2.imshow("Reconstructed images (train - test)", result)
            cv2.waitKey()
            cv2.destroyAllWindows()

    def generated_images_multi_dim(self, display=True):
        n = 4  # figure with 15x15 digits
        digit_size = 256
        figure = np.zeros((digit_size * n, digit_size * n))

        grid=np.empty([n*n, self.latent_dim])
        for a in range(self.latent_dim):
            grid[:,a]= norm.ppf(np.linspace(0.05, 0.95, n*n))

        plane = []

        for i in range(n):
            for j in range(n):
                plane.append([i, j])
        print(plane)

        for step in range(n*n):
                z_sample = np.array(grid[step])
                #print('\n\n\ncurrent z sample :', z_sample.shape)
                z_sample = np.tile(z_sample, self.batch_size).reshape(self.batch_size, self.latent_dim)
                x_decoded = self.generator.predict(z_sample, batch_size=self.batch_size)
                #plt.imshow(x_decoded)

                #tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
                #X_tsne = tsne.fit_transform(z_sample)

                #print("x_decoded shape: ", x_decoded.shape) [19,256,256,1]
                digit = x_decoded[0].reshape(digit_size, digit_size)
                i=plane[step][0]
                j=plane[step][1]
                #print("i: " + str(i) + ", j: " + str(j))
                figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit
                #figure[X_tsne[0][0] * digit_size: (X_tsne[0][0] + 1) * digit_size, X_tsne[0][1] * digit_size: (X_tsne[0][1] + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='gray')
        plt.savefig(self.directory+"reconstruction_1.png", cmap='gray')
        plt.imshow(figure, cmap='copper')
        plt.savefig(self.directory+"reconstruction_2.png", cmap='copper')
        plt.imshow(figure, cmap='gist_gray')
        plt.savefig(self.directory+"reconstruction_3.png", cmap='gist_gray')
        if (display):
            plt.show()
        plt.close()

    def generate_images_2d(self):
        # display a 2D manifold of the digits
        n = 2  # figure with 15x15 digits
        digit_size = self.img_x
        figure = np.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
        # to produce values of the latent variables z, since the prior of the latent space is Gaussian
        # linspace: returns n evenly spaced numbers from 0.05 to 0.95
        # suppongo che norm.ppf li riporti a media 0(sicuro) e varianza 1.

        grid_x = norm.ppf(np.linspace(-100, 100, n))
        grid_y = norm.ppf(np.linspace(-300, 100, n))

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]])
                print('\n\n\ncurrent z sample :', z_sample.shape)
                z_sample = np.tile(z_sample, self.batch_size).reshape(self.batch_size, self.latent_dim)
                x_decoded = self.generator.predict(z_sample, batch_size=self.batch_size)
                #print("x_decoded shape: ", x_decoded.shape) [19,256,256,1]
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='gray')
        plt.savefig(self.directory+"reconstruction_1.png", cmap='gray')
        plt.imshow(figure, cmap='copper')
        plt.savefig(self.directory+"reconstruction_2.png", cmap='copper')
        plt.imshow(figure, cmap='gist_gray')
        plt.savefig(self.directory+"reconstruction_3.png", cmap='gist_gray')
        plt.legend()
        #plt.show()
        plt.close()

    def plot_test_image(self, variationalAutoencoder, tests_num=4):
        img_x=variationalAutoencoder.img_x
        img_y=variationalAutoencoder.img_y

        assert img_x==img_y, "Something's wrong: img_x should be = img_y"
        starting_index=random.choice(range(self.x_test.shape[0]-tests_num))

        assert tests_num <=8, "Errore, tests_num must be between 0 and 8"

        predicted_images_list = []
        original_images_list = []

        for index in range(tests_num):
            original_image = np.reshape(self.x_test[starting_index+index], newshape=[img_x, img_y])
            #print("original image shape: ", original_image.shape)

            if variationalAutoencoder.model == 'basic':
                encoded_image= self.encoder.predict(np.reshape(original_image, newshape=[1, img_x*img_y]))  # result is 1,256,256,1
            else:
                encoded_image= self.encoder.predict(np.reshape(original_image, newshape=[1, img_x, img_y, 1]))  # result is 1,256,256,1

            predicted_image = np.reshape(self.generator.predict(encoded_image), newshape=[img_x, img_y])

            predicted_images_list.append(predicted_image)
            original_images_list.append(original_image)

        predicted_images=np.array(predicted_images_list)
        original_images=np.array(original_images_list)

        plt.figure(figsize=(10, 10))
        figure_count=1
        print("\n\n")
        for i in range(tests_num):
            #print(i)
            plt.subplot(tests_num,2,figure_count)  #devo partire da 1 per plottare
            figure_count+=1
            plt.imshow(original_images[i])

            plt.subplot(tests_num,2,figure_count)
            figure_count+=1
            plt.imshow(predicted_images[i])

        plt.savefig(self.directory + "plot_test_image.png", cmap='gray')
        plt.close()
