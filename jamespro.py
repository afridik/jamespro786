{

 "cells": [

  {

   "cell_type": "code",

   "execution_count": 1,

   "metadata": {},

   "outputs": [],

   "source": [

    "import numpy as np\n",

    "from scipy.sparse import linalg\n",

    "from scipy import sparse\n",

    "from matplotlib import pyplot as plt\n",

    "import matplotlib.image as img"

   ]

  },

  {

   "cell_type": "markdown",

   "metadata": {},

   "source": [

    "## Aufgabe 1"

   ]

  },

  {

   "cell_type": "code",

   "execution_count": 2,

   "metadata": {},

   "outputs": [],

   "source": [

    "def laplacian(N,M):\n",

    "    IM = sparse.eye(M)\n",

    "    IN = sparse.eye(N)\n",

    "    DN = -2*sparse.eye(N) + sparse.eye(N,N,-1) + sparse.eye(N,N,1)\n",

    "    DM = -2*sparse.eye(M) + sparse.eye(M,M,-1) + sparse.eye(M,M,1)\n",

    "    return sparse.kron(IM,DN) + sparse.kron(DM,IN)"

   ]

  },

  {

   "cell_type": "code",

   "execution_count": 3,

   "metadata": {},

   "outputs": [

    {

     "data": {

      "image/png": "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\n",

      "text/plain": [

       "<Figure size 432x288 with 2 Axes>"

      ]

     },

     "metadata": {

      "needs_background": "light"

     },

     "output_type": "display_data"

    }

   ],

   "source": [

    "N = 4\n",

    "M = 3\n",

    "plt.imshow(laplacian(N,M).toarray())\n",

    "plt.title('Laplacian: M=%d, N=%d' % (M, N),loc='center')\n",

    "cax = plt.axes([0.8, 0.125, 0.075, 0.3])         # colorbar [left, bottom, width, height]\n",

    "plt.colorbar(cax=cax)\n",

    "plt.show()"

   ]

  },

  {

   "cell_type": "code",

   "execution_count": 4,

   "metadata": {},

   "outputs": [],

   "source": [

    "def clone_gray2(g,f):\n",

    "    N = len(g)\n",

    "    M = len(g[0])\n",

    "    # contruct diagonalmatrix whose diagonal elements are zero or one corresponding to the boundery elements of g\n",

    "    diagonal = np.zeros(N*M)\n",

    "    fflat = f.flatten('F')\n",

    "    for i in range(N*M):\n",

    "        diagonal[i] = fflat[i]\n",

    "    for i in range(N+1,N*M - N - 1):\n",

    "        diagonal[i] = 0\n",

    "    for i in range(0,M):\n",

    "        diagonal[i*N-1] = fflat[i]\n",

    "        diagonal[i*N] = fflat[i]\n",

    "    gflat = g.flatten('F')\n",

    "    laplacemat = laplacian(N,M)\n",

    "    b = laplacemat @ gflat\n",

    "    b -= (laplacemat  @ fflat) #substract boundery values\n",

    "    return linalg.cg(laplacemat,b)"

   ]

  },

  {

   "cell_type": "code",

   "execution_count": 5,

   "metadata": {},

   "outputs": [],

   "source": [

    "original = img.imread('C:/Users/Tilma/Uni-Stuff/NuMa/PA2/water.jpg').astype(np.int32)\n",

    "pic = img.imread('C:/Users/Tilma/Uni-Stuff/NuMa/PA2/bear.jpg').astype(np.int32)\n",

    "newpic = np.zeros(shape=(50,50,3))\n",

    "for i in range(3):\n",

    "    f = original[50:100,50:100,i]\n",

    "    g = pic[50:100,50:100,i] \n",

    "    newpic[:,:,i] = clone_gray2(g,f)[0].reshape(50,50)\n",

    "    #newpic[:,:,i] = (f - clone_gray(g,f)[0].reshape(50,50))"

   ]

  },

  {

   "cell_type": "code",

   "execution_count": 6,

   "metadata": {},

   "outputs": [

    {

     "ename": "NameError",

     "evalue": "name '__file__' is not defined",

     "output_type": "error",

     "traceback": [

      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",

      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",

      "\u001b[1;32m<ipython-input-6-db889a1b62ed>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[0mfig\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m(\u001b[0m\u001b[0max1\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0max2\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0max3\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msubplots\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m3\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      2\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 3\u001b[1;33m \u001b[0mdir_path\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mos\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mpath\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdirname\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mos\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mpath\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mrealpath\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0m__file__\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      4\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[0mimg1\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mimg\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mimread\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'/bear.jpg'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mastype\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mint32\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",

      "\u001b[1;31mNameError\u001b[0m: name '__file__' is not defined"

     ]

    },

    {

     "data": {

      "image/png": "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\n",

      "text/plain": [

       "<Figure size 432x288 with 3 Axes>"

      ]

     },

     "metadata": {

      "needs_background": "light"

     },

     "output_type": "display_data"

    }

   ],

   "source": [

    "fig, (ax1, ax2, ax3) = plt.subplots(1,3)\n",

    "\n",

    "\n",

    "img1 = img.imread('bear.jpg').astype(np.int32)\n",

    "img2 = img.imread('water.jpg').astype(np.int32)\n",

    "\n",

    "ax1.imshow(img1[:,:,2],cmap='Blues') \n",

    "ax2.imshow(img2[:,:,2],cmap='Blues')\n",

    "ax3.imshow(newpic)\n",

    "plt.tight_layout()\n",

    "plt.show()"

   ]

  }

 ],

 "metadata": {

  "kernelspec": {

   "display_name": "Python 3",

   "language": "python",

   "name": "python3"

  },

  "language_info": {

   "codemirror_mode": {

    "name": "ipython",

    "version": 3

   },

   "file_extension": ".py",

   "mimetype": "text/x-python",

   "name": "python",

   "nbconvert_exporter": "python",

   "pygments_lexer": "ipython3",

   "version": "3.7.4"

  }

 },

 "nbformat": 4,

 "nbformat_minor": 4

}

