import matplotlib.pyplot as plt

def show_image(img, pixelMap):
    plt.figure()
    plt.imshow(img, extent=(pixelMap['X'][0][0].min(), pixelMap['X'][0][0].max(), pixelMap['Z'][0][0].max(), pixelMap['Z'][0][0].min()), origin='upper')
    plt.colorbar()


def show_image_pts(img, pixelMap, pts, index=2):
    plt.figure()
    plt.imshow(img, extent=(pixelMap['X'][0][0].min(), pixelMap['X'][0][0].max(), pixelMap['Z'][0][0].max(), pixelMap['Z'][0][0].min()), origin='upper')
    plt.scatter(pts[:,0], pts[:,index],s=2,c='k',zorder=2)
    plt.colorbar()