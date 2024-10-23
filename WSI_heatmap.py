import openslide
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


# wsi dir
# src = "WSI/KOH_Dataset_test_lambda/Fungal_Positive" # source directory of WSIs
dst = "test/Test_WSIs" # destination directory of high resolution images converted from WSIs
dir_heatmap = "test/output" # directory of heatmaps outputs
overlay_dir = "test/overlay"

# loop over the WSIs of the directory and save images with high resolution
for wsi_name in os.listdir(src):
    # open wsi
    wsi = openslide.OpenSlide(os.path.join(src, wsi_name))
    # convert to high resolution image
    img = wsi.read_region((0,0), 0, wsi.level_dimensions[0])
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    # save image with same name as wsi but in dst
    # create if not exists
    if not os.path.exists(dst):
        os.makedirs(dst)
    # save image and remove 'svs' extension
    cv2.imwrite(os.path.join(dst, os.path.splitext(wsi_name)[0] + ".png"), img)
    # close wsi
    wsi.close()
# done


# Create a scale bar
scale_bar = np.zeros((100, 10, 3), dtype=np.uint8)
# Use the 'winter' colormap
colorjet = plt.get_cmap("winter")
# Loop over the scale bar and fill with colors from the colormap
for i in range(100):
    # Get color from colormap
    color = colorjet(i / 100)
    # Fill row with color
    scale_bar[99 - i, :, :] = (np.array(color[:3]) * 255).astype(np.uint8)
# Plot the scale bar
fig, ax = plt.subplots(figsize=(1, 25))
ax.imshow(scale_bar, aspect='auto')
ax.set_title("Color Scale", fontsize=25)
# Adjust the y-axis to show probabilities
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_yticklabels([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
ax.set_xticks([])  # Hide x-axis ticks
# Add y-axis label
ax.set_ylabel('Patch Score', fontsize=25)
# Remove spines for a cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
# change label size
ax.tick_params(axis='y', labelsize=25)
plt.show()
plt.savefig("test/scale_bar.png", bbox_inches='tight', pad_inches=0.1)


# create folder if not exists
if not os.path.exists(dir_heatmap):
    os.makedirs(dir_heatmap)
if not os.path.exists(overlay_dir):
    os.makedirs(overlay_dir)

# create overlay of heatmap on WSI
# List directories
wsi_list = sorted(os.listdir(dst))
heatmap_list = sorted(os.listdir(dir_heatmap))
print("WSI List:", wsi_list)
print("Attention Map List", heatmap_list)

for wsi_name in wsi_list:
    for heatmap_name in heatmap_list:
        if os.path.splitext(wsi_name)[0] == os.path.splitext(heatmap_name)[0]:
            try:
                print(os.path.splitext(wsi_name)[0])
                img_wsi = cv2.imread(os.path.join(dst, wsi_name))
                img_heatmap = cv2.imread(os.path.join(dir_heatmap, heatmap_name))

                # # Convert heatmap to grayscale if not already
                # if len(img_heatmap.shape) == 3:
                #     img_heatmap = cv2.cvtColor(img_heatmap, cv2.COLOR_BGR2GRAY)
                
                # convert heatmap to greyscale by taking the highest value of the 3 channels
                img_heatmap = img_heatmap.max(axis=2)
                
                # # heatmap values min max with average
                # print("Heatmap values min: ", img_heatmap.min())
                # print("Heatmap values max: ", img_heatmap.max())
                # print("Heatmap values mean: ", img_heatmap.mean())
                # # mean without the 0s
                # print("Heatmap values mean without 0s: ", img_heatmap[img_heatmap != 0].mean())
                # # min without 0s
                # print("Heatmap values min without 0s: ", img_heatmap[img_heatmap != 0].min())

                # # Apply a color map to the heatmap
                # img_heatmap = cv2.applyColorMap(img_heatmap, cv2.COLORMAP_JET)
                # try colormap winter
                img_heatmap = cv2.applyColorMap(img_heatmap, cv2.COLORMAP_WINTER)  

                # Rotate and resize heatmap
                img_heatmap = cv2.rotate(img_heatmap, cv2.ROTATE_90_CLOCKWISE)
                img_heatmap = cv2.flip(img_heatmap, 1) # flip on vertical axis
                img_heatmap = cv2.resize(img_heatmap, (img_wsi.shape[1], img_wsi.shape[0]))

                # Create overlay with modified alpha and blending
                alpha = 0.65
                overlay = cv2.addWeighted(img_heatmap, alpha, img_wsi, 1 - alpha, 0)

                # Save overlay image
                if not os.path.exists(overlay_dir):
                    os.makedirs(overlay_dir)
                cv2.imwrite(os.path.join(overlay_dir, os.path.basename(wsi_name)), overlay)
                print("Overlay created for", os.path.splitext(wsi_name)[0])

                # Break out of the inner loop once a match is found and processed
                break
            except Exception as e:
                print(e)
                continue
        # break
print("Done!")