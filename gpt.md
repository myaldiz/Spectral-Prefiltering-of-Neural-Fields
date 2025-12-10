I am preparing the slides for my paper Spectral Prefiltering of Neural Fields. 
Read the paper from arxiv, understand the motivation. 
I would like to motivate the problem and show that neural fields provide only point-wise evaluation. 
For different resolutions, we need to filter neural fields through MC sampling etc.

```python
# Colors
COLOR_BOX = '#D46CCD'   # Pinkish (Your Palette)
COLOR_POINTS = '#4E71BE' # Blueish (Your Palette)

SRC_RES = 512
SAMPLE_RES = 32
FRAMES = 60
MIN_SPAN = 0.4
MAX_SPAN = 1.5

spans = np.linspace(MIN_SPAN, MAX_SPAN, FRAMES)

# Use a clean style
plt.style.use('seaborn-v0_8-white') 

# Create Figure with Transparent Background
fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
fig.patch.set_alpha(0.0) # Transparent figure background

# A. Global Context View
image = (full_field.grid[0] + 1.0) / 2.0
bg_img_np = image.permute(1, 2, 0).detach().cpu().numpy()

# Plot context
im_context = axs[0].imshow(bg_img_np, extent=(-1, 1, 1, -1), cmap='gray')
axs[0].set_title("Input Signal", color='gray')
axs[0].axis('off')
axs[0].patch.set_alpha(0.0) # Transparent axis

# Init Dynamic Artists
# Box with your Pastel Pink
rect = patches.Rectangle((0,0), 0, 0, linewidth=3, edgecolor=COLOR_BOX, facecolor='none')
axs[0].add_patch(rect)

# Points with your Pastel Blue
# 'antialiased=True' and smaller size (s=1.5) helps reduce Moiré patterns
scat = axs[0].scatter([], [], s=1.5, c=COLOR_POINTS, alpha=0.9, antialiased=True) 

# B. Naive View
dummy_data = np.zeros((SAMPLE_RES, SAMPLE_RES, 3))
im_naive = axs[1].imshow(dummy_data, vmin=0, vmax=1, interpolation='nearest') 
axs[1].set_title(f"Naive Sampling", color=COLOR_BOX)
axs[1].axis('off')
axs[1].patch.set_alpha(0.0)

# C. Ours View
im_ours = axs[2].imshow(dummy_data, vmin=0, vmax=1, interpolation='nearest')
axs[2].set_title(f"Spectral Prefiltering", color=COLOR_POINTS)
axs[2].axis('off')
axs[2].patch.set_alpha(0.0)

# --- 4. Render Loop ---
OUTPUT_FILENAME = 'vis_motivation.mp4'
writer = imageio.get_writer(OUTPUT_FILENAME, fps=15)

print(f"Rendering {FRAMES} frames to {OUTPUT_FILENAME}...")

for i, span in enumerate(spans):
    # 1. Logic
    bounds = (-span/2, -span/2, span/2, span/2)
    coords = make_coord_grid(resolution=SAMPLE_RES, bounds=bounds, device=device)

    # Naive
    naive_out = full_field(coords, mode='bilinear')

    # Ours (Simulation)
    pixel_size_norm = span / SAMPLE_RES
    target_res = int(2.0 / pixel_size_norm)
    
    if target_res < SRC_RES:
        # Simulate prefiltering
        img_down = F.interpolate(full_field.grid, size=(target_res, target_res), mode='area')
        adaptive_field = GridField(img_down).to(device)
        ours_out = adaptive_field(coords, mode='bilinear')
    else:
        ours_out = full_field(coords, mode='bilinear')

    # 2. Update Visuals
    
    # Update Box
    rect.set_xy((bounds[0], bounds[1]))
    rect.set_width(span)
    rect.set_height(span)
    
    # Update Points
    c_np = coords.cpu().numpy()
    scat.set_offsets(c_np)
    
    # Update Images
    naive_img = naive_out.view(SAMPLE_RES, SAMPLE_RES, 3).detach().cpu().numpy() * 0.5 + 0.5
    ours_img = ours_out.view(SAMPLE_RES, SAMPLE_RES, 3).detach().cpu().numpy() * 0.5 + 0.5
    
    # Clamp for safety
    naive_img = np.clip(naive_img, 0, 1)
    ours_img = np.clip(ours_img, 0, 1)

    im_naive.set_data(naive_img)
    im_ours.set_data(ours_img)
    
    # 3. Capture Frame with Alpha
    fig.canvas.draw()
    
    # Get RGBA buffer
    frame = np.asarray(fig.canvas.buffer_rgba())
    writer.append_data(frame)

writer.close()
plt.close()
print(f"Done. Saved {OUTPUT_FILENAME}")
```

I would like to fix couple things.
- Text resolution looks really bad in the video. Please fix this.
- Sample points also look aliased. Can you improve their rendering quality?
- Visualization should not be only zoom. Go around square trajectory first go right zoomed in, then down and simulatenously zoom out, the go left with zoomed out, then go up and zoom in again. Don't sample outside the image. Draw the rectangle based on zoomed out region covering until the edges.