import logging
mpl_logger = logging.getLogger('matplotlib')  # must before import matplotlib
mpl_logger.setLevel(logging.WARNING)
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pylab as plt


def save_figure_to_numpy(fig):
    # Ensure the figure is drawn
    fig.canvas.draw()
    
    # Get the RGBA buffer from the canvas
    rgba_buffer = fig.canvas.buffer_rgba()
    
    # Convert to a NumPy array
    data = np.asarray(rgba_buffer, dtype=np.uint8)
    
    # Extract RGB channels (discard alpha) and reshape to (height, width, 3)
    data = data[:, :, :3]
    
    # Transpose to (3, height, width) as required
    data = np.transpose(data, (2, 0, 1))
    
    return data


def plot_waveform_to_numpy(waveform):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot()
    ax.plot(range(len(waveform)), waveform,
            linewidth=0.1, alpha=0.7, color='blue')

    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.ylim(-1, 1)
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()

    return data


def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data
