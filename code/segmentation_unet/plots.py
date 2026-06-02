import numpy as np
import matplotlib.pyplot as plt
import config


def plot_training_progress(batch_losses, val_losses, val_mious, val_per_class, save_path, title=''):
    num_epochs = len(val_losses)
    if num_epochs == 0:
        return

    epoch_x = np.arange(1, num_epochs + 1)
    batch_x = np.linspace(1, num_epochs, len(batch_losses))
    per_class_arr = np.array(val_per_class)
    colors = ['gray', 'red', 'blue', 'orange']

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    ax1.plot(batch_x, batch_losses, color='lightblue', alpha=0.4, linewidth=1, label='Train (batch)')
    win = max(len(batch_losses) // (num_epochs * 2), 5)
    if len(batch_losses) > win:
        smoothed = np.convolve(batch_losses, np.ones(win) / win, mode='valid')
        ax1.plot(np.linspace(1, num_epochs, len(smoothed)), smoothed,
                 color='blue', linewidth=2, label='Train (trend)')
    ax1.plot(epoch_x, val_losses, color='red', marker='o', linewidth=2,
             linestyle='--', label='Val loss')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training & Validation Loss - {title}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epoch_x, val_mious, color='green', marker='s', linewidth=2, label='Val mIoU')
    ax2.set_ylabel('mIoU')
    ax2.set_title(f'Validation mIoU - {title}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.0)

    for i in range(config.NUM_CLASSES):
        if per_class_arr.shape[0] > 0:
            ax3.plot(epoch_x, per_class_arr[:, i], color=colors[i],
                     marker='.', linewidth=1.5, label=config.CLASS_NAMES[i])
    ax3.set_ylabel('IoU')
    ax3.set_xlabel('Epoch')
    ax3.set_title(f'Per-Class Validation IoU - {title}')
    ax3.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.0)

    for ax in [ax1, ax2, ax3]:
        for e in epoch_x:
            ax.axvline(x=e, color='gray', linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'[INFO] Training graph saved to {save_path}')
