import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os

def generate_plots():
    density_files = sorted(glob.glob("density_z0_step_*.dat"))
    potential_files = sorted(glob.glob("potential_z0_step_*.dat"))

    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    for density_file, potential_file in zip(density_files, potential_files):
        # Extract step number for title
        step = density_file.split("_")[-1].split(".")[0]

        # Load density and potential slices
        density = np.loadtxt(density_file)
        potential = np.loadtxt(potential_file)

        # Plot density
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title(f"Density Slice (z=0, Step={step})")
        plt.imshow(density, origin='lower', cmap='viridis', extent=[0, 128, 0, 128])
        plt.colorbar(label='Density')
        plt.xlabel('x')
        plt.ylabel('y')

        # Plot potential
        plt.subplot(1, 2, 2)
        plt.title(f"Potential Slice (z=0, Step={step})")
        plt.imshow(potential, origin='lower', cmap='plasma', extent=[0, 128, 0, 128])
        plt.colorbar(label='Potential')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(output_dir, f"plot_step_{step}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot: {plot_path}")

def create_gif(output_path="images/simulation.gif"):
    plot_files = sorted(glob.glob("plots/plot_step_*.png"))
    frames = [Image.open(plot_file) for plot_file in plot_files]

    # Save as GIF
    frames[0].save(
        output_path, save_all=True, append_images=frames[1:], duration=100, loop=0
    )
    print(f"GIF saved as {output_path}")

if __name__ == "__main__":
    print("Generating plots")
    generate_plots()

    print("Creating GIF")
    create_gif()
