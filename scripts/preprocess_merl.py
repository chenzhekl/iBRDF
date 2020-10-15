import os
from argparse import ArgumentParser
from os import path
from multiprocessing import Pool
import functools

from scipy.interpolate import RegularGridInterpolator

from common import *


def process(in_folder, out_folder, filename):
    data = load_merl(path.join(in_folder, filename))
    data[data < 0.0] = 0.0

    mask = data[0] != 0
    axis = 0
    bound = data[0].shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    scale_ratio = np.power((bound + 1) / BRDF_SAMPLING_RES_THETA_H, 2)

    theta_h = (
        np.power(np.linspace(0.0, 1.0, BRDF_SAMPLING_RES_THETA_H), 2)
        * BRDF_SAMPLING_RES_THETA_H
    )
    theta_d = (
        np.linspace(0.0, 1.0, BRDF_SAMPLING_RES_THETA_D) * BRDF_SAMPLING_RES_THETA_D
    )
    phi_d = np.linspace(0.0, 1.0, BRDF_SAMPLING_RES_PHI_D // 2) * (
        BRDF_SAMPLING_RES_PHI_D // 2
    )

    thv, tdv, pdv = np.meshgrid(theta_h, theta_d, phi_d, indexing="ij")
    grid = np.stack([thv.flatten(), tdv.flatten(), pdv.flatten()], axis=0).T
    grid = np.reshape(
        grid,
        [
            BRDF_SAMPLING_RES_THETA_H,
            BRDF_SAMPLING_RES_THETA_D,
            BRDF_SAMPLING_RES_PHI_D // 2,
            3,
        ],
    )

    warpped_grid = grid.copy()
    warpped_grid[:, :, :, 0] = grid[:, :, :, 0] * scale_ratio
    warpped_grid = np.reshape(warpped_grid, [-1, 3])

    new_dist = []
    for color_id in range(3):
        mono_brdf = data[color_id]
        interpolator = RegularGridInterpolator(
            [theta_h, theta_d, phi_d], mono_brdf, method="linear"
        )
        new_dist.append(interpolator(warpped_grid))

    new_dist = np.stack(new_dist, axis=0)
    save_merl(new_dist, path.join(out_folder, filename))

    print(f"Processed {filename}")


def main():
    parser = ArgumentParser()
    parser.add_argument("input", type=str, help="Path to the input folder")
    parser.add_argument("output", type=str, help="Path to the output folder")
    args = parser.parse_args()

    brdf_list = []
    for _, _, files in os.walk(args.input):
        for f in files:
            if not f.endswith(".binary"):
                continue
            brdf_list.append(f)

    print(f"Found {len(brdf_list)} MERL materials")

    with Pool() as p:
        p.map(functools.partial(process, args.input, args.output), brdf_list)


if __name__ == "__main__":
    main()
