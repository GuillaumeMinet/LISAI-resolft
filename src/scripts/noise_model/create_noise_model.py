"""User entry point for creating histogram + GMM noise models.

Edit values in USER PARAMETERS, then run:
    python src/scripts/noise_model/create_noise_model.py
"""

from lisai.data.noise_model import NoiseModelBuildConfig, build_noise_model


def main():
    # =========================
    # USER PARAMETERS
    # =========================
    dataset_name = "Mito_fixed"

    # subfolders are relative to canonical dataset dir:
    # <data_root>/Data/<dataset_name>/<subfolder>
    signal_subfolder = r"preprocess\recon\gt_avg\train"
    observation_subfolder = r"preprocess\recon\inp_mltpl_snr\train"  # or "same"

    # int for one level, or list[int] for multiple SNR levels
    noise_level = 9
    signal_idx = None
    create_avg_signal = False
    create_avg_signal_n_frames = 3  # int or "all"

    # preprocessing before model fit
    norm_sig_to_obs = True
    normalize_everything = True
    clip = -3.0
    crop_size = (1200, 1200)
    filters = ("tif", "tiff")

    # fitting params
    histogram_bins = 256
    gmm_n_gaussian = 5
    gmm_n_coeff = 4
    gmm_n_epochs = 3000
    gmm_learning_rate = 0.05
    gmm_batch_size = 250_000
    gmm_min_sigma = 1.0

    # output naming / diagnostics
    overwrite = False
    signal_info = "SnrAvg"
    save_name = None  # None -> auto name
    display = False
    save_gmm_plot = True

    cfg = NoiseModelBuildConfig(
        dataset_name=dataset_name,
        signal_subfolder=signal_subfolder,
        observation_subfolder=observation_subfolder,
        noise_level=noise_level,
        signal_idx=signal_idx,
        create_avg_signal=create_avg_signal,
        create_avg_signal_n_frames=create_avg_signal_n_frames,
        norm_sig_to_obs=norm_sig_to_obs,
        normalize_everything=normalize_everything,
        clip=clip,
        crop_size=crop_size,
        filters=filters,
        histogram_bins=histogram_bins,
        gmm_n_gaussian=gmm_n_gaussian,
        gmm_n_coeff=gmm_n_coeff,
        gmm_n_epochs=gmm_n_epochs,
        gmm_learning_rate=gmm_learning_rate,
        gmm_batch_size=gmm_batch_size,
        gmm_min_sigma=gmm_min_sigma,
        overwrite=overwrite,
        signal_info=signal_info,
        save_name=save_name,
        display=display,
        save_gmm_plot=save_gmm_plot,
    )

    result = build_noise_model(cfg)
    print("\nNoise model build completed.")
    print(f"Save dir: {result.save_dir}")
    print(f"Histogram: {result.histogram_path}")
    print(f"GMM: {result.gmm_path}")
    print(f"Norm prm: {result.norm_prm_path}")
    print(f"Info: {result.info_path}")
    if result.gmm_plot_png_path is not None:
        print(f"GMM plot (png): {result.gmm_plot_png_path}")
    if result.gmm_plot_svg_path is not None:
        print(f"GMM plot (svg): {result.gmm_plot_svg_path}")


if __name__ == "__main__":
    main()
