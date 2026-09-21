"""User entry point for checking a trained GMM noise model.

Edit values in USER PARAMETERS, then run:
    python src/scripts/noise_model/check_noise_model.py
"""

from lisai.data.noise_model import NoiseModelCheckConfig, check_noise_model


def main():
    # =========================
    # USER PARAMETERS
    # =========================
    dataset_name = "Vim_fixed_mltplSNR_30nm"

    # subfolders are relative to canonical dataset dir:
    # <data_root>/Data/<dataset_name>/<subfolder>
    signal_subfolder = r"inference\N2V\Vim_fixed_Avg1-3_no_clipping"
    observation_subfolder = r"dump\rec\timelapses_gathered"  # or "same"

    # choose one:
    noise_model_name = None  # e.g. "Noise0_SigN2Vavg_Clip-3_norm_bis"
    noise_model_path = r"E:\dl_monalisa\NoiseModels\Noise0_SigN2Vavg_Clip-3_norm_bis\GMM.npz"

    # data selection
    noise_level = 0  # int or list[int]
    signal_idx = 1
    create_avg_signal = False
    create_avg_signal_n_frames = 1  # int or "all"

    # preprocessing before comparison
    norm_sig_to_obs = False
    normalize_everything = True
    clip = 0.0
    crop_size = None
    filters = ("tif", "tiff")

    # plotting
    histogram_bins = 256
    display = True
    display_crop_size = 400
    gmm_plot_points = 15
    gmm_plot_max_columns = 4
    save_plot = True
    plot_basename = "GMM_check_plot"

    cfg = NoiseModelCheckConfig(
        dataset_name=dataset_name,
        signal_subfolder=signal_subfolder,
        observation_subfolder=observation_subfolder,
        noise_model_name=noise_model_name,
        noise_model_path=noise_model_path,
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
        display=display,
        display_crop_size=display_crop_size,
        gmm_plot_points=gmm_plot_points,
        gmm_plot_max_columns=gmm_plot_max_columns,
        save_plot=save_plot,
        plot_basename=plot_basename,
    )

    result = check_noise_model(cfg)
    print("\nNoise model check completed.")
    print(f"Noise model: {result.noise_model_path}")
    print(f"Files used: {result.n_files_used}")
    print(f"Signal shape: {result.signal_shape}")
    print(f"Observation shape: {result.observation_shape}")
    print(f"Signal range: [{result.min_signal}, {result.max_signal}]")
    print(f"Observation stats: mean={result.avg_obs}, std={result.std_obs}")
    if result.plot_png_path is not None:
        print(f"Plot (png): {result.plot_png_path}")
    if result.plot_svg_path is not None:
        print(f"Plot (svg): {result.plot_svg_path}")


if __name__ == "__main__":
    main()
