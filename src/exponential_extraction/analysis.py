from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import signal
from scipy.optimize import curve_fit


def parse_complex_number(value: str) -> complex:
    """Convert a complex number string that uses ``i`` into Python complex form."""
    return complex(value.replace("i", "j"))


def mono_exp(t, amplitude, Tm_ns, offset):
    """Return a monoexponential decay."""
    return amplitude * np.exp(-t / Tm_ns) + offset


def stretched_exp(t, amplitude, Tm_ns, beta, offset):
    """Return a stretched-exponential decay."""
    return amplitude * np.exp(-((t / Tm_ns) ** beta)) + offset


def get_project_root() -> Path:
    """Resolve the repository root by locating the local ``data`` directory."""
    root = Path.cwd()
    if not (root / "data").exists() and (root.parent / "data").exists():
        root = root.parent
    data_dir = root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Could not find data directory at {data_dir}")
    return root


def extract_temperature(filename: str) -> float:
    """Extract the temperature value in kelvin from a measurement filename."""
    match = re.search(r"_(\d+\.?\d{0,1}?)K", filename, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Could not extract temperature from {filename}")
    return float(match.group(1))


def find_measurement_files(data_dir: Path, pattern: str = "*Tm*K.dat") -> pd.DataFrame:
    """Find matching data files and return them as a temperature-sorted table."""
    files = list(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No measurement files matching {pattern!r} were found in {data_dir}"
        )
    file_table = pd.DataFrame(
        {
            "T (K)": [extract_temperature(path.name) for path in files],
            "File": [str(path) for path in files],
        }
    )
    return file_table.sort_values("T (K)", ignore_index=True)


def load_Tm_dataset(file_table: pd.DataFrame) -> pd.DataFrame:
    """Load the Tm traces and add derived columns used in the analysis."""
    traces = []
    for _, record in file_table.iterrows():
        trace = pd.read_csv(
            record.File,
            sep="\t",
            converters={1: parse_complex_number},
            header=None,
        ).rename(columns={0: "2tau (ns)", 1: "Echo"})
        trace["T (K)"] = record["T (K)"]
        trace["Real"] = trace["Echo"].apply(np.real)
        real_max = trace["Real"].max()
        trace["RealNorm"] = trace["Real"] / real_max
        trace["Saturation"] = 100 * trace["Real"] / trace.loc[0, "Real"]
        traces.append(trace)
    dataset = pd.concat(traces, ignore_index=True)
    return dataset.set_index("T (K)").sort_index()


def build_plot_colors(temperatures: pd.Series) -> list[str]:
    """Create a temperature-ordered list of Plotly RGB colors."""
    colors = 255 * plt.cm.brg(np.linspace(0, 0.5, len(temperatures)))
    return [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b, _ in colors]


def apply_plot_style(fig: go.Figure, title: str, x_title: str, y_title: str) -> go.Figure:
    """Apply the shared Plotly layout used throughout the notebook."""
    fig.update_layout(
        template="simple_white",
        title=title,
        title_font_size=26,
        xaxis=dict(tickformat="f", hoverformat="0.1f", title=x_title, mirror=True),
        yaxis=dict(tickformat="0.2f", hoverformat="0.3f", title=y_title, mirror=True),
        font=dict(family="Tahoma", size=22),
        margin_t=60,
        legend=dict(font=dict(family="Tahoma", size=16), title="T (K)", orientation="v"),
    )
    return fig


def filter_signals(dataset: pd.DataFrame, temperatures: pd.Series, order: int, wn: float):
    """Low-pass filter the normalized traces and locate minima in the filtered data."""
    b_coeff, a_coeff = signal.butter(order, Wn=wn, btype="lowpass")
    filtered_signals = {}
    filtered_frames = {}
    peak_indices = {}

    for temperature in temperatures:
        filtered = signal.filtfilt(b_coeff, a_coeff, dataset.loc[temperature, "RealNorm"])
        filtered_signals[temperature] = filtered
        filtered_frames[temperature] = pd.DataFrame(
            {"Filtered": filtered, "T (K)": temperature}
        ).set_index("T (K)")
        peak_indices[temperature] = signal.find_peaks(-filtered)[0]

    return filtered_signals, filtered_frames, peak_indices


def expand_peak_positions(
    filtered_values: np.ndarray, peak_positions: np.ndarray, neighbours: int, tolerance: float
) -> list[int]:
    """Expand each detected minimum by including nearby points within the tolerance window."""
    expanded_positions = set(int(position) for position in peak_positions)

    for position in peak_positions:
        peak_value = filtered_values[position]
        lower_bound = peak_value / (1 + tolerance)
        upper_bound = peak_value * (1 + tolerance)
        for step in range(1, neighbours + 1):
            for candidate in (position - step, position + step):
                if 0 <= candidate < len(filtered_values):
                    if lower_bound <= filtered_values[candidate] <= upper_bound:
                        expanded_positions.add(int(candidate))

    return sorted(expanded_positions)


def build_peak_windows(
    dataset: pd.DataFrame,
    temperatures: pd.Series,
    filtered_signals: dict,
    filtered_frames: dict,
    peak_indices: dict,
    neighbours: int,
    tolerance: float,
    cutoff: float,
):
    """Build the fitting windows from the filtered minima and neighboring points."""
    peak_windows = {}

    for temperature in temperatures:
        selected_positions = expand_peak_positions(
            filtered_signals[temperature],
            peak_indices[temperature],
            neighbours=neighbours,
            tolerance=tolerance,
        )
        peak_windows[temperature] = (
            pd.concat(
                [
                    pd.DataFrame(dataset.loc[temperature, "2tau (ns)"].iloc[selected_positions]),
                    pd.DataFrame(dataset.loc[temperature, "RealNorm"].iloc[selected_positions]),
                    pd.DataFrame(filtered_frames[temperature].iloc[selected_positions]),
                ],
                axis=1,
            )
            .drop_duplicates()
            .sort_values(by="2tau (ns)")
            .reset_index()
        )
        peak_windows[temperature] = peak_windows[temperature].loc[
            peak_windows[temperature]["2tau (ns)"] >= cutoff
        ].set_index("T (K)")

    return peak_windows


def build_peak_rectangles(peak_windows: dict, temperatures: pd.Series) -> pd.DataFrame:
    """Convert contiguous peak-window regions into rectangle boundaries for plotting."""
    rectangles = []

    for temperature in temperatures:
        indexed = peak_windows[temperature].reset_index()
        if len(indexed) < 2:
            continue

        start = 0
        step_reference = indexed["2tau (ns)"].iloc[1] - indexed["2tau (ns)"].iloc[0]

        for idx in range(len(indexed) - 1):
            current_step = indexed["2tau (ns)"].iloc[idx + 1] - indexed["2tau (ns)"].iloc[idx]
            last_segment = idx == len(indexed) - 2

            if not np.isclose(current_step, step_reference) or last_segment:
                end_idx = idx if not last_segment else idx + 1
                t_left = indexed["2tau (ns)"].iloc[start]
                t_right = indexed["2tau (ns)"].iloc[end_idx]
                y_slice = indexed["RealNorm"].iloc[start : end_idx + 1]
                y_max = y_slice.max()
                y_min = y_slice.min()

                if np.isclose(t_left, t_right):
                    t_left -= step_reference / 2
                    t_right += step_reference / 2
                    y_max *= 1.25
                    y_min /= 1.25

                rectangles.append(
                    {
                        "T (K)": temperature,
                        "t_l": t_left,
                        "t_r": t_right,
                        "Max": y_max,
                        "Min": y_min,
                    }
                )
                start = idx + 1

    return pd.DataFrame(rectangles).set_index("T (K)")


def rectangles_to_plotly_paths(rectangles: pd.DataFrame, temperatures: pd.Series):
    """Convert rectangle boundaries into Plotly-compatible filled trace paths."""
    x_paths = []
    y_paths = []

    for temperature in temperatures:
        if temperature in rectangles.index:
            temperature_rectangles = rectangles.loc[[temperature]]
        else:
            temperature_rectangles = pd.DataFrame(columns=rectangles.columns)

        x_values = []
        y_values = []

        for idx, row in enumerate(temperature_rectangles.itertuples()):
            if idx != 0:
                x_values.append(None)
                y_values.append(None)
            x_values.extend([row.t_l, row.t_l, row.t_r, row.t_r, row.t_l])
            y_values.extend([row.Min, row.Max, row.Max, row.Min, row.Min])

        x_paths.append(x_values)
        y_paths.append(y_values)

    return x_paths, y_paths


def fit_model(
    peak_windows: dict,
    temperatures: pd.Series,
    model_func,
    p0: list[float],
    bounds: tuple[list[float], list[float]],
    parameter_names: list[str],
) -> pd.DataFrame:
    """Fit one decay model to every temperature-dependent fitting window."""
    fitted_parameters = {}

    for temperature in temperatures:
        fit_values, _ = curve_fit(
            f=model_func,
            xdata=peak_windows[temperature]["2tau (ns)"],
            ydata=peak_windows[temperature]["RealNorm"],
            p0=p0,
            bounds=bounds,
        )
        fitted_parameters[temperature] = fit_values

    return pd.DataFrame.from_dict(
        fitted_parameters, orient="index", columns=parameter_names
    ).reset_index(names="T (K)")


def run_Tm_analysis(
    data_dir: Path,
    measurement_pattern: str,
    filter_order: int,
    filter_wn: float,
    peak_neighbours: int,
    peak_tolerance: float,
    peak_cutoff_ns: float,
    fit_definitions: dict,
) -> dict:
    """Run the full Tm analysis pipeline and return the intermediate results."""
    Tm_files = find_measurement_files(data_dir, pattern=measurement_pattern)
    Tm_data = load_Tm_dataset(Tm_files)
    temperatures = Tm_files["T (K)"]

    filtered_signals, filtered_frames, peak_indices = filter_signals(
        Tm_data,
        temperatures,
        order=filter_order,
        wn=filter_wn,
    )
    peak_windows = build_peak_windows(
        Tm_data,
        temperatures,
        filtered_signals,
        filtered_frames,
        peak_indices,
        neighbours=peak_neighbours,
        tolerance=peak_tolerance,
        cutoff=peak_cutoff_ns,
    )
    peak_rectangles = build_peak_rectangles(peak_windows, temperatures)
    rectangle_paths = rectangles_to_plotly_paths(peak_rectangles, temperatures)

    fit_results = {}
    for fit_name, fit_settings in fit_definitions.items():
        fit_results[fit_name] = fit_model(
            peak_windows,
            temperatures,
            **fit_settings,
        )

    return {
        "Tm_files": Tm_files,
        "Tm_data": Tm_data,
        "temperatures": temperatures,
        "filtered_signals": filtered_signals,
        "filtered_frames": filtered_frames,
        "peak_indices": peak_indices,
        "peak_windows": peak_windows,
        "peak_rectangles": peak_rectangles,
        "rectangle_paths": rectangle_paths,
        "fit_results": fit_results,
    }


def plot_raw_decay(
    dataset: pd.DataFrame,
    file_table: pd.DataFrame,
    title: str = "Tm echo decay of PorphVO (Q-band)",
    x_title: str = "2&#964; (ns)",
    y_title: str = "Echo signal (arb. un.)",
) -> go.Figure:
    """Plot the raw real-part decay traces for all temperatures."""
    temperatures = file_table["T (K)"]
    colors = build_plot_colors(temperatures)
    fig = go.Figure()

    for color, temperature in zip(colors, temperatures):
        fig.add_trace(
            go.Scatter(
                x=dataset.loc[temperature, "2tau (ns)"],
                y=dataset.loc[temperature, "Real"] / 1e6,
                mode="lines",
                customdata=dataset.loc[temperature, "Saturation"].to_numpy(),
                name=f"{temperature} K",
                marker=dict(color=color),
                hovertemplate="<b>Tm echo decay</b><br>"
                + f"T = {temperature} K<br>"
                + "2&#964;: %{x:0.0f} ns<br>"
                + "Echo signal: %{y:0.4f}<br>"
                + "Relative signal: %{customdata:0.1f} %<extra></extra>",
            )
        )

    return apply_plot_style(fig, title, x_title, y_title)


def add_raw_peak_trace(
    fig: go.Figure,
    temperature_data: pd.DataFrame,
    temperature: float,
    color: str,
) -> None:
    """Add the normalized raw trace for one temperature to the figure."""
    fig.add_trace(
        go.Scatter(
            x=temperature_data["2tau (ns)"],
            y=temperature_data["RealNorm"],
            mode="lines",
            customdata=temperature_data["Saturation"].to_numpy(),
            name=f"{temperature} K",
            marker=dict(color=color),
            hovertemplate="<b>Tm echo decay</b><br>"
            + f"T = {temperature} K<br>"
            + "2&#964;: %{x:0.0f} ns<br>"
            + "Echo signal: %{y:0.3f}<br>"
            + "Relative signal: %{customdata:0.1f} %<extra></extra>",
        )
    )


def add_filtered_peak_trace(
    fig: go.Figure,
    temperature_data: pd.DataFrame,
    filtered_signal: np.ndarray,
    temperature: float,
) -> None:
    """Add the filtered trace for one temperature to the figure."""
    fig.add_trace(
        go.Scatter(
            x=temperature_data["2tau (ns)"],
            y=filtered_signal,
            mode="lines",
            name=f"{temperature} K (filtered)",
            marker=dict(color="black"),
            visible="legendonly",
        )
    )


def add_fit_points_trace(fig: go.Figure, temperature_peak_window: pd.DataFrame, temperature: float) -> None:
    """Add the selected fitting points for one temperature to the figure."""
    fig.add_trace(
        go.Scatter(
            x=temperature_peak_window["2tau (ns)"],
            y=temperature_peak_window["RealNorm"],
            mode="markers",
            name=f"{temperature} K (fit points)",
            marker=dict(color="black", size=10),
            visible="legendonly",
        )
    )


def add_fit_area_trace(
    fig: go.Figure,
    x_path: list[float],
    y_path: list[float],
    temperature: float,
) -> None:
    """Add the highlighted fitting area for one temperature to the figure."""
    fig.add_trace(
        go.Scatter(
            x=x_path,
            y=y_path,
            name=f"{temperature} K (fit area)",
            fill="toself",
            fillcolor="rgba(255,200,65,0.3)",
            line_color="orange",
            visible="legendonly",
        )
    )


def build_fit_x_grid(temperature_data: pd.DataFrame, num_points: int = 200) -> np.ndarray:
    """Create a smooth x-grid spanning the trace for plotting fitted curves."""
    return np.linspace(
        temperature_data["2tau (ns)"].iloc[0],
        temperature_data["2tau (ns)"].iloc[-1],
        num_points,
    )


def add_model_fit_traces(
    fig: go.Figure,
    x_plot: np.ndarray,
    mono_parameters: np.ndarray,
    gaussian_parameters: np.ndarray,
    stretched_parameters: np.ndarray,
    temperature: float,
    show_gaussian_fit: bool = True,
    show_stretched_fit: bool = True,
) -> None:
    """Add monoexponential and optionally Gaussian-style and stretched fit curves."""
    fig.add_trace(
        go.Scatter(
            x=x_plot,
            y=mono_exp(x_plot, *mono_parameters),
            mode="lines",
            name=f"{temperature} K (mono fit)",
            line=dict(color="brown", width=2, dash="dash"),
            visible="legendonly",
        )
    )
    if show_gaussian_fit:
        fig.add_trace(
            go.Scatter(
                x=x_plot,
                y=stretched_exp(x_plot, *gaussian_parameters),
                mode="lines",
                name=f"{temperature} K (Gaussian fit)",
                line=dict(color="dimgray", width=2, dash="dot"),
                visible="legendonly",
            )
        )
    if show_stretched_fit:
        fig.add_trace(
            go.Scatter(
                x=x_plot,
                y=stretched_exp(x_plot, *stretched_parameters),
                mode="lines",
                name=f"{temperature} K (stretched fit)",
                line=dict(color="black", width=2, dash="dash"),
                visible="legendonly",
            )
        )


def plot_peak_selection(
    dataset: pd.DataFrame,
    file_table: pd.DataFrame,
    filtered_signals: dict,
    peak_windows: dict,
    rectangle_paths: tuple[list[list[float]], list[list[float]]],
    mono_fit: pd.DataFrame,
    gaussian_fit: pd.DataFrame,
    stretched_fit: pd.DataFrame,
    title: str = "Tm echo decay",
    x_title: str = "2&#964; (ns)",
    y_title: str = "Echo signal (arb. un.)",
    show_gaussian_fit: bool = True,
    show_stretched_fit: bool = True,
) -> go.Figure:
    """Plot the normalized traces together with fitting windows and fitted curves."""
    temperatures = file_table["T (K)"]
    colors = build_plot_colors(temperatures)
    x_paths, y_paths = rectangle_paths
    fig = go.Figure()

    for idx, temperature in enumerate(temperatures):
        temperature_data = dataset.loc[temperature]
        temperature_peak_window = peak_windows[temperature]
        mono_parameters = mono_fit.loc[mono_fit["T (K)"] == temperature].iloc[0, 1:].to_numpy()
        gaussian_parameters = gaussian_fit.loc[
            gaussian_fit["T (K)"] == temperature
        ].iloc[0, 1:].to_numpy()
        stretched_parameters = stretched_fit.loc[
            stretched_fit["T (K)"] == temperature
        ].iloc[0, 1:].to_numpy()
        x_plot = build_fit_x_grid(temperature_data)

        add_raw_peak_trace(fig, temperature_data, temperature, colors[idx])
        add_filtered_peak_trace(fig, temperature_data, filtered_signals[temperature], temperature)
        add_fit_points_trace(fig, temperature_peak_window, temperature)
        add_fit_area_trace(fig, x_paths[idx], y_paths[idx], temperature)
        add_model_fit_traces(
            fig,
            x_plot,
            mono_parameters,
            gaussian_parameters,
            stretched_parameters,
            temperature,
            show_gaussian_fit=show_gaussian_fit,
            show_stretched_fit=show_stretched_fit,
        )

    return apply_plot_style(fig, title, x_title, y_title)


def plot_Tm_vs_temperature(
    mono_fit: pd.DataFrame,
    stretched_fit: pd.DataFrame,
    gaussian_fit: pd.DataFrame,
    title: str = "Tm vs T",
    x_title: str = "T (K)",
    y_title: str = "Tm (ns)",
    x_range: list[float] | None = None,
    y_range: list[float] | None = None,
    show_stretched_fit: bool = True,
    show_gaussian_fit: bool = True,
) -> go.Figure:
    """Plot fitted Tm values against temperature for the available fit models."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=mono_fit["T (K)"],
            y=mono_fit["Tm (ns)"],
            mode="markers",
            name="Tm mono fit",
            marker=dict(size=12),
        )
    )
    if show_stretched_fit:
        fig.add_trace(
            go.Scatter(
                x=stretched_fit["T (K)"],
                y=stretched_fit["Tm (ns)"],
                mode="markers",
                name="Tm stretched fit",
                marker=dict(size=12),
            )
        )
    if show_gaussian_fit:
        fig.add_trace(
            go.Scatter(
                x=gaussian_fit["T (K)"],
                y=gaussian_fit["Tm (ns)"],
                mode="markers",
                name="Tm Gaussian fit",
                marker=dict(size=12),
            )
        )

    fig = apply_plot_style(fig, title, x_title, y_title)
    fig.update_xaxes(range=x_range)
    fig.update_yaxes(range=y_range, tickformat="0.0f")
    return fig
