from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import argparse
import sys
import os
from numpy.typing import NDArray
from tqdm import tqdm
from scipy.io import netcdf_file


def read_cdf_spectra(file_path: str, limit: Optional[int] = None) -> List[
    Tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """
    Read mass spectra from a NetCDF file using scipy's netcdf_file

    Args:
        file_path: Path to the NetCDF file
        limit: Optional limit for number of spectra to read

    Returns:
        List of tuples containing (masses, intensities) for each spectrum
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    spectra = []
    try:
        with netcdf_file(file_path, 'r') as cdf:
            # Get the total number of scans
            point_count = cdf.variables['point_count'][:]
            scan_index = cdf.variables['scan_index'][:]
            mass_values = cdf.variables['mass_values'][:]
            intensity_values = cdf.variables['intensity_values'][:]

            total_scans = len(point_count)
            if limit is not None:
                total_scans = min(limit, total_scans)
                print(f"Limited to first {total_scans} scans from {file_path}")

            with tqdm(total=total_scans, desc=f"Reading {os.path.basename(file_path)}", unit="scan") as pbar:
                for scan_number in range(total_scans):
                    start_idx = scan_index[scan_number]

                    # Handle last scan specially
                    if scan_number == len(point_count) - 1:
                        end_idx = len(mass_values)
                    else:
                        end_idx = scan_index[scan_number + 1]

                    # Extract masses and intensities for this scan
                    scan_masses = mass_values[start_idx:end_idx].astype(np.float64)
                    scan_intensities = intensity_values[start_idx:end_idx].astype(np.float64)

                    # Some basic validation
                    if len(scan_masses) > 0 and len(scan_masses) == len(scan_intensities):
                        spectra.append((scan_masses, scan_intensities))

                    pbar.update(1)

            return spectra

    except Exception as e:
        raise RuntimeError(f"Error reading NetCDF file {file_path}: {str(e)}")


def average_spectrum(spectra: List[Tuple[NDArray[np.float64], NDArray[np.float64]]],
                     mass_precision: float = 0.01) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Average multiple spectra using efficient binning

    Args:
        spectra: List of (masses, intensities) tuples
        mass_precision: Mass precision for binning

    Returns:
        Tuple of (averaged_masses, averaged_intensities)
    """
    # Concatenate all masses and intensities
    all_masses = np.concatenate([m for m, _ in spectra])
    all_intensities = np.concatenate([i for _, i in spectra])

    # Create mass bins
    mass_min = np.min(all_masses)
    mass_max = np.max(all_masses)
    num_bins = int((mass_max - mass_min) / mass_precision) + 1

    # Use histogram to bin the masses and sum intensities
    bins = np.linspace(mass_min, mass_max, num_bins)
    summed_intensities, bin_edges = np.histogram(all_masses, bins=bins, weights=all_intensities)
    counts, _ = np.histogram(all_masses, bins=bins)

    # Calculate bin centers and average intensities
    mass_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Avoid division by zero
    mask = counts > 0
    averaged_intensities = np.zeros_like(summed_intensities, dtype=np.float64)
    averaged_intensities[mask] = summed_intensities[mask] / counts[mask]

    # Remove empty bins
    mask = averaged_intensities > 0
    return mass_centers[mask], averaged_intensities[mask]


def format_spectrum(masses: NDArray[np.float64], intensities: NDArray[np.float64]) -> str:
    """Format spectrum as space-separated 'mass:intensity' pairs"""
    return ' '.join(f"{m:.4f}:{i:.2f}" for m, i in zip(masses, intensities))


def normalize_spectrum(intensities: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize spectrum intensities to 0-100 range"""
    max_intensity = np.max(intensities)
    if max_intensity > 0:
        return (intensities / max_intensity) * 1000
    return intensities


def clean_spectrum(masses: NDArray[np.float64], intensities: NDArray[np.float64]) -> Tuple[
    NDArray[np.float64], NDArray[np.float64]]:
    """Remove zero-intensity ions from the spectrum"""
    mask = intensities > 0.01
    return masses[mask], intensities[mask]


def merge_spectra(masses1: NDArray[np.float64], intensities1: NDArray[np.float64],
                  masses2: NDArray[np.float64], intensities2: NDArray[np.float64],
                  mass_precision: float = 0.01) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Merge two spectra using efficient binning
    """
    # Create unified mass bins
    mass_min = min(np.min(masses1), np.min(masses2))
    mass_max = max(np.max(masses1), np.max(masses2))
    num_bins = int((mass_max - mass_min) / mass_precision) + 1
    bins = np.linspace(mass_min, mass_max, num_bins)

    # Bin both spectra
    intensities1_binned, _ = np.histogram(masses1, bins=bins, weights=intensities1)
    intensities2_binned, _ = np.histogram(masses2, bins=bins, weights=intensities2)

    # Calculate bin centers
    mass_centers = (bins[:-1] + bins[1:]) / 2

    # Average the intensities
    merged_intensities = (intensities1_binned + intensities2_binned) / 2

    # Remove empty bins
    mask = merged_intensities > 0
    return mass_centers[mask], merged_intensities[mask]


def process_and_merge_spectra(spectra1: List[Tuple[NDArray[np.float64], NDArray[np.float64]]],
                              spectra2: List[Tuple[NDArray[np.float64], NDArray[np.float64]]],
                              mass_precision: float = 0.01) -> List[Dict[str, str]]:
    """
    Process and merge spectra from two files using efficient binning, scan by scan
    """
    # Ensure we have equal number of scans or take the minimum
    num_scans = min(len(spectra1), len(spectra2))
    results = []

    with tqdm(total=num_scans, desc="Processing scans", unit="scan") as pbar:
        for scan_idx in range(num_scans):
            # Get current scan pair
            masses1, intensities1 = spectra1[scan_idx]
            masses2, intensities2 = spectra2[scan_idx]

            # Clean and normalize spectra
            masses1, intensities1 = clean_spectrum(masses1, intensities1)
            masses2, intensities2 = clean_spectrum(masses2, intensities2)

            # Skip empty spectra
            if len(masses1) == 0 or len(masses2) == 0:
                continue

            norm_intensities1 = normalize_spectrum(intensities1)
            norm_intensities2 = normalize_spectrum(intensities2)

            # Merge the normalized spectra
            merged_masses, merged_intensities = merge_spectra(
                masses1, norm_intensities1,
                masses2, norm_intensities2,
                mass_precision
            )
            merged_intensities = normalize_spectrum(merged_intensities)

            # Format and store the results for this scan
            scan_result = {
                'CI': format_spectrum(masses1, norm_intensities1),
                'EI': format_spectrum(masses2, norm_intensities2),
                'MERGE': format_spectrum(merged_masses, merged_intensities)
            }
            results.append(scan_result)
            pbar.update(1)

    return results


def read_and_merge_spectra(file1_path: str, file2_path: str, output_path: str,
                           limit: Optional[int] = None) -> None:
    """
    Read and merge spectra from two NetCDF files
    """
    try:
        # Read spectra from both files
        spectra1 = read_cdf_spectra(file1_path, limit)
        spectra2 = read_cdf_spectra(file2_path, limit)

        # Process and merge the spectra scan by scan
        merged_data = process_and_merge_spectra(spectra1, spectra2)

        # Save results
        with tqdm(total=1, desc="Saving results") as pbar:
            df = pd.DataFrame(merged_data)
            df.to_csv(output_path, index=False)
            pbar.update(1)

        print(f"\nSuccessfully created merged spectra file: {output_path}")
        print(f"Processed {len(spectra1)} scans from {file1_path}")
        print(f"Processed {len(spectra2)} scans from {file2_path}")

    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge and normalize spectra from two NetCDF files",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-ci', '--ci_file', required=True, type=str,
                        help='Path to the first NetCDF file (CI spectrum)')
    parser.add_argument('-ei', '--ei_file', required=True, type=str,
                        help='Path to the second NetCDF file (EI spectrum)')
    parser.add_argument('-o', '--output', required=True, type=str,
                        help='Path for the output CSV file')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit processing to first N scans (default: process all)')

    return parser.parse_args()


if __name__ == "__main__":
    args: argparse.Namespace = parse_arguments()
    read_and_merge_spectra(args.ci_file, args.ei_file, args.output, args.limit)
