import numpy as np
import sys
import os
from struct import pack
from os import path
import imageio.v2 as imageio
from struct import unpack, calcsize
import zipfile
import re

# Illuminant helper functions
def load_illuminant_csv(file_path):
    """
    Loads an illuminant from a comma-separated value file as used by LSPDD.org.
    :return: 1D NumPy arrays (wavelengths, fluxs).
    """
    with open(file_path, "r") as file:
        text = file.read()
        name_re = r"(Category|Brand|Model)\s*:\s*(.*)"
        name = " ".join([match.group(2) for match in re.finditer(name_re, text)])
        name = name.replace(" Standard Illuminant", "")
        name = name.replace(" Biological Sensitivity", "")
        name = name.replace(" Monochromatic", "")
        name = name.replace(" null", "")
        wavelengths = list()
        fluxs = list()
        for match in re.finditer(r"^\s*([0-9].*),([0-9].*)", text, flags=re.MULTILINE):
            wavelengths.append(float(match.group(1)))
            fluxs.append(float(match.group(2)))
    return name, np.asarray(wavelengths), np.asarray(fluxs)


def wavelength_to_phase(wavelength):
    """Converts wavelengths to phases in [-pi, 0] using the XYZ warp."""
    warp_wavelengths = np.linspace(360.0, 830.0, 95)
    warp_phases = [-3.141592654,-3.141592654,-3.141592654,-3.141592654,-3.141591857,-3.141590597,-3.141590237,-3.141432053,-3.140119041,-3.137863071,-3.133438967,-3.123406739,-3.106095749,-3.073470612,-3.024748900,-2.963566246,-2.894461907,-2.819659701,-2.741784136,-2.660533432,-2.576526605,-2.490368187,-2.407962868,-2.334138406,-2.269339880,-2.213127747,-2.162806279,-2.114787412,-2.065873394,-2.012511127,-1.952877310,-1.886377224,-1.813129945,-1.735366957,-1.655108108,-1.573400329,-1.490781436,-1.407519056,-1.323814008,-1.239721795,-1.155352390,-1.071041833,-0.986956525,-0.903007113,-0.819061538,-0.735505101,-0.653346027,-0.573896987,-0.498725202,-0.428534515,-0.363884284,-0.304967687,-0.251925536,-0.205301867,-0.165356255,-0.131442191,-0.102998719,-0.079687644,-0.061092401,-0.046554594,-0.035419229,-0.027113640,-0.021085743,-0.016716885,-0.013468661,-0.011125245,-0.009497032,-0.008356318,-0.007571826,-0.006902676,-0.006366945,-0.005918355,-0.005533442,-0.005193920,-0.004886397,-0.004601975,-0.004334090,-0.004077698,-0.003829183,-0.003585923,-0.003346286,-0.003109231,-0.002873996,-0.002640047,-0.002406990,-0.002174598,-0.001942639,-0.001711031,-0.001479624,-0.001248405,-0.001017282,-0.000786134,-0.000557770,-0.000332262,0.000000000]
    return np.interp(wavelength, warp_wavelengths, warp_phases)


def prepare_illuminant_spectrum(output_file, name, wavelengths, fluxs, csv_path, resolution=1024):
    """
    Prepares an illuminant spectrum for use in a renderer. It ends up being a
    texture that can be queried to perform importance sampling, whilst
    simultaneously querying importance-weighted color matching functions.
    :param output_file: The file to which the prepared spectrum will be
        written.
    :param name: The name of the spectrum.
    :param wavelengths: Wavelengths at which the illuminant is sampled.
    :param fluxs: Corresponding spectral flux for each wavelength.
    :param resolution: The width of the generated texture.
    """

    # Sort the input wavelengths
    shuffle = np.argsort(wavelengths)
    wavelengths, fluxs = wavelengths[shuffle], fluxs[shuffle]
    # Load the color matching functions
    xyz_array = np.loadtxt(csv_path, delimiter=",")
    xyz_wavelengths, xyz = xyz_array[:, 0], xyz_array[:, 1:]
    # Allegedly, you are supposed to treat the XYZ curves as piecewise constant
    spacing = xyz_wavelengths[1] - xyz_wavelengths[0]
    xyz_wavelengths, xyz = [np.repeat(c, 2, 0) for c in [xyz_wavelengths, xyz]]
    xyz_wavelengths[::2] -= 0.4999 * spacing
    xyz_wavelengths[1::2] += 0.4999 * spacing
    # Prepare linear sRGB (Rec. 709) triples for monochromatic light
    xyz_to_rec_709 = np.asarray([
        [+3.2406255, -1.5372080, -0.4986286],
        [-0.9689307, +1.8757561, +0.0415175],
        [+0.0557101, -0.2040211, +1.0569959],
    ])
    rgb_wavelengths = xyz_wavelengths
    rgb = np.einsum("ab, cb -> ca", xyz_to_rec_709, xyz)
    # Derive an importance sampling density from the RGB triples
    rgb_importance = np.sum(np.abs(rgb), axis=1)
    rgb_importance /= np.trapz(rgb_importance, rgb_wavelengths)
    # Prepare the range of random numbers for querying the texture
    rands = np.linspace(0.0, 1.0, resolution + 1)[:-1] + 0.5 / resolution
    # Combine the flux and the RGB importance into a single densely sampled
    # density
    dense_wavelengths = np.linspace(rgb_wavelengths.min(), rgb_wavelengths.max(), 1000001)
    importance = np.interp(dense_wavelengths, wavelengths, fluxs)
    importance *= np.interp(dense_wavelengths, rgb_wavelengths, rgb_importance)
    # Determine the sampled wavelengths for the random numbers
    cdf = np.cumsum(importance)
    sample_wavelengths = np.interp(rands, cdf / cdf[-1], dense_wavelengths)
    # Get the corresponding linear sRGB triples and divide by the RGB
    # importance (but not by the illuminant, because we assume that this
    # cancels with the illuminant in the integrand)
    sample_rgb = np.stack([np.interp(sample_wavelengths, rgb_wavelengths, rgb[:, i]) for i in range(3)], axis=-1)
    sample_rgb /= np.interp(sample_wavelengths, rgb_wavelengths, rgb_importance)[:, np.newaxis]
    # Convert wavelengths to phases using the XYZ warp
    sample_phases = wavelength_to_phase(sample_wavelengths)
    # Compute the integral for the spectrum
    integral = np.trapz(fluxs, wavelengths)
    # Compute aggregate RGB for the spectrum
    total_rgb = np.mean(sample_rgb, axis=0) * integral
    # Create the output file
    with open(output_file, "wb") as file:
        file.write(pack("8s", b"spectrum"))
        version = 0
        file.write(pack("Q", version))
        file.write(pack("255sc", name.encode("utf-8"), b"\0"))
        file.write(pack("I", resolution))
        file.write(pack("fff", *total_rgb))
        file.write(pack("f", integral))
        texture = np.hstack([sample_rgb, sample_phases[:, np.newaxis]])
        texture = np.asarray(texture, dtype=np.float16).view(np.uint16)
        file.write(pack((4 * resolution) * "H", * texture.flat))

# Texture conversion functions
def read_lookup_table(file):
    """
    Reads a 3D lookup table from a binary file-like object using a simple ad
    hoc file format and returns it as array of shape (depth, height, width, 3)
    with uint8 data type.
    """
    header = "QQQQffffff"
    version, width, height, depth, min_x, min_y, min_z, max_x, max_y, max_z = unpack(header, file.read(calcsize(header)))
    cell_count = width * height * depth
    payload = cell_count * "fff"
    entries = unpack(payload, file.read(calcsize(payload)))
    # Turn it into an array of low-dynamic range RGB triples
    lut = np.reshape(entries, (depth, height, width, 3))
    lut = np.asarray(np.minimum(255.0, np.round(lut * 255.0)), dtype=np.uint8)
    return lut


def srgb_to_fourier_srgb(source_file, dest_file, lut):
    """
    Converts a single image file from sRGB to Fourier sRGB.
    :param source_file: Path to the source file.
    :param dest_file: Path to the destination file.
    :param lut: A lookup table as returned by read_lookup_table().
    """
    image = imageio.imread(source_file)
    mapped = lut[image[:, :, 2], image[:, :, 1], image[:, :, 0]]
    imageio.imwrite(dest_file, mapped)


def srgb_to_fourier_srgb_batch(destination_directory, source_directory=None, source_name_re=r"(.*)_BaseColor\.", dest_name_replacement=r"\1_BaseColorFourierSRGB.", skip_existing=True, output_format=".png", lut_path=""):
    """
    This function performs batch conversion of textures using sRGB to the
    Fourier sRGB color space.
    :param destination_directory: Output textures are written to this
        directory. Gets created if it does not exist.
    :param source_directory: The directory that is searched for textures.
        Defaults to the destination directory.
    :param source_name_re: Only files in source_directory whose file name
        matches this regular expression get converted.
    :param dest_name_replacement: A replacement pattern for re.sub() used to
        generate the destination file name from the regex match of the source
        file name.
    :param skip_existing: Pass True to skip files that would require the output
        file to be overwritten. Otherwise, output files are overwritten without
        prompt.
    :param output_format: The file format extension for output textures.
    :param lut_path: Path to the 256^3 lookup table used for the conversion.
        If it does not exist, the function looks for a zipped version of it
        with .zip extension.
    """
    
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    if source_directory is None:
        source_directory = destination_directory

    # Fix: If the path provided is already the zip, use it. 
    # Otherwise, check if the .dat exists, if not, try adding .zip
    actual_zip_path = lut_path if lut_path.endswith('.zip') else lut_path + ".zip"
    
    if os.path.exists(lut_path) and not lut_path.endswith('.zip'):
        with open(lut_path, "rb") as lut_file:
            lut = read_lookup_table(lut_file)
    elif os.path.exists(actual_zip_path):
        try:
            with zipfile.ZipFile(actual_zip_path) as lut_zip_file:
                with lut_zip_file.open("srgb_to_fourier_srgb.dat") as lut_file:
                    lut = read_lookup_table(lut_file)
        except Exception as e:
            print(f"Failed to load lookup table from zip: {e}")
            return
    else:
        print(f"ERROR: Could not find LUT at {lut_path} or {actual_zip_path}")
        return

    # ... (rest of the loop remains the same)
    for file in os.listdir(source_directory):
        if re.match(source_name_re, file):
            new_file = re.sub(source_name_re, dest_name_replacement, file, 1)
            new_file = os.path.splitext(new_file)[0] + output_format
            new_path = os.path.join(destination_directory, new_file)
            if not (skip_existing and os.path.exists(new_path)):
                srgb_to_fourier_srgb(os.path.join(source_directory, file), new_path, lut)

# Main function
if __name__ == "__main__":
    BASE_DATA = r"C:/GitHubFiles/GradWork/HybridSpectralRenderer/data"
    TEXTURE_DIR = r"C:/GitHubFiles/GradWork/HybridSpectralRenderer/data/testscene_textures"
    CSV_XYZ_FILE = os.path.join(BASE_DATA, "ciexyz31.csv")
    LUT_FILE = os.path.join(BASE_DATA, "srgb_to_fourier_srgb.dat.zip")

    print("Starting Texture Conversion...")
    # Pass the zip path directly
    srgb_to_fourier_srgb_batch(TEXTURE_DIR, TEXTURE_DIR, lut_path=LUT_FILE)

    print("Starting Illuminant Conversion...")
    for i in range(2469, 2802):
        source_file = os.path.join(BASE_DATA, "lspdd", "csv", str(i))
        if os.path.exists(source_file):
            output_dir = os.path.join(BASE_DATA, "lspdd", "texture")
            if not os.path.exists(output_dir): os.makedirs(output_dir)
            output_path = os.path.join(output_dir, f"{i}.spectrum")
            name, wavelengths, fluxs = load_illuminant_csv(source_file)
            # Make sure to pass CSV_XYZ_FILE here!
            prepare_illuminant_spectrum(output_path, name, wavelengths, fluxs, CSV_XYZ_FILE)
            print(f"Processed Illuminant: {name}")

    print("--- ALL CONVERSIONS COMPLETE ---")