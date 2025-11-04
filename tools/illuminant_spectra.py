import re
import numpy as np
from struct import pack
from os import path


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


def prepare_illuminant_spectrum(output_file, name, wavelengths, fluxs, resolution=1024):
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
    xyz_array = np.loadtxt("../data/ciexyz31.csv", delimiter=",")
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
    rgb_importance /= np.trapezoid(rgb_importance, rgb_wavelengths)
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
    integral = np.trapezoid(fluxs, wavelengths)
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


if __name__ == "__main__":
    for i in range(2469, 2802):
        source_file = "../data/lspdd/csv/%d" % i
        if path.exists(source_file):
            name, wavelengths, fluxs = load_illuminant_csv(source_file)
            prepare_illuminant_spectrum("../data/lspdd/texture/%d.spectrum" % i, name, wavelengths, fluxs)
