import numpy as np
import os
import imageio.v2 as imageio
from struct import unpack, calcsize
import zipfile
import re


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


def srgb_to_fourier_srgb_batch(destination_directory, source_directory=None, source_name_re=r"(.*)_BaseColor\.", dest_name_replacement=r"\1_BaseColorFourierSRGB.", skip_existing=True, output_format=".png", lut_path="../data/srgb_to_fourier_srgb.dat"):
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
    # Open the lookup table, either from a file or a zip file
    if os.path.exists(lut_path):
        with open(lut_path, "rb") as lut_file:
            lut = read_lookup_table(lut_file)
    else:
        try:
            with zipfile.ZipFile(lut_path + ".zip") as lut_zip_file:
                with lut_zip_file.open("srgb_to_fourier_srgb.dat") as lut_file:
                    lut = read_lookup_table(lut_file)
        except RuntimeError as e:
            print("Failed to load a lookup table from a zip file. No textures were converted. Please unzip ../data/srgb_to_fourier_srgb.dat.zip manually. The full exception is:")
            print(str(e))
            return
    # Iterate over image files
    for file in os.listdir(source_directory):
        if re.match(source_name_re, file):
            new_file = re.sub(source_name_re, dest_name_replacement, file, 1)
            new_file = os.path.splitext(new_file)[0] + output_format
            new_path = os.path.join(destination_directory, new_file)
            if not (skip_existing and os.path.exists(new_path)):
                srgb_to_fourier_srgb(os.path.join(source_directory, file), new_path, lut)
