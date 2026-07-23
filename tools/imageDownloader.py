import os
import requests
from urllib.parse import urlparse, parse_qs
from PIL import Image, UnidentifiedImageError


def download_tartan_image(url, save_path="tartans"):
    # Parse the URL to extract the 'ref' parameter for the filename
    parsed_url = urlparse(url)
    ref_param = parse_qs(parsed_url.query).get('ref', ['unknown'])[0]
    filename = f"tartan_{ref_param}.png"

    # Create the full file path
    filepath = os.path.join(save_path, filename)

    try:
        print(f"Downloading image to {filepath}... ", end='', flush=True)

        # Send a GET request to the URL
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP errors

        # Write the image content to the file
        with open(filepath, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print("Download complete!")

    except requests.exceptions.RequestException as e:
        print(f"Failed to download the image. Error: {e}")

    try:
        Image.open(filepath)
    except UnidentifiedImageError:
        print("Invalid image! Removed.")
        os.remove(filepath)
        return False
    return True


if __name__ == "__main__":
    import random

    ref_range = range(1, 15494)
    num_downloads = 300
    img_res = (775, 775)
    folder_path = "../training-data/shirting/check/main/tartan check/"

    existing_refs = map(lambda f: f.removeprefix('tartan_').removesuffix('.png'), os.listdir(folder_path))
    ref_range = list(set(ref_range).difference(set(existing_refs)))

    for ref in random.sample(ref_range, min(len(ref_range), num_downloads)):
        target_url = f"https://www.tartanregister.gov.uk/tartanImagePrototype?ref={ref}&width={img_res[0]}&height={img_res[1]}"
        download_tartan_image(target_url, folder_path)
