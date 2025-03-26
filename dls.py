import os
import requests

start_bn = 22730533
end_bn = 22799999

base_url = "https://gis1.oit.ohio.gov/ZIPARCHIVES_III/ELEVATION/3DEP/LIDAR/POR/"

download_folder = "downloads"
os.makedirs(download_folder, exist_ok=True)

for bn in range(start_bn, end_bn + 1):
    filename = f"BN{bn}.zip"
    file_url = base_url + filename
    file_path = os.path.join(download_folder, filename)

    print(f"Downloading {file_url}...")

    response = requests.get(file_url, stream=True)

    if response.status_code == 200:
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"Saved {filename} to {download_folder}")
    else:
        print(f"Failed to download {filename} (status code: {response.status_code})")

print("Download complete!")
