import csv
import datetime
from itertools import chain
import json
from pathlib import Path
from PIL import Image
import numpy as np
import skvideo.io
from skimage import exposure
from sklearn import preprocessing
import subprocess
import logging
import sys
import tempfile


logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)


import h5py
from tqdm import tqdm

import atr

# List of all target types present in the data.
tgt_types = [
    "MAN",
    "MAN_KNEE",
    "PICKUP",
    "SUV",
    "BTR70",
    "BRDM2",
    "BMP2",
    "T72",
    "ZSU23",
    "2S3",
    "D20",
    "MTLB",
    "BB_SUB_AMB",
    "BB_AMB",
    "BB_H1",
    "BB_H2",
    "M113",
    "VEHICLE",
]
# A LabelEncoder is used to transform the tgt_types from str to int
le = preprocessing.LabelEncoder()
le.fit(tgt_types)

data_directory = Path("/data/ATR Database/")
output_directory = Path("/data/atr_database_output/")
when = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

data_directory_content = list(
    chain(
        data_directory.glob("cegr/arf/*.arf"),
        data_directory.glob("i1co/arf/*.arf"),
    )
)
## If you want to process just one or a few:
# data_directory_content = list(data_directory.glob("cegr/arf/cegr02015_0097.arf"))
for i in tqdm(data_directory_content):
    camera = i.stem[:4]
    scenario = i.stem[4:]

    logging.info(f"Starting {camera} and {scenario}")

    # Load the data with the ATR module
    atr_data = atr.ATR(data_directory, camera, scenario)
    atr_metadata, atr_imagedata = atr_data.get_data()

    def write_hdf5(directory, image_data):
        """Writes image data into an HDF5 file."""
        with h5py.File(directory / "video.h5", "w") as f:
            f["data"] = image_data

    def write_mp4(directory, image_data):
        """Writes image data into an MP4 file.

        Also performs intensity scaling. Note that the video is also compressed, so only for display purposes.
        """
        image_data_uint8 = exposure.rescale_intensity(
            image_data,
            in_range=(
                np.percentile(image_data, 0.01),
                np.percentile(image_data, 99.99),
            ),
            out_range="uint8",
        )
        skvideo.io.vwrite(directory / "video.mp4", image_data_uint8)

    def write_avi_unscaled(directory, image_data):
        """Writes image data into an AVI file.

        Writes the raw uint16 data into the AVI file, using ffmpeg."""
        image_data = image_data.astype(np.uint16)

        with tempfile.TemporaryDirectory() as temp_dir:
            for idx, frame in enumerate(image_data):
                im_filename = Path(temp_dir.name) / f"{idx:05d}.tif"
                im_pil = Image.fromarray(frame)
                im_pil.save(im_filename)

            output_file = directory / "video.avi"
            command = [
                "ffmpeg",
                "-i",
                f"{temp_dir}/%05d.tif",
                "-c:v",
                "ffv1",
                f"{output_file}",
            ]
            subprocess.run(command, check=True)

    # List with the different types of output. Extend the list with a name + writer function, to add different output types.
    output = []
    output.append(
        {
            "name": str(camera + "_hdf5"),
            "writer": write_hdf5,
        }
    )
    output.append(
        {
            "name": str(camera + "_mp4_rescaled_uint8"),
            "writer": write_mp4,
        }
    )
    output.append(
        {
            "name": str(camera + "_avi_uint16"),
            "writer": write_avi_unscaled,
        }
    )

    for element in output:
        video_directory = (
            output_directory / "video" / scenario / element["name"]
        )
        video_directory.mkdir(parents=True, exist_ok=True)

        # Write the video file
        element["writer"](video_directory, atr_imagedata)

        # Create the log, each line contains the UNIX timestamp of the frame
        with open(video_directory / "video.log", "w") as f:
            for _, frame in sorted(
                atr_metadata.items(), key=lambda x: x[1]["frame"]
            ):
                f.write(str(frame["time"].timestamp()) + "\n")

        results_directory = (
            output_directory / "results" / scenario / element["name"]
        )

        # Write the annotations to a file
        results_annotations_directory = results_directory / "_annotations"
        results_annotations_directory.mkdir(parents=True, exist_ok=True)
        with open(
            results_annotations_directory
            / str("annotations_" + when + ".csv"),
            "w",
        ) as f:
            annotation_writer = csv.writer(f, delimiter=",")
            annotation_writer.writerow(
                [
                    "timestamp",
                    "bbox_x",
                    "bbox_y",
                    "bbox_w",
                    "bbox_h",
                    "class_id",
                    "label",
                    "track_id",
                    "xx",
                    "yy",
                ]
            )
            # NOTE:
            # class_id is a number that represents the classes (see list on top of this file). This is
            # added seperately, since the different MAN in the videos have their own PLYID. Therefore the
            # PLYID is used as a track (representing the individual objects), but their class_id is
            # the same (numerical representation of the class MAN).
            for _, frame in sorted(
                atr_metadata.items(), key=lambda x: x[1]["frame"]
            ):
                for target_id in filter(lambda x: "tgt" in x, frame):
                    # Some tgttypes are mislabelled.
                    tgttype = frame[target_id]["tgttype"]
                    if "BTR" == tgttype:
                        tgttype = "BTR70"

                    if "MAN_KNEE" == tgttype:
                        tgttype = "MAN"

                    if "cegr" == camera:
                        if "02015_0097" == scenario:
                            bbox = (
                                frame[target_id]["pixloc"][0],
                                frame[target_id]["pixloc"][1],
                                1,
                                1,
                            )
                        else:
                            bbox = (
                                frame[target_id]["bbox_x"],
                                frame[target_id]["bbox_y"],
                                frame[target_id]["bbox_w"],
                                frame[target_id]["bbox_h"],
                            )
                    elif "i1co" == camera:
                        # We only have the centre pixel annotated, so create a 1x1 bbox
                        bbox = (
                            frame[target_id]["pixloc"][0],
                            frame[target_id]["pixloc"][1],
                            1,
                            1,
                        )

                    result = []
                    result.append(frame["time"].timestamp())
                    result.append(bbox[0])
                    result.append(bbox[1])
                    result.append(bbox[2])
                    result.append(bbox[3])
                    result.append(le.transform([tgttype])[0])
                    result.append(frame[target_id]["tgttype"])
                    result.append(int(frame[target_id]["plyid"]))
                    result.append(frame[target_id]["pixloc"][0])
                    result.append(frame[target_id]["pixloc"][1])
                    annotation_writer.writerow(result)

        # Dump all metadata in the results video folder
        results_video_directory = results_directory / "video"
        results_video_directory.mkdir(parents=True, exist_ok=True)
        with open(results_video_directory / "metadata.json", "w") as f:
            json.dump(atr_metadata, f, default=str)
