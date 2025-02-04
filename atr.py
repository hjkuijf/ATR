from collections import OrderedDict
import datetime
import errno
from functools import reduce
import operator
import os
from pathlib import Path
import struct
import typing

from lark import Lark, Transformer
import numpy as np
import pandas

PathLike = typing.Union[str, os.PathLike]


class ATR:
    """Class to load datasets from the ATR dataset."""

    def __init__(self, atr_directory: PathLike, camera: str, scenario: str) -> None:
        """Create a new instance, to load a dataset from disk.

        Parameters:
            atr_directory: root directory where the ATR dataset is stored
            camera: name of the camera to load data from (cegr or i1co)
            scenario: name of the scenario to load

        This initialization function will check whether all required file exist on the disk. That
        includes the agt/arf file with the metadata and video itself. But also the imetrics and
        bbox_met files in the Metric directory, from which the boundingboxes will be determined.
        """
        # Check if the required directories exist
        self._atr_directory = Path(atr_directory)
        self._camera_directory = self._atr_directory / camera
        for directory in [self._atr_directory, self._camera_directory]:
            if not directory.is_dir():
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), directory
                )

        # Check if the required files exist
        self._scenario_agt_filename = (
            self._camera_directory / "agt" / str(camera + scenario + ".agt")
        )
        self._scenario_arf_filename = (
            self._camera_directory / "arf" / str(camera + scenario + ".arf")
        )
        for filename in [
            self._scenario_agt_filename,
            self._scenario_arf_filename,
        ]:
            if not filename.exists():
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), filename
                )

        # For the cegr-camera, we also have metric data stored.
        if "cegr" in camera:
            self._bbox_met_filename = (
                self._atr_directory / "Metric" / str(camera + scenario + ".bbox_met")
            )
            self._imetrics_filename = (
                self._atr_directory / "Metric" / str(camera + scenario + ".imetrics")
            )
            for filename in [
                self._bbox_met_filename,
                self._imetrics_filename,
            ]:
                if not filename.exists():
                    raise FileNotFoundError(
                        errno.ENOENT, os.strerror(errno.ENOENT), filename
                    )

        # Prepare the Lark parser that loads *.agt files.
        with open(
            Path(__file__).parent / "agt_grammar.txt", "r"
        ) as agt_grammar_filename:
            self._agt_parser = Lark(agt_grammar_filename.read(), parser="lalr")

    def _get_metadata(self) -> [dict]:
        # Read the *.agt file, with a Lark parser. This reads the AGT file and returns a Lark.Tree,
        # which is next transformed into a dictionary. This dictionary contains all the meta-data from
        # the video
        with open(self._scenario_agt_filename, "r") as agt_file:
            agt_tree = self._agt_parser.parse(agt_file.read())
        agt: dict = AGTTransformer().transform(agt_tree)

        def filter_frames(args):
            """Helper function that only keeps metadata from frames; and ignores other metadata (like the name of the scenario)."""
            key, _ = args
            return "frame" in key

        # The metadata contains two major sections:
        # sensect: metadata about the sensor (the video frames), like the timestamp
        # tgtsect: metadata about the targets visible in each frame
        # Both sections also contain non-frame related data, which will be filtered out.
        sensect_frames = dict(filter(filter_frames, agt["sensect"].items()))
        tgtsect_frames = self._correct_tgtsect_frames(
            dict(filter(filter_frames, agt["tgtsect"].items()))
        )

        # Assert both dict have the same timestamps
        for key, value in sensect_frames.items():
            # Some videos (and some individual frames) do not have a time. Let's assume these are correct.
            if "time" in tgtsect_frames[key]:
                assert (
                    value["time"] == tgtsect_frames[key]["time"]
                ), f"timestamps do not match for {key}"

        # Merge the targets with the sensor information
        sensect_frames = self._deep_update(sensect_frames, tgtsect_frames)

        # Some videos end with incorrect or corrupted frames. For example: i1co 02012_0001, last frame
        # is from the year 1901. We remove these frames from the end.
        timestamps = [
            sensect_frames[key]["time"].timestamp()
            for key in sensect_frames
            if "frame" in key
        ]
        timestamp_diff = np.diff(timestamps)
        for key, diff in zip(
            reversed(list(sensect_frames.keys())), reversed(timestamp_diff)
        ):
            if diff < 0:
                sensect_frames.pop(key)

        # Get metrics and bounding boxes. NOTE: ONLY FOR CEGR SCENARIOS, but not 02015_0097 (bounding boxes are incorrect)
        if "cegr" in str(self._scenario_agt_filename) and "cegr02015_0097" not in str(self._scenario_agt_filename):
            imetrics: pandas.DataFrame = self._get_imetrics()
            bbox_met: pandas.DataFrame = self._get_bboxmet()

            # Create bounding boxes like we would want them (x, y, w, h)
            for _, frame in sensect_frames.items():
                # Look up the bbox_met information for this frame
                bbox_mets = bbox_met[bbox_met.frame == frame["frame"]]

                # For every target in bboxmet, we will add the bbox to the frame
                for _, row in bbox_mets.iterrows():
                    assert "tgt_" + row.plyid in frame

                    # Create bbox
                    frame["tgt_" + row.plyid]["bbox_x"] = row.upper_left_x
                    frame["tgt_" + row.plyid]["bbox_y"] = row.upper_left_y
                    frame["tgt_" + row.plyid]["bbox_w"] = 2 * (
                        frame["tgt_" + row.plyid]["pixloc"][0] - row.upper_left_x
                    )
                    frame["tgt_" + row.plyid]["bbox_h"] = 2 * (
                        frame["tgt_" + row.plyid]["pixloc"][1] - row.upper_left_y
                    )

        # The annotation of the first frame of 02015_0097 is incorrect.
        if "cegr02015_0097" in str(self._scenario_agt_filename):
            del sensect_frames['frame_1']['tgt_587']

        return sensect_frames

    def get_data(self) -> [dict, np.ndarray]:
        """Read the dataset and return a dict with metadata and an ndarray with the video."""
        metadata = self._get_metadata()
        return (
            metadata,
            self._get_imagedata()[: len(metadata)],
        )  # crop video to modified length

    def _get_imagedata(self) -> [np.ndarray]:
        # Read the video file. 32 bytes header (8 ints), followed by N shorts
        with open(self._scenario_arf_filename, "rb") as video_binary_file:
            data = video_binary_file.read(32)
            magic_num = struct.unpack(">I", data[0:4])[0]
            version = struct.unpack(">I", data[4:8])[0]
            num_rows = struct.unpack(">I", data[8:12])[0]
            num_cols = struct.unpack(">I", data[12:16])[0]
            image_type = struct.unpack(">I", data[16:20])[0]
            num_frames = struct.unpack(">I", data[20:24])[0]
            image_offset = struct.unpack(">I", data[24:28])[0]
            subheader_flags = struct.unpack(">I", data[28:32])[0]

            # There can be secondary headers, so skip to the start of the video data
            video_binary_file.seek(image_offset)

            image_data_raw = np.fromfile(video_binary_file, dtype=">i2")
            image_data = image_data_raw.reshape((-1, num_rows, num_cols))

            assert (
                image_data.shape[0] == num_frames
            ), f"incorrect number of frames read from {self._scenario_arf_filename}"

        return image_data

    def _convert_nf_std_dev(self, value):
        if "NULL" in value:
            return np.nan
        return np.float64(value)

    def _correct_tgtsect_frames(self, tgtsect_frames):
        """Sometimes, the tgtsect_frames have an offset of the video length (1799 or 1874) in the frame numbering."""
        tgtsect_frames_keys = tgtsect_frames.keys()

        # Do we have an offset?
        tgtsect_frames_keys_has_offset = False
        for i in tgtsect_frames_keys:
            # Check for frames > 1900, because i1co files hav 1875 frames
            if int(i.split("_")[1]) > 1900:
                tgtsect_frames_keys_has_offset = True
                break

        offset = 1799 if "cegr" in str(self._scenario_agt_filename) else 1874

        # If so: correct the offset
        if tgtsect_frames_keys_has_offset:
            for i in sorted(tgtsect_frames_keys):
                new_framenumber = int(i.split("_")[1]) - offset
                new_i = "frame_" + str(new_framenumber)
                tgtsect_frames[new_i] = tgtsect_frames.pop(i)
                tgtsect_frames[new_i]["frame"] = new_framenumber

        return tgtsect_frames

    def _get_imetrics(self) -> pandas.DataFrame:
        """Load an *.imetrics file into a pandas DataFrame"""
        with open(self._imetrics_filename, "r") as f:
            headers = f.readline().split()

            if len(headers) < 2:  # Empty file, for the black bodies
                return pandas.DataFrame()

            return pandas.read_csv(
                f,
                sep=",",
                header=None,
                names=headers,
                converters={"nf_std_dev": self._convert_nf_std_dev},
            )

    def _get_bboxmet(self):
        """Load an *.bbox_met file into a pandas DataFrame"""
        headers = [
            "site",
            "na",
            "base",
            "sensor",
            "scen",
            "frame",
            "plyid",
            "snr",
            "null",
            "upper_left_x",
            "upper_left_y",
            "mean_tgt",
            "std_tgt",
            "pot",
            "eff_pot",
            "mean_bkg",
            "std_bkg",
            "pob",
        ]
        with open(self._bbox_met_filename, "r") as f:
            return pandas.read_csv(
                f,
                sep=",",
                header=None,
                names=headers,
                converters={"nf_std_dev": self._convert_nf_std_dev, "plyid": str},
            )

    # Taken from: https://github.com/pydantic/pydantic/blob/fd2991fe6a73819b48c906e3c3274e8e47d0f761/pydantic/utils.py#L200C1-L208C27
    KeyType = typing.TypeVar("KeyType")

    def _deep_update(
        self,
        mapping: typing.Dict[KeyType, typing.Any],
        *updating_mappings: typing.Dict[KeyType, typing.Any],
    ) -> typing.Dict[KeyType, typing.Any]:
        updated_mapping = mapping.copy()
        for updating_mapping in updating_mappings:
            for k, v in updating_mapping.items():
                if (
                    k in updated_mapping
                    and isinstance(updated_mapping[k], dict)
                    and isinstance(v, dict)
                ):
                    updated_mapping[k] = self._deep_update(updated_mapping[k], v)
                else:
                    updated_mapping[k] = v
        return updated_mapping


class AGTTransformer(Transformer):
    """Transform a Lark Tree into a dict, for the ATR dataset."""

    def range(self, args):
        return {"range": float(args[0])}

    def aspect(self, args):
        return {"aspect": float(args[0])}

    def elevation(self, args):
        return {"elevation": float(args[0])}

    def azimuth(self, args):
        return {"azimuth": float(args[0])}

    def fov(self, args):
        return {"fov": (float(args[0]), float(args[1]))}

    def pixloc(self, args):
        return {"pixloc": (int(args[0]), int(args[1]))}

    def comment(self, args):
        """Converts a comment into a dict.

        Some comments contain the frame number. Those are important! These will be returned
        as a frame instead."""
        comment_value = args[0][1:-1]
        if "Frame" in comment_value:
            return {"frame": int(comment_value.split()[1])}
        return {"comment": comment_value}

    def tgttype(self, args):
        return {"tgttype": args[0][1:-1]}

    def name(self, args):
        return {"name": args[0][1:-1]}

    def scenario(self, args):
        return {"scenario": args[0][1:-1]}

    def plyid(self, args):
        """PLYID is supposed to be a str, but in this dataset it can be a number with leading zeros.

        We convert to int and back, because sometimes the data is stored as 054 and sometimes as 54. Treating this
        as a string will say '054' != '54'; but we need these to be equal. So we convert to int to drop
        any leading zeros and then back to str."""
        return {"plyid": str(int(args[0][1:-1]))}

    def keyword(self, args):
        """Converts a keyword into a dict.

        Some keywords contain the frame number. Those are important! These will be returned
        as a frame instead.

        Other keywords will simply be split and the key and value are returned. Sometimes a
        unit is provided for the value, which is ignored. See the documentation for the units
        if you need them."""
        keyword_value = args[0][1:-1]
        if "Frame" in keyword_value:
            return {"frame": int(keyword_value.split()[1])}
        return {
            keyword_value.split()[0]: keyword_value.split()[1]
        }  # keyword_value.split()[2] is the unit of the value -> see documentation

    def senupd(self, args):
        """senupd contains a number of dicts, one for each comment / keyword / parameter.

        All sub-dicts will be flattened into a larger dict. The key for this resulting
        dict is frame_{framenumber}."""
        senupd_dict = reduce(operator.ior, args, {})
        if "frame" in senupd_dict:
            return {"frame_" + str(senupd_dict["frame"]): senupd_dict}

        # If we have no explicit frame numbers, we will base it on timestamp
        return {"time_" + str(senupd_dict["time"].timestamp()): senupd_dict}

    def tgt(self, args):
        tgt_dict = reduce(operator.ior, args, {})
        return {"tgt_" + tgt_dict["plyid"]: tgt_dict}

    def sensect(self, args):
        # The parameter args is a list, which is ordered. We can use this to infer the
        # frame numbers, in case they are not present in the individual senupd.
        sensect_ordered = reduce(operator.ior, args, OrderedDict())

        # Some files do not have frame numbers, but are indexed by time. Convert to frame numbering.
        # This does not have any effect when frame_ frames already exist.
        timestamp_keys = [x for x in sensect_ordered.keys() if "time" in x]
        for i, timestamp_key in enumerate(timestamp_keys):
            sensect_ordered["frame_" + str(i + 1)] = sensect_ordered.pop(timestamp_key)

        return {"sensect": sensect_ordered}

    def tgtupd(self, args):
        tgtupd_dict = reduce(operator.ior, args, {})
        return {"frame_" + str(tgtupd_dict["frame"]): tgtupd_dict}

    def tgtsect(self, args):
        return {"tgtsect": reduce(operator.ior, args, {})}

    def prjsect(self, args):
        return {"prjsect": reduce(operator.ior, args, {})}

    def sect(self, args):
        return reduce(operator.ior, args, {})

    def agt(self, args):
        return args[0]

    def start(self, args):
        return args[0]

    def time(self, args):
        agt_date = datetime.date(int(args[0]), 1, 1) + datetime.timedelta(
            days=int(args[1])
        )
        agt_time = datetime.time(
            int(args[2]), int(args[3]), int(args[4]), int(args[5]) * 1000
        )
        return {"time": datetime.datetime.combine(agt_date, agt_time)}
