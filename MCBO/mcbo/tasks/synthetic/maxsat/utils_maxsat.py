import os
import pathlib
import logging

MAXSAT_DIR = "mcbo/tasks/data/maxsat/"


def download_maxsat60_data():
    if not pathlib.Path(MAXSAT_DIR + "frb10-6-4.wcnf").exists():
        logging.info("frb10-6-4.wcnf not found. Downloading...")

        url = (
            "http://bounce-resources.s3-website-us-east-1.amazonaws.com/wms_crafted.tgz"
        )
        logging.info(f"Downloading {url}")

        import requests

        response = requests.get(url, verify=False)

        with open(MAXSAT_DIR + "wms_crafted.tgz", "wb") as file:
            file.write(response.content)

        import tarfile

        with tarfile.open(MAXSAT_DIR + "wms_crafted.tgz", "r:gz") as tar:
            tar.extractall(MAXSAT_DIR)
            # move data/maxsat/wms_crafted/frb/frb10-6-4.wcnf to data/maxsat/frb10-6-4.wcnf
            pathlib.Path(MAXSAT_DIR + "wms_crafted/frb/frb10-6-4.wcnf").rename(
                MAXSAT_DIR + "frb10-6-4.wcnf"
            )
            # delete data/maxsat/wms_crafted (even though it is not empty)
            import shutil

            shutil.rmtree(MAXSAT_DIR + "wms_crafted")
        # delete .tgz file
        pathlib.Path(MAXSAT_DIR + "wms_crafted.tgz").unlink()
        logging.info("Data extracted!")


def download_maxsat125_data():
    if not pathlib.Path(
        MAXSAT_DIR + "cluster-expansion-IS1_5.0.5.0.0.5_softer_periodic.wcnf"
    ).exists():
        logging.info(
            "cluster-expansion-IS1_5.0.5.0.0.5_softer_periodic.wcnf not found. Downloading..."
        )
        import requests

        url = "http://bounce-resources.s3-website-us-east-1.amazonaws.com/mse18-new.zip"
        logging.info(f"Downloading {url}")

        response = requests.get(url, verify=False)

        with open(MAXSAT_DIR + "ce.zip", "wb") as file:
            file.write(response.content)

        import zipfile

        with zipfile.ZipFile(MAXSAT_DIR + "ce.zip", "r") as zip_ref:
            zip_ref.extractall(MAXSAT_DIR)

        # extract data/maxsat/mse18-new/cluster-expansion/benchmarks/IS1_5.0.5.0.0.5_softer_periodic.wcnf.gz
        import gzip, shutil

        with gzip.open(
            MAXSAT_DIR + "mse18-new/cluster-expansion/benchmarks/IS1_5.0.5.0.0.5_softer_periodic.wcnf.gz",
            "rb",
        ) as f_in:
            # save to data/maxsat/cluster-expansion-IS1_5.wcnf
            with open(
                MAXSAT_DIR + "cluster-expansion-IS1_5.0.5.0.0.5_softer_periodic.wcnf",
                "wb",
            ) as f_out:
                shutil.copyfileobj(f_in, f_out)

        shutil.rmtree(MAXSAT_DIR + "mse18-new")

        # delete .zip file
        pathlib.Path(MAXSAT_DIR + "ce.zip").unlink()
        logging.info("Data extracted!")


class WCNF:
    """
    Helper class for reading and parsing WCNF files. Works only for weighted CNF without constraints.

    Attributes:
        weights (list): List of weights for each clause.
        clauses (list): List of clauses.
        nv (int): Number of variables.
    """

    def __init__(self, file_path: str):
        """
        Constructor for WCNF class.

        Args:
            file_path: Path to the WCNF file.
        """
        file_name = pathlib.Path(file_path).name
        file_path = os.path.join(MAXSAT_DIR, file_name)

        weights = []
        clauses = []

        with open(file_path, "r") as f:
            lines = f.readlines()
            lines = [line.strip().replace("\n", " ") for line in lines]
            lines = [
                line.strip().split(" ")[:-1]
                for line in lines
                if line != "" and line[0] != "c" and line[0] != "p"
            ]

            for line in lines:
                weight = int(line[0])
                clause = [int(literal) for literal in line[1:] if len(literal) > 0]
                weights.append(weight)
                clauses.append(clause)

            self.weights = weights
            self.clauses = clauses
            self.nv = max([abs(literal) for clause in clauses for literal in clause])
