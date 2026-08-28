#!/usr/bin/env python
"""
fetch_test_data.py
Written by Tyler Sutterley (08/2026)
Download files necessary to run the test suite

CALLING SEQUENCE:
    python fetch_test_data.py

COMMAND LINE OPTIONS:
    --help: list the command line options
    -D X, --directory X: working data directory
    -t X, --timeout X: timeout in seconds for blocking operations
    -M X, --mode X: Local permissions mode of the files downloaded

UPDATE HISTORY:
    Written 08/2026
"""

import io
import shutil
import hashlib
import pathlib
import argparse
import posixpath
import urllib.request

_default_directory = pathlib.Path.cwd()

def fetch_test_data(
    directory: str | pathlib.Path = _default_directory,
    N: int = 324,
    mode: oct = 0o775,
    **kwargs,
):
    """
    Download files necessary to run the test suite

    Parameters
    ----------
    directory: str or pathlib.Path
        Download directory
    N: int, default 324
        Number of nodes for ``mat`` file
    mode: oct, default 0o775
        Permissions mode of output local files
    """
    # create download directory if it doesn't exist
    directory = pathlib.Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True, mode=mode)
    # mat file for number of nodes
    matfile = f"md{N:05d}.mat"
    local = directory.joinpath(matfile)
    # path to test file
    HOST = [
        "https://github.com",
        "gradywright",
        "spherepts",
        "raw",
        "master",
        "nodes",
        "max_determinant",
        matfile,
    ]
    URL = posixpath.join(*HOST)
    # fetch the test data
    from_http(URL, local)
    # change the permissions mode
    local.chmod(mode=mode)
    # return the local file
    return local


# PURPOSE: download a file from a http host
def from_http(
    remote: str,
    local: str | pathlib.Path,
    timeout: int | None = None,
    chunk: int = 16384,
    **kwargs,
):
    """
    Download a file from a ``http`` host

    Parameters
    ----------
    remote: str
        Remote URL to fetch
    local: str or pathlib.Path
        Path to local file
    timeout: int or NoneType, default None
        Timeout in seconds for blocking operations
    chunk: int, default 16384
        Chunk size for transfer encoding
    """
    # convert to absolute path
    local = pathlib.Path(local).expanduser().absolute()
    if local.exists():
        with local.open(mode="rb") as fid:
            local_hash = hashlib.md5(fid.read()).hexdigest()
    else:
        local_hash = ""
    # try downloading from http
    try:
        # Create and submit request.
        request = urllib.request.Request(remote, **kwargs)
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.request.HTTPError as exc:
        raise
    except urllib.request.URLError as exc:
        exc.message = "Check internet connection"
        raise
    else:
        # copy remote file contents to bytesIO object
        remote_buffer = io.BytesIO()
        shutil.copyfileobj(response, remote_buffer, chunk)
        remote_buffer.seek(0)
        # generate checksum hash for remote file
        remote_hash = hashlib.md5(remote_buffer.getvalue()).hexdigest()
        # compare checksums
        if local_hash != remote_hash:
            # store bytes to file using chunked transfer encoding
            remote_buffer.seek(0)
            with local.open(mode="wb") as f:
                shutil.copyfileobj(remote_buffer, f, chunk)


# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Download data for running the test suite"""
    )
    # command line parameters
    # working data directory for location of tide models
    parser.add_argument(
        "--directory",
        "-D",
        type=pathlib.Path,
        default=_default_directory,
        help="Working data directory",
    )
    # number of nodes
    parser.add_argument(
        "--nodes",
        "-n",
        type=int,
        default=324,
        help="Number of nodes for ``mat`` file",
    )
    # connection timeout
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=3600,
        help="Timeout in seconds for blocking operations",
    )
    # permissions mode of the local directories and files (number in octal)
    parser.add_argument(
        "--mode",
        "-M",
        type=lambda x: int(x, base=8),
        default=0o775,
        help="Permissions mode of the files downloaded",
    )
    # return the parser
    return parser


# This is the main part of the program that calls the individual functions
def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args, _ = parser.parse_known_args()

    # fetch test data
    fetch_test_data(
        directory=args.directory,
        N=args.nodes,
        timeout=args.timeout,
        mode=args.mode,
    )


# run main program
if __name__ == "__main__":
    main()
