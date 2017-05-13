import base64
import hashlib
import os

from fs.osfs import OSFS
from fs.s3fs import S3FS

from objectdetection.utils.s3_client import S3Client, connect_s3, S3_DEFAULT_BUCKET


class S3FSPatched(S3FS):
    ''' Patched S3FS class, to work with eu-central-1 '''
    @property
    def _s3conn(self):
        ''' Pass endpoint to connection object '''
        return connect_s3(self._bucket_name)


def get_fs_by_name(name, bucket=S3_DEFAULT_BUCKET):
    ''' Create abstract filesystem by name

        Args:
            name: Filesystem type (s3, local)
            bucket: bucket to be used for S3

        Return:
            filesystem
    '''
    if name == "local":
        return OSFS("/")
    elif name == "s3":
        s3fs = S3FSPatched(bucket)
        return s3fs
    else:
        RuntimeError("Unsupported filesystem type: " + name)


def transfer_extension(src, dst):
    ''' Add extension from src to dst, if present
    '''

    base, ext = os.path.splitext(src)
    if ext:
        dst = dst + ext

    return dst


def abspath(path):
    ''' Make absolute path and expand home dir (~)
    '''
    return os.path.abspath(os.path.expanduser(path))


class VFS:

    ''' One-directional, virtual filesystem abstraction layer.

        Can define different filesystems for input and output:

        All read operations are forwarded to the input filesystem,
        whereas all write operations are forwarded to the output filesystem.

        Copies are made in one direction, i.e. from input to output

        Args:
            ifs: input filesystem (local, s3)
            ofs: output filesystem (local, s3)
            bucket: s3 bucket
            temp_dir: directory to store temporary files

    '''

    def __init__(self, ifs="local", ofs="local", bucket=S3_DEFAULT_BUCKET, temp_dir="/tmp/vfs"):
        self.ifs_type = ifs
        self.ofs_type = ofs

        self.ifs = get_fs_by_name(ifs, bucket)
        self.ofs = get_fs_by_name(ofs, bucket)

        self.bucket = bucket

        if 's3' in [self.ifs_type, self.ofs_type]:
            self.s3_client = S3Client(bucket=bucket)

        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)
        self.temp_dir = temp_dir

    def exists(self, path):
        ''' Check if path exists [read method]
        '''
        return self.ifs.exists(path)

    def isdir(self, path, ):
        ''' Check if path is directory [read method]
        '''
        return self.ifs.isdir(path)

    def isfile(self, path):
        ''' Check if path is file [read method]
        '''
        return self.ifs.isfile(path)

    def listdir(self, path, wildcard=None):
        ''' List directory content [read method]
        '''
        return self.ifs.listdir(path, wildcard)

    def makedir(self, path, allow_recreate=True, **options):
        ''' Make directory [read method]
        '''
        return self.ofs.makedir(path, allow_recreate=allow_recreate, **options)

    def open(self, name, mode=None):
        ''' Make directory [read or write method, depending on mode]
        '''
        if mode == "w":
            return self.ofs.open(name, mode)
        elif mode == "r":
            name = abspath(name)
            return self.ifs.open(name, mode)
        else:
            RuntimeError("Open mode '%s' not supported" % mode)

    def copy(self, src, dst):
        ''' Copy file from input fs to output fs [read write method]
        '''
        if self.ifs_type == self.ofs_type:
            self.ifs.copy(src, dst)

        elif self.ifs_type == "local" and self.ofs_type == "s3":
            if self.ofs.isdir(dst):
                dst = os.path.join(dst, os.path.basename(src))
            self.s3_client.upload_file(src, dst)

        elif self.ifs_type == "s3" and self.ofs_type == "local":
            if self.ofs.isdir(dst):
                dst = os.path.join(dst, os.path.basename(src))
            self.s3_client.get_file(src, dst)

    def _download_file(self, path, cache=False):
        ''' Download a file from s2, optional caching

            Args:
                path: s2 key
                cache (boolean): With cache=True, the file is cached,
                                 where path is used as cache key.
                                 If the file is requested again (with cache=True),

                                 the first instance of the file is downloaded.
                                 Any calls with cache=False, will not affect
                                 this behaviour
            Return:
                local_path of the downloaded file
        '''
        if cache:
            local_path = self._mkcache_path(path)
            if os.path.exists(local_path):
                print "File found in cache: %s => %s" % (path, local_path)
                # cache hit
                return local_path
        else:
            local_path = self._mktemp_path(path)

        print "Download file from " + self._s3_url(path)
        self.s3_client.get_file(path, local_path)
        return local_path

    def _mkcache_path(self, path):
        name = base64.urlsafe_b64encode(
            hashlib.sha1(path).digest()).replace("=", "u")
        if path:
            name = transfer_extension(path, name)
        return os.path.join(self.temp_dir, name)

    def _mktemp_path(self, path=None):
        name = base64.urlsafe_b64encode(os.urandom(16)).replace("=", "u")
        if path:
            name = transfer_extension(path, name)
        return os.path.join(self.temp_dir, name)

    def _s3_url(self, path):
        return os.path.join("s3://", self.bucket, path)

    def read_path(self, path, cache=False):
        return read_path(self, path, cache)

    def write_path(self, path):
        return write_path(self, path)


class read_path():
    ''' Context manager for abstract reading files.

        If the input fs is remote (s3), the file is downloaded,
        other wise the local path is returned.

        In case of remote read and cache=False, the local_file
        is deleted after the context is exits.

        Example:

            with read_path(vfs, path) as local_path:
                # read the file here

        Args:
            vsf: the VFS instance
            path: path or s3 key to read
            cache: whether to cache downloaded files
    '''

    def __init__(self, vfs, path, cache=False):
        self.path = path
        self.cache = cache
        self.vfs = vfs
        self.remote = (vfs.ifs_type == 's3')

    def __enter__(self):
        if self.remote:
            self.local_path = self.vfs._download_file(self.path, self.cache)
            return self.local_path
        else:
            return self.path

    def __exit__(self, *args):
        if not self.cache and self.remote:
            os.remove(self.local_path)


class write_path():
    ''' Context manager for abstract writing files.

        If the output fs is remote (s3), the file is saved locally and
        uploaded to the remote target, specified by path.
        The local temporary file is deleted after the upload.

        If the output fs is local, the path is returned to perform writes locally.

        Example:

            with write_path(vfs, path) as local_path:
                # write to the local path

        Args:
            vsf: the VFS instance
            path: path or s3 key to read
    '''

    def __init__(self, vfs, path):
        self.path = path
        self.vfs = vfs
        self.remote = (vfs.ofs_type == 's3')

    def __enter__(self):
        if self.remote:
            self.local_path = self.vfs._mktemp_path(self.path)
            return self.local_path
        else:
            return self.path

    def __exit__(self, *args):
        if self.remote:
            print "Upload file to " + self.vfs._s3_url(self.path)
            self.vfs.s3_client.upload_file(self.local_path, self.path)
            os.remove(self.local_path)
