import os
import re

from boto.s3 import connect_to_region
from boto.s3.key import Key

import logger

S3_BUCKET_REGIONS = {
    "terraloupe": "eu-central-1",
    "terraloupe-test": "eu-central-1",
    "terraloupe-euwest": "eu-west-1"
}

S3_DEFAULT_BUCKET = 'terraloupe-euwest'


def get_region_for_bucket(bucket):
    if bucket in S3_BUCKET_REGIONS:
        return S3_BUCKET_REGIONS[bucket]
    else:
        return


def connect_s3(bucket):
    region = get_region_for_bucket(bucket)
    if region:
        return connect_to_region(region)
    else:
        logger.warn(
            "No region for bucket '{}' defined, switch to sigv3".format(bucket))
        return S3Connection()


class S3Client:

    def __init__(self, bucket=None, prefix=None):

        # work-around for frankfurt region
        self.conn = connect_s3(bucket)

        self.bucket_name = bucket
        self.prefix = prefix

        self.bucket = self.conn.get_bucket(self.bucket_name)
        if not bucket:
            raise Exception("Could not retrieve s3 bucket '%s'" %
                            self.bucket_name)

    def upload_file(self, filename, key=None):

        if not key:
            key = os.path.basename(filename)

        key = self._get_boto_key_for(key)
        return key.set_contents_from_filename(filename)

    def list_keys(self, prefix, extension=None, directories=False):

        if directories:
            result = self.bucket.list(prefix + "/", "/")
        else:
            result = self.bucket.list(prefix=prefix)

        keys = [k.name for k in result]

        if extension:
            r = re.compile(".+\.%s$" % extension)
            keys = filter(r.match, keys)

        return keys

    def exists(self, key):
        if self.prefix:
            key = os.path.join(self, self.prefix, key)

        return bool(self.bucket.get_key(key))

    def get_file(self, key, filename=None, prefix=None):

        if not filename:
            filename = os.path.basename(key)
        if prefix:
            filename = os.path.join(prefix, filename)

        key = self._get_boto_key_for(key)
        key.get_contents_to_filename(filename)
        return filename

    def _get_boto_key_for(self, key):
        if self.prefix:
            key = os.path.join(self.prefix, key)

        boto_key = Key(self.bucket)
        boto_key.key = key
        return boto_key
