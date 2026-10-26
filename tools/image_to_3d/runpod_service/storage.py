"""Shared S3-compatible storage configuration for the client and worker."""
import json
import os
from urllib.parse import urlsplit


def setting(name, legacy=None, default=None):
    return os.environ.get(name) or (os.environ.get(legacy) if legacy else None) or default


def bucket_name():
    bucket = setting('S3_BUCKET', 'RUNPOD_VOLUME_ID')
    if not bucket:
        raise ValueError('Set S3_BUCKET (or RUNPOD_VOLUME_ID for volume storage).')
    return bucket


def s3_client():
    import boto3
    from botocore.config import Config
    endpoint = setting('S3_ENDPOINT_URL', 'RUNPOD_S3_ENDPOINT')
    if endpoint:
        parsed = urlsplit(endpoint)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('S3_ENDPOINT_URL must be an HTTPS service endpoint without credentials or query parameters.')
    access = setting('S3_ACCESS_KEY_ID', 'RUNPOD_S3_ACCESS_KEY_ID')
    secret = setting('S3_SECRET_ACCESS_KEY', 'RUNPOD_S3_SECRET_ACCESS_KEY')
    if bool(access) != bool(secret):
        raise ValueError('Supply both S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY, or use the AWS credential chain.')
    credentials = {}
    if access:
        credentials = {'aws_access_key_id': access, 'aws_secret_access_key': secret}
        if os.environ.get('S3_SESSION_TOKEN'):
            credentials['aws_session_token'] = os.environ['S3_SESSION_TOKEN']
    style = os.environ.get('S3_ADDRESSING_STYLE', 'path')
    if style not in ('path', 'virtual', 'auto'):
        raise ValueError('S3_ADDRESSING_STYLE must be path, virtual or auto.')
    return boto3.Session().client('s3', endpoint_url=endpoint,
        region_name=setting('S3_REGION', 'RUNPOD_S3_REGION', 'us-east-1'), **credentials,
        config=Config(signature_version='s3v4', s3={'addressing_style': style},
                      connect_timeout=10, read_timeout=120,
                      retries={'total_max_attempts': 5, 'mode': 'standard'},
                      request_checksum_calculation='when_required', response_checksum_validation='when_required'))


def read_result(storage, bucket, key):
    try:
        response = storage.get_object(Bucket=bucket, Key=key)
    except storage.exceptions.NoSuchKey:
        return None
    with response['Body'] as stream:
        return json.load(stream)
