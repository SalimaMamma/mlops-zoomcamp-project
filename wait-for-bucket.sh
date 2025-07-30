#!/bin/sh

echo "⏳ Waiting for S3 bucket '${S3_BUCKET}' to exist..."

while true; do
  aws --endpoint-url=${AWS_ENDPOINT_URL} s3 ls s3://${S3_BUCKET} > /dev/null 2>&1
  if [ $? -eq 0 ]; then
    echo "✅ Bucket '${S3_BUCKET}' exists!"
    break
  fi
  sleep 2
done
