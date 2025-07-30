resource "aws_s3_bucket" "crypto_mlops_bucket" {
  bucket = var.bucket_name

  tags = {
    Name        = "Crypto MLOps Bucket"
    Environment = "local"
  }
}

resource "aws_s3_bucket_public_access_block" "allow_public" {
  bucket = aws_s3_bucket.crypto_mlops_bucket.id

  block_public_acls   = false
  block_public_policy = false
  ignore_public_acls  = false
  restrict_public_buckets = false
}

# Facultatif : versioning
resource "aws_s3_bucket_versioning" "versioning" {
  bucket = aws_s3_bucket.crypto_mlops_bucket.id

  versioning_configuration {
    status = "Enabled"
  }
}
