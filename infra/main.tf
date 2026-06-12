terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  # Configure a remote backend (S3 + DynamoDB lock) before first apply, e.g.:
  # backend "s3" { bucket = "professionalrag-tfstate" key = "infra.tfstate" region = "us-east-1" }
}

provider "aws" {
  region = var.aws_region
}

# CloudFront ACM certificates and CLOUDFRONT-scope WAF must live in us-east-1,
# regardless of the stack's region. Used by domain.tf.
provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"
}

data "aws_caller_identity" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  tags       = { Project = var.project, ManagedBy = "terraform" }
}

# ── Secrets ───────────────────────────────────────────────────────────────
# Values are populated out-of-band (console / CLI), not in code/state.
resource "aws_secretsmanager_secret" "app" {
  name        = "${var.project}/app"
  description = "ANTHROPIC_API_KEY, PINECONE_API_KEY, VOYAGE_API_KEY, ProfessionalRAG_KEY, VISIT_SALT"
  tags        = local.tags
}

# ── Data stores ───────────────────────────────────────────────────────────
resource "aws_dynamodb_table" "visits" {
  name         = "${var.project}-visits"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "pk"
  range_key    = "sk"

  attribute {
    name = "pk"
    type = "S"
  }
  attribute {
    name = "sk"
    type = "S"
  }
  tags = local.tags
}

resource "aws_s3_bucket" "docs" {
  bucket = var.docs_bucket
  tags   = local.tags
}

resource "aws_s3_bucket_public_access_block" "docs" {
  bucket                  = aws_s3_bucket.docs.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ── ECR ───────────────────────────────────────────────────────────────────
resource "aws_ecr_repository" "query" {
  name                 = "${var.project}-query"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
  tags = local.tags
}

resource "aws_ecr_repository" "ingest" {
  name                 = "${var.project}-ingest"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
  tags = local.tags
}
