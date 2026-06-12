# ── Shared Lambda execution role ──────────────────────────────────────────
data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda" {
  name               = "${var.project}-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
  tags               = local.tags
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Least-privilege app permissions: read secret, read docs bucket, RW visits table.
data "aws_iam_policy_document" "lambda_app" {
  statement {
    sid       = "ReadSecret"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [aws_secretsmanager_secret.app.arn]
  }
  statement {
    sid       = "ReadDocs"
    actions   = ["s3:GetObject", "s3:HeadObject"]
    resources = ["${aws_s3_bucket.docs.arn}/*"]
  }
  statement {
    sid       = "VisitsTable"
    actions   = ["dynamodb:PutItem", "dynamodb:Query", "dynamodb:Scan", "dynamodb:DescribeTable"]
    resources = [aws_dynamodb_table.visits.arn]
  }
}

resource "aws_iam_role_policy" "lambda_app" {
  name   = "${var.project}-lambda-app"
  role   = aws_iam_role.lambda.id
  policy = data.aws_iam_policy_document.lambda_app.json
}

# Secret is injected at cold start by the app reading SECRETS_ARN. (Pydantic
# Settings still reads individual env vars; a small bootstrap in config can
# hydrate them from this secret — see infra/README.md.)
locals {
  common_env = {
    SECRETS_ARN    = aws_secretsmanager_secret.app.arn
    AWS_REGION_APP = var.aws_region
    EMF_NAMESPACE  = "${var.project}/rag"
  }
}

# ── Query / chat Lambda (hot path) ────────────────────────────────────────
resource "aws_lambda_function" "query" {
  function_name = "${var.project}-query"
  role          = aws_iam_role.lambda.arn
  package_type  = "Image"
  image_uri     = var.query_image_uri
  timeout       = 60
  memory_size   = 512

  environment {
    variables = local.common_env
  }
  tags = local.tags
}

# Function URL with response streaming for the /chat SSE endpoint.
# When a custom domain is configured, lock the URL to AWS_IAM so it is only
# reachable through CloudFront (which signs requests via OAC) — the raw
# *.lambda-url host then returns 403. Without a domain it stays open (NONE) and
# the app's Bearer auth is the only gate.
resource "aws_lambda_function_url" "query" {
  function_name      = aws_lambda_function.query.function_name
  authorization_type = local.domain_enabled ? "AWS_IAM" : "NONE"
  invoke_mode        = "RESPONSE_STREAM"

  cors {
    allow_origins = ["*"]
    allow_methods = ["*"]
    allow_headers = ["authorization", "content-type"]
  }
}

# Public invoke permission for the open (no-domain) path. When a domain is
# configured the URL is AWS_IAM-locked instead and this is replaced by the
# CloudFront-scoped permission in domain.tf.
resource "aws_lambda_permission" "public_invoke" {
  count                  = local.domain_enabled ? 0 : 1
  statement_id           = "AllowPublicInvoke"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = aws_lambda_function.query.function_name
  principal              = "*"
  function_url_auth_type = "NONE"
}

# ── Ingest Lambda (cold path, S3-triggered) ───────────────────────────────
resource "aws_lambda_function" "ingest" {
  function_name = "${var.project}-ingest"
  role          = aws_iam_role.lambda.arn
  package_type  = "Image"
  image_uri     = var.ingest_image_uri
  timeout       = 300
  memory_size   = 1024

  environment {
    variables = local.common_env
  }
  tags = local.tags
}

resource "aws_lambda_permission" "ingest_s3" {
  statement_id  = "AllowS3Invoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.ingest.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.docs.arn
}

resource "aws_s3_bucket_notification" "docs" {
  bucket = aws_s3_bucket.docs.id
  lambda_function {
    lambda_function_arn = aws_lambda_function.ingest.arn
    events              = ["s3:ObjectCreated:*"]
  }
  depends_on = [aws_lambda_permission.ingest_s3]
}
