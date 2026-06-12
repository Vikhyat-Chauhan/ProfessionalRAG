# Custom domain for the query/chat API: ACM cert + CloudFront in front of the
# Lambda Function URL. All resources are gated on var.domain_name, so the whole
# feature is a no-op until you set it.

locals {
  domain_enabled  = var.domain_name != ""
  route53_enabled = local.domain_enabled && var.route53_zone_id != ""
  waf_enabled     = local.domain_enabled && var.enable_waf
  # Function URL is "https://<host>/"; CloudFront origins want the bare host.
  function_url_host = trimsuffix(trimprefix(aws_lambda_function_url.query.function_url, "https://"), "/")
}

# ── ACM certificate (must be in us-east-1 for CloudFront) ──────────────────
resource "aws_acm_certificate" "cert" {
  count             = local.domain_enabled ? 1 : 0
  provider          = aws.us_east_1
  domain_name       = var.domain_name
  validation_method = "DNS"
  tags              = local.tags

  lifecycle {
    create_before_destroy = true
  }
}

# Auto-create the DNS validation records when a Route 53 zone is provided.
resource "aws_route53_record" "cert_validation" {
  for_each = local.route53_enabled ? {
    for dvo in aws_acm_certificate.cert[0].domain_validation_options :
    dvo.domain_name => {
      name   = dvo.resource_record_name
      type   = dvo.resource_record_type
      record = dvo.resource_record_value
    }
  } : {}

  zone_id         = var.route53_zone_id
  name            = each.value.name
  type            = each.value.type
  records         = [each.value.record]
  ttl             = 60
  allow_overwrite = true
}

# Blocks until the cert is issued. With Route 53 that's automatic; without it,
# this waits for you to add the records from the `acm_validation_records` output.
resource "aws_acm_certificate_validation" "cert" {
  count                   = local.domain_enabled ? 1 : 0
  provider                = aws.us_east_1
  certificate_arn         = aws_acm_certificate.cert[0].arn
  validation_record_fqdns = local.route53_enabled ? [for r in aws_route53_record.cert_validation : r.fqdn] : [for dvo in aws_acm_certificate.cert[0].domain_validation_options : dvo.resource_record_name]
}

# ── Managed CloudFront policies ────────────────────────────────────────────
# Don't cache (this is a dynamic API), and forward all viewer headers EXCEPT
# Host — Lambda Function URLs reject a mismatched Host header. This also passes
# the Authorization bearer token straight through.
data "aws_cloudfront_cache_policy" "disabled" {
  name = "Managed-CachingDisabled"
}

data "aws_cloudfront_origin_request_policy" "all_viewer_except_host" {
  name = "Managed-AllViewerExceptHostHeader"
}

# Origin Access Control — CloudFront SigV4-signs every request to the Lambda
# Function URL, which is set to AWS_IAM auth. This makes the raw *.lambda-url
# host unreachable; only CloudFront can invoke the function.
resource "aws_cloudfront_origin_access_control" "lambda" {
  count                             = local.domain_enabled ? 1 : 0
  name                              = "${var.project}-lambda-oac"
  description                       = "Signs CloudFront -> Lambda Function URL requests"
  origin_access_control_origin_type = "lambda"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

# ── CloudFront distribution ────────────────────────────────────────────────
resource "aws_cloudfront_distribution" "cdn" {
  count           = local.domain_enabled ? 1 : 0
  enabled         = true
  is_ipv6_enabled = true
  comment         = "${var.project} chat API"
  aliases         = [var.domain_name]
  web_acl_id      = local.waf_enabled ? aws_wafv2_web_acl.cdn[0].arn : null
  tags            = local.tags

  origin {
    domain_name              = local.function_url_host
    origin_id                = "lambda-query"
    origin_access_control_id = aws_cloudfront_origin_access_control.lambda[0].id

    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  default_cache_behavior {
    target_origin_id         = "lambda-query"
    viewer_protocol_policy   = "redirect-to-https"
    allowed_methods          = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
    cached_methods           = ["GET", "HEAD"]
    cache_policy_id          = data.aws_cloudfront_cache_policy.disabled.id
    origin_request_policy_id = data.aws_cloudfront_origin_request_policy.all_viewer_except_host.id
    compress                 = true
  }

  viewer_certificate {
    acm_certificate_arn      = aws_acm_certificate_validation.cert[0].certificate_arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }
}

# Allow this specific CloudFront distribution to invoke the IAM-locked Function URL.
resource "aws_lambda_permission" "cloudfront_invoke" {
  count                  = local.domain_enabled ? 1 : 0
  statement_id           = "AllowCloudFrontInvoke"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = aws_lambda_function.query.function_name
  principal              = "cloudfront.amazonaws.com"
  source_arn             = aws_cloudfront_distribution.cdn[0].arn
  function_url_auth_type = "AWS_IAM"
}

# Point the domain at CloudFront (Route 53 only; otherwise use the outputs).
resource "aws_route53_record" "alias" {
  for_each = local.route53_enabled ? toset(["A", "AAAA"]) : toset([])
  zone_id  = var.route53_zone_id
  name     = var.domain_name
  type     = each.value

  alias {
    name                   = aws_cloudfront_distribution.cdn[0].domain_name
    zone_id                = aws_cloudfront_distribution.cdn[0].hosted_zone_id
    evaluate_target_health = false
  }
}

# ── Optional rate-based WAF ────────────────────────────────────────────────
resource "aws_wafv2_web_acl" "cdn" {
  count    = local.waf_enabled ? 1 : 0
  provider = aws.us_east_1 # CLOUDFRONT scope must be us-east-1
  name     = "${var.project}-cdn"
  scope    = "CLOUDFRONT"
  tags     = local.tags

  default_action {
    allow {}
  }

  rule {
    name     = "rate-limit"
    priority = 1

    action {
      block {}
    }

    statement {
      rate_based_statement {
        limit              = var.waf_rate_limit
        aggregate_key_type = "IP"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${var.project}-cdn-rate-limit"
      sampled_requests_enabled   = true
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "${var.project}-cdn"
    sampled_requests_enabled   = true
  }
}

# Notify when WAF starts blocking — a sign of rate-limit abuse or an attack.
# CLOUDFRONT-scope WAF metrics are published in us-east-1 with Region="Global";
# the alarm must live there too (and so reference an SNS topic in us-east-1,
# which aws_sns_topic.alerts is when aws_region is the default us-east-1).
resource "aws_cloudwatch_metric_alarm" "waf_blocked" {
  count               = local.waf_enabled ? 1 : 0
  provider            = aws.us_east_1
  alarm_name          = "${var.project}-cdn-waf-blocked"
  namespace           = "AWS/WAFV2"
  metric_name         = "BlockedRequests"
  statistic           = "Sum"
  comparison_operator = "GreaterThanThreshold"
  threshold           = var.waf_blocked_alarm_threshold
  period              = 300
  evaluation_periods  = 1
  alarm_actions       = [aws_sns_topic.alerts.arn]
  treat_missing_data  = "notBreaching"
  tags                = local.tags

  dimensions = {
    WebACL = aws_wafv2_web_acl.cdn[0].name
    Region = "Global"
    Rule   = "ALL"
  }
}
