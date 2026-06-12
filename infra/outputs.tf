output "query_function_url" {
  description = "Public HTTPS endpoint for the chat widget."
  value       = aws_lambda_function_url.query.function_url
}

output "query_ecr_repo" {
  value = aws_ecr_repository.query.repository_url
}

output "ingest_ecr_repo" {
  value = aws_ecr_repository.ingest.repository_url
}

output "ci_deploy_role_arn" {
  description = "Set as AWS_DEPLOY_ROLE_ARN secret in GitHub Actions."
  value       = aws_iam_role.ci_deploy.arn
}

output "secret_arn" {
  value = aws_secretsmanager_secret.app.arn
}

# ── Custom domain (only meaningful when var.domain_name is set) ─────────────
output "custom_domain_url" {
  description = "HTTPS endpoint once DNS resolves."
  value       = var.domain_name == "" ? "" : "https://${var.domain_name}"
}

output "cloudfront_domain" {
  description = "CloudFront target. Without Route 53, create a CNAME from domain_name to this value."
  value       = local.domain_enabled ? aws_cloudfront_distribution.cdn[0].domain_name : ""
}

output "acm_validation_records" {
  description = "Without Route 53, add these DNS records to validate the ACM certificate (apply blocks until they resolve)."
  value = local.domain_enabled ? [
    for dvo in aws_acm_certificate.cert[0].domain_validation_options : {
      name  = dvo.resource_record_name
      type  = dvo.resource_record_type
      value = dvo.resource_record_value
    }
  ] : []
}
