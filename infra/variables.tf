variable "aws_region" {
  description = "AWS region (must match the Pinecone/DynamoDB region)."
  type        = string
  default     = "us-east-1"
}

variable "project" {
  description = "Resource name prefix."
  type        = string
  default     = "professionalrag"
}

variable "docs_bucket" {
  description = "S3 bucket that triggers ingestion on upload."
  type        = string
  default     = "professionalrag-docs"
}

variable "query_image_uri" {
  description = "ECR image URI (with tag/digest) for the query Lambda. Set by CI."
  type        = string
}

variable "ingest_image_uri" {
  description = "ECR image URI (with tag/digest) for the ingest Lambda. Set by CI."
  type        = string
}

variable "github_repo" {
  description = "owner/name of the GitHub repo allowed to assume the CI deploy role."
  type        = string
  default     = "VikhyatChauhan/ProfessionalRAG"
}

variable "alarm_email" {
  description = "Email subscribed to the CloudWatch alarm SNS topic (optional)."
  type        = string
  default     = ""
}

# ── Custom domain (optional) ───────────────────────────────────────────────
variable "domain_name" {
  description = "Custom domain for the chat API, e.g. chat.vikhyatchauhan.com. Empty disables CloudFront/ACM entirely."
  type        = string
  default     = ""
}

variable "route53_zone_id" {
  description = "Route 53 hosted zone ID for domain_name. If set, ACM validation and the alias record are created automatically; if empty, the records are emitted as outputs for manual DNS entry."
  type        = string
  default     = ""
}

variable "enable_waf" {
  description = "Attach a rate-based AWS WAF web ACL to the CloudFront distribution (replaces slowapi's per-instance limiter). Adds ~$5/mo + per-request cost."
  type        = bool
  default     = false
}

variable "waf_rate_limit" {
  description = "Max requests per 5-minute window per client IP before WAF blocks (when enable_waf = true)."
  type        = number
  default     = 2000
}

variable "waf_blocked_alarm_threshold" {
  description = "Blocked requests in a 5-minute window that trips the WAF alarm (when enable_waf = true)."
  type        = number
  default     = 100
}
